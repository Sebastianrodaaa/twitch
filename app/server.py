"""Local dashboard. Replays step the recorded decisions through the same game."""
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from aiohttp import web

from twitch.game import FLIP, FPS, RULES, Game
from twitch.run import ARMS, read, scene

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
NAME = r"[a-z0-9-]+"
LIVE = web.AppKey("live", tuple)
LIVE_LOCK = web.AppKey("live_lock", asyncio.Lock)


def recordings():
    return sorted(p.stem[:-2] for p in RESULTS.glob("*-E.json")
                  if re.fullmatch(NAME, p.stem[:-2]) and (RESULTS / (p.stem[:-1] + "A.json")).is_file())


def play(data):
    game = Game(data["seed"], data["flip_frames"])
    answers = {a["at"]: a for a in data["answers"]}
    last_ms = None
    for n, tick in enumerate(data["ticks"]):
        answer = answers.get(n)
        if answer:
            last_ms = answer["ms"]
        event = game.step(tick[0])
        yield scene(game, event, tick, answer, last_ms)


def bins(data):
    rows = [[] for _ in range(10)]
    for a in data["answers"]:
        if a["audit"] and a["confidence"] is not None and a["frame"] // data["flip_frames"] == a["at"] // data["flip_frames"]:
            rows[min(9, int(a["confidence"] * 10))].append(a)
    return [{"p": sum(a["confidence"] for a in row) / len(row),
             "agreement": sum(a["agreed"] for a in row) / len(row), "n": len(row)} for row in rows if row]


def summary(name):
    rows = {}
    for arm in ARMS:
        path = RESULTS / f"{name}-{arm}.json"
        if path.is_file():
            rows[arm] = read(path)["metrics"]
    return rows


async def index(request):
    return web.FileResponse(ROOT / "app/index.html")


async def run_list(request):
    return web.json_response(recordings())


async def replay(request):
    name = request.match_info["name"]
    if name not in recordings():
        raise web.HTTPNotFound()
    left, right = (read(RESULTS / f"{name}-{arm}.json") for arm in "AE")
    if (any(left[k] != right[k] for k in ("seed", "flip_frames", "fps"))
            or len(left["ticks"]) != len(right["ticks"])):
        raise web.HTTPConflict(text="Recording metadata does not match")
    response = web.json_response({
        "seed": right["seed"], "flip": right["flip_frames"], "rules": RULES, "fps": FPS,
        "rows": list(zip(play(left), play(right))), "results": summary(name), "agreement": bins(right),
    })
    response.enable_compression()
    return response


async def live(request):
    app = request.app
    # ponytail: one local comparison at a time; add a worker queue for shared serving.
    lock = app[LIVE_LOCK]
    if lock.locked():
        raise web.HTTPConflict(text="A live run is active")
    ws = web.WebSocketResponse()
    await lock.acquire()
    processes = []
    try:
        await ws.prepare(request)
        seconds, url = app[LIVE]
        for arm in "AE":
            processes.append(await asyncio.create_subprocess_exec(
                sys.executable, "-m", "twitch.run", "--arm", arm, "--seed", "17",
                "--seconds", str(seconds), "--teacher", url, "--stream",
                stdout=asyncio.subprocess.PIPE, cwd=ROOT,
            ))
        await ws.send_json({"seed": 17, "flip": FLIP, "rules": RULES, "fps": FPS,
                            "total": round(seconds * FPS)})
        while not ws.closed:
            lines = await asyncio.gather(*(p.stdout.readline() for p in processes))
            if not all(lines):
                await ws.send_json({"error": "Live run stopped. Check the local llama.cpp server log."})
                break
            pair = [json.loads(line) for line in lines]
            if pair[0].get("type") == "result":
                await ws.send_json({"results": dict(zip("AE", pair))})
                break
            await ws.send_json({"row": pair})
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        for p in processes:
            if p.returncode is None:
                try:
                    p.terminate()
                except ProcessLookupError:
                    pass
        await asyncio.gather(*(p.wait() for p in processes))
        lock.release()
        await ws.close()
    return ws


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--port", type=int, default=8000)
    cli.add_argument("--host", default="127.0.0.1")
    cli.add_argument("--teacher", default=os.getenv("TWITCH_TEACHER", "http://127.0.0.1:8081"))
    cli.add_argument("--seconds", type=float, default=120)
    args = cli.parse_args()
    if round(args.seconds * FPS) < 1:
        cli.error("Live duration must include at least one frame")
    app = web.Application()
    app[LIVE_LOCK] = asyncio.Lock()
    app[LIVE] = (args.seconds, args.teacher)
    app.router.add_get("/", index)
    app.router.add_get("/api/runs", run_list)
    app.router.add_get("/api/replay/{name}", replay)
    app.router.add_get("/ws", live)
    app.router.add_static("/fonts", ROOT / "app/fonts")
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
