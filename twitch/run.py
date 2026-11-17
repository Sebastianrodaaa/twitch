"""Run the experiment on CPU, or re-execute it with recorded answer arrivals."""
import argparse
import json
import os
import platform
from collections import Counter
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import torch

from twitch.game import FLIP, FPS, Game, heuristic
from twitch.reflex import Reflex
from twitch.teacher import Queue, beveridge, frame_text, qwen, timed_score

ARMS = {"A": "LLM only", "B": "Paused teacher", "C": "Rule heuristic", "D": "Frozen reflex",
        "E": "twitch", "F": "Beveridge", "G": "Random"}
TICK = ("target", "driver", "reflex_ms", "confidence", "threshold", "lag_ms")
MINIMUM = 0.25


def teacher_label(frame, probabilities):
    p = list(probabilities)
    for action in (0, 2):
        if frame.target(action) == frame.lane:
            p[1] += p[action]
            p[action] = 0.0
    return int(np.argmax(p))


def fit_offline(reflex, data):
    if data["arm"] != "B":
        raise ValueError("Frozen reflex training expects a paused-teacher run")
    game = Game(data["seed"], data["flip_frames"])
    rows = {answer["frame"]: answer for answer in data["answers"]}
    for tick in data["ticks"][:game.flip:]:
        frame = game.frame()
        answer = rows.get(frame.number)
        if answer and frame.number % 4 == 0:
            x = reflex.inputs(frame)
            reflex.observe(x, answer["p"], None, frame, frame.epoch, False)
        game.step(tick[0])


def recovery(answers, flip, frames):
    rows = []
    for n in range(flip, frames, flip):
        current = [a for a in answers if n <= a["frame"] <= a["at"] < min(n + flip, frames)]
        audits = []
        recovered = None
        for i, answer in enumerate(current, 1):
            if answer["audit"]:
                audits.append(answer.get("agreed"))
                if len(audits) >= 10 and sum(audits[-10:]) >= 9:
                    recovered = (i, (answer["at"] - n + 1) / FPS)
                    break
        rows.append({"frame": n, "answers": recovered[0] if recovered else None,
                     "seconds": recovered[1] if recovered else None, "observed_answers": len(current),
                     "audits": sum(a["audit"] for a in current)})
    return rows


def audit_stats(answers, flip):
    epochs = {}
    for answer in answers:
        if answer["audit"] and answer["agreed"] is not None and answer["frame"] // flip == answer["at"] // flip:
            epochs.setdefault(str(answer["frame"] // flip), []).append(answer["agreed"])
    return {epoch: {"correct": sum(values), "n": len(values)} for epoch, values in epochs.items()}


def stats(game, arm, ticks, answers, scores):
    frames = len(ticks)
    drivers = Counter(t[1] for t in ticks)
    latency = [t[2] for t in ticks if t[2] is not None]
    if not latency:
        latency = [a["ms"] for a in answers]
    post = game.flip * 5

    def percentile(values, p):
        return round(float(np.percentile(values, p)), 3) if values else None

    return {
        "score": game.score, "score_per_min": game.score * 60 * FPS / frames,
        "post_fifth_score_per_min": (game.score - scores[post - 1]) * 60 * FPS / (frames - post)
        if frames > post else None,
        "catches": game.catches, "crashes": game.crashes, "wipeouts": game.wipeouts,
        "share": {k: v / frames for k, v in sorted(drivers.items())},
        "latency_median_ms": percentile(latency, 50), "latency_p95_ms": percentile(latency, 95),
        "teacher_median_ms": percentile([a["ms"] for a in answers] if arm in "ABDE" else [], 50),
        "answers_per_1000": 1000 * len(answers) / frames if arm in "ABDE" else 0,
        "late_frames": sum(t[5] > (1000 / FPS) for t in ticks),
        "recovery": recovery(answers, game.flip, frames) if arm == "E" else [],
        "audit_by_epoch": audit_stats(answers, game.flip),
    }


def scene(game, event, tick, answer, last_ms):
    return {
        "lane": game.lane, "score": game.score, "hearts": game.hearts,
        "items": [[i.lane, i.shape, i.color, i.hit - game.number, i.travel] for i in game.items],
        "event": event, "tick": tick, "answer": bool(answer), "teacher_ms": last_ms,
    }


def run(arm, seed, frames, flip=FLIP, url="http://127.0.0.1:8081", training=None,
        cached=None, emit=None, fast=False):
    if frames < 1 or flip < 1:
        raise ValueError("A run needs positive frame and flip counts")
    if fast and arm not in "CG":
        raise ValueError("Unpaced execution is only defined for C and G")
    torch.set_num_threads(1)
    game = Game(seed, flip)
    rng = np.random.default_rng(seed + 400)
    reflex = Reflex(seed) if arm in "DE" else None
    if arm == "D":
        fit_offline(reflex, training)
    scorer = None
    if cached is None and arm in "ABDEF":
        scorer = beveridge() if arm == "F" else qwen(url)
    queue = None
    if scorer and arm != "B":
        minimum = 1.0 if arm == "A" else MINIMUM if arm in "DE" else 0
        queue = Queue(scorer, minimum, budget_frames=flip if arm == "E" else None)
    arrivals = {a["at"]: a for a in cached["answers"]} if cached else {}
    snapshots = {}
    ticks, answers, scores = [], [], []
    last_ms = None
    held = game.lane
    event = ""
    epoch = 0
    started = perf_counter()
    try:
        for n in range(frames):
            if cached is None and arm != "B" and not fast:
                sleep(max(0, started + n / FPS - perf_counter()))
            lag = max(0, (perf_counter() - started - n / FPS) * 1000) if cached is None and arm != "B" and not fast else 0
            frame = game.frame()
            if arm == "E" and epoch != frame.epoch:
                epoch = frame.epoch
                reflex.reset()
            threshold = reflex.threshold if arm == "E" else None
            if reflex:
                t0 = perf_counter()
                x = reflex.inputs(frame)
                act, prob, raw = reflex.decide(x, frame)
                confidence = float(prob[act])
                latency = (perf_counter() - t0) * 1000
            else:
                act, confidence, raw, x, latency = None, None, None, None, None
            snapshots[n] = (frame, x, raw, confidence)
            snapshots.pop(n - 3000, None)
            result = None
            if cached:
                if n in arrivals:
                    a = arrivals[n]
                    result = a["frame"], a["text"], a["audit"], a["p"], a["ms"]
            elif arm == "B":
                text = frame_text(frame)
                # The game tick cannot advance until the scoring call returns.
                p, ms = timed_score(scorer, text)
                result = n, text, False, p, ms
            elif queue and (reply := queue.poll()):
                q, p, ms = reply
                result = q.frame.number, q.text, q.audit, p, ms

            answer = None
            if result:
                f, text, audit, p, last_ms = result
                past, past_x, past_raw, past_confidence = snapshots[f]
                agreed = None if past_raw is None else int(past_raw.argmax()) == teacher_label(past, p)
                answer = {"frame": f, "at": n, "text": text, "audit": audit, "p": p,
                          "ms": last_ms, "agreed": agreed, "confidence": past_confidence}
                answers.append(answer)
                if past.epoch == frame.epoch:
                    held = past.target(teacher_label(past, p))
                if arm == "E":
                    reflex.observe(past_x, p, past_raw, past, epoch, audit)
            if queue:
                if arm == "E":
                    queue.offer(frame, confidence, event in ("crash", "wipeout"))
                queue.submit(frame, arm == "E")

            if arm == "D" or (threshold is not None and confidence >= threshold):
                target, driver = frame.target(act), "reflex"
            elif arm == "C":
                t0 = perf_counter()
                target, driver = frame.target(heuristic(frame)), "heuristic"
                latency = (perf_counter() - t0) * 1000
            elif arm == "G":
                t0 = perf_counter()
                target, driver = frame.target(int(rng.integers(3))), "random"
                latency = (perf_counter() - t0) * 1000
            else:
                target, driver = held, "teacher"
                if arm == "F":
                    driver = "beveridge"
            if cached:
                latency, lag = cached["ticks"][n][2], cached["ticks"][n][5]
            tick = [target, driver, round(latency, 4) if latency is not None else None,
                    round(confidence, 6) if confidence is not None else None, threshold, round(lag, 3)]
            event = game.step(target)
            ticks.append(tick)
            scores.append(game.score)
            if emit:
                emit(scene(game, event, tick, answer, last_ms))
        wall = perf_counter() - started
    finally:
        if queue:
            queue.close()
    return {
        "version": 1, "arm": arm, "seed": seed, "fps": FPS, "flip_frames": flip, "tick_fields": TICK,
        "wall_s": round(wall, 3) if cached is None else cached["wall_s"],
        "platform": platform.platform() if cached is None else cached["platform"],
        "torch": str(torch.__version__), "teacher_url": "local llama.cpp" if arm in "ABDE" else None,
        "ticks": ticks, "answers": answers, "metrics": stats(game, arm, ticks, answers, scores),
    }


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    os.replace(tmp, path)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--arm", choices=ARMS, default="E")
    cli.add_argument("--seed", type=int, default=17)
    cli.add_argument("--seconds", type=float, default=120)
    cli.add_argument("--flip-seconds", type=float, default=25)
    cli.add_argument("--teacher", default="http://127.0.0.1:8081")
    cli.add_argument("--training", help="arm B training log for arm D")
    cli.add_argument("--cached", help="re-execute an existing log using recorded teacher arrivals")
    cli.add_argument("--out", type=Path)
    cli.add_argument("--stream", action="store_true")
    cli.add_argument("--fast", action="store_true", help="skip wall-clock pacing for heuristic and random arms")
    args = cli.parse_args()
    training = read(args.training) if args.training else None
    cached = read(args.cached) if args.cached else None
    arm, seed, frames, flip = ((cached["arm"], cached["seed"], len(cached["ticks"]), cached["flip_frames"])
                               if cached else (args.arm, args.seed, round(args.seconds * FPS),
                                               round(args.flip_seconds * FPS)))
    if arm == "D" and training is None:
        cli.error("arm D requires --training")
    data = run(arm, seed, frames, flip, args.teacher, training, cached,
               (lambda s: print(json.dumps(s), flush=True)) if args.stream else None, args.fast)
    if args.out:
        write(args.out, data)
    print(json.dumps({"type": "result", **data["metrics"]}))


if __name__ == "__main__":
    main()
