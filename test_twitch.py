"""Run the two-minute smoke scenario using real cached Qwen answers by default."""
import argparse
import re
from pathlib import Path

from app.server import play
from scripts.plots import table
from twitch.game import FPS, Game, RULES
from twitch.run import read, run, write
from twitch.teacher import frame_text

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RULE = "|".join(map(re.escape, RULES))
ITEM = r"lane [1-5] (?:blue|mint) (?:circle|square)"
DROP = rf"{ITEM}(?:, {ITEM})*"
DOCUMENT = (rf"Rule: (?:{RULE})\. Ship: lane [1-5] of 5\. First drop: {DROP}\."
            rf"(?: Second drop: {DROP}\.)? Drop times: \d+(?:, \d+)? frames\.")


def verify(data):
    frames = len(data["ticks"])
    game = Game(data["seed"], data["flip_frames"])
    snapshots = {}
    for target, *_ in data["ticks"]:
        frame = game.frame()
        snapshots[frame.number] = frame
        game.step(target)
    answers = data["answers"]
    assert len(set(a["frame"] for a in answers)) == len(answers), "Duplicate frozen frame"
    assert len(set(a["at"] for a in answers)) == len(answers), "Duplicate delivery tick"
    for a in answers:
        frame = snapshots[a["frame"]]
        assert 0 <= a["frame"] <= a["at"] < frames
        assert re.fullmatch(DOCUMENT, a["text"]), a["text"]
        assert a["text"].startswith(f"Rule: {RULES[frame.rule]}. Ship: lane {frame.lane + 1} of 5.")
        assert a["text"] == frame_text(frame)
        assert abs(sum(a["p"]) - 1) < 2e-5
    if data["arm"] == "B":
        assert len(answers) == frames and all(a["at"] == a["frame"] for a in answers)
        assert data["wall_s"] * 1000 >= sum(a["ms"] for a in answers) - 2
    if data["arm"] == "E":
        flip = data["flip_frames"]
        assert [r["frame"] for r in data["metrics"]["recovery"]] == list(range(flip, frames, flip))
        assert all(data["ticks"][n][4] is None for n in range(flip, frames, flip))
        assert all(t[4] is not None and t[3] + 0.000001 >= t[4]
                   for t in data["ticks"] if t[1] == "reflex")
    assert list(play(data))[-1]["score"] == data["metrics"]["score"] == game.score


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--live", action="store_true", help="measure a new CPU smoke run, without replacing checked-in logs")
    cli.add_argument("--teacher", default="http://127.0.0.1:8081")
    cli.add_argument("--outdir", type=Path, default=Path(".smoke"))
    cli.add_argument("--all", action="store_true", help="rebuild all committed runs with their cached teacher arrivals")
    args = cli.parse_args()
    paths = sorted(RESULTS.glob("*.json" if args.all else "smoke-17-[ABE].json"))
    if args.live:
        args.outdir.mkdir(parents=True, exist_ok=True)
        for arm in "AEB":
            data = run(arm, 17, FPS * 120, url=args.teacher)
            verify(data)
            path = args.outdir / f"smoke-17-{arm}.json"
            write(path, data)
            print(f"{arm} {data['wall_s']} s elapsed, score/min {data['metrics']['score_per_min']}", flush=True)
    else:
        assert {f"smoke-17-{arm}.json" for arm in "ABE"} <= {path.name for path in paths}, "Missing A/B/E smoke logs"
        for path in paths:
            data = read(path)
            assert data["arm"] in "ABCDEFG"
            training = read(path.with_name(path.stem[:-1] + "B.json")) if data["arm"] == "D" else None
            rebuilt = run(data["arm"], data["seed"], len(data["ticks"]), data["flip_frames"],
                          training=training, cached=data)
            verify(rebuilt)
            # CPU kernels can round confidence differently. All discrete outcomes must match.
            for actual, expected in zip(rebuilt["answers"], data["answers"], strict=True):
                assert actual.keys() == expected.keys(), path.name
                for key, value in actual.items():
                    if key == "confidence" and value is not None:
                        assert abs(value - expected[key]) <= 0.002, path.name
                    else:
                        assert value == expected[key], path.name
            for actual, expected in zip(rebuilt["ticks"], data["ticks"], strict=True):
                assert actual[:3] == expected[:3] and actual[4:] == expected[4:], path.name
                if actual[3] is None:
                    assert expected[3] is None, path.name
                else:
                    assert abs(actual[3] - expected[3]) <= 0.002, path.name
            assert rebuilt["metrics"] == data["metrics"], path.name
            print(f"Reproduced {path.name}", flush=True)
        assert table() in (ROOT / "README.md").read_text()
    print("Smoke assertions passed", flush=True)


if __name__ == "__main__":
    main()
