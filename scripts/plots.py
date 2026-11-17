"""Build the README run table and monochrome figures from committed logs."""
import argparse
from pathlib import Path

import numpy as np

from app.server import bins
from twitch.game import FPS
from twitch.run import read

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs/screenshots"
MARK = "<!-- experiment-table -->"
HEAD = "| Run | Arm | Sim. min | Score/min | Reflex % | Median / p95 ms | Teacher / 1k | Wall s |"


def logs():
    return [(p.stem, read(p)) for p in sorted(RESULTS.glob("*.json"))]


def table():
    rows = [HEAD, "|" + "---|" * 8]
    for name, r in logs():
        m = r["metrics"]
        mid, p95 = m["latency_median_ms"], m["latency_p95_ms"]
        latency = "—" if mid is None else f"{mid:.3f} / {p95:.3f}"
        rows.append(
            f"| {name.rsplit('-', 1)[0]} | {r['arm']} | {len(r['ticks']) / FPS / 60:g} "
            f"| {m['score_per_min']:.1f} | {m['share'].get('reflex', 0) * 100:.1f} | {latency} "
            f"| {m['answers_per_1000']:.1f} | {r['wall_s']:.1f} |"
        )
    return "\n".join(rows)


def interval(values, statistic=np.mean):
    """Percentile bootstrap over the units supplied by the caller."""
    if not len(values):
        return None
    rng = np.random.default_rng(1711)
    draws = rng.choice(values, size=(10_000, len(values)), replace=True)
    return tuple(float(q) for q in np.percentile(statistic(draws, axis=1), [2.5, 97.5]))


def study_summary():
    groups = {}
    for name, run in logs():
        groups.setdefault(name.rsplit("-", 1)[0], {})[run["arm"]] = run
    for name, arms in groups.items():
        if "E" not in arms:
            continue
        e = arms["E"]["metrics"]
        b = arms["B"]["metrics"] if "B" in arms else {}
        a = arms["A"]["metrics"] if "A" in arms else {}
        d = arms["D"]["metrics"]["audit_by_epoch"].get("1") if "D" in arms else None
        within_40 = [r["answers"] is not None and r["answers"] <= 40 for r in e["recovery"]]
        recovered = [r["answers"] for r in e["recovery"] if r["answers"] is not None]
        oracle = b.get("post_fifth_score_per_min")
        ratio_b = (e["post_fifth_score_per_min"] / oracle
                   if e["post_fifth_score_per_min"] is not None and oracle and oracle > 0 else None)
        ratio_a = e["score_per_min"] / a["score_per_min"] if a.get("score_per_min", 0) > 0 else None
        print({
            "run": name, "E_to_B_after_fifth_flip": ratio_b, "E_to_A": ratio_a,
            "E_median_ms": e["latency_median_ms"], "E_teacher_per_1000": e["answers_per_1000"],
            "D_first_flip_agreement": d["correct"] / d["n"] if d else None,
            "flips": len(within_40), "recovered_within_40": sum(within_40),
            "recovery_fraction_interval": interval(within_40),
            "median_answers_if_recovered": float(np.median(recovered)) if recovered else None,
            "conditional_median_interval": interval(recovered, np.median),
        })


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "black", "savefig.facecolor": "black", "axes.facecolor": "black",
        "text.color": "white", "axes.edgecolor": "#777777", "axes.labelcolor": "white",
        "xtick.color": "white", "ytick.color": "white", "font.family": "monospace",
    })

    def plot():
        return plt.subplots(figsize=(10, 3.25), layout="constrained")

    def save(fig, name):
        fig.savefig(OUT / (name + ".png"), dpi=125)
        plt.close(fig)

    data = dict(logs())
    arms = {key[-1]: r for key, r in data.items() if key.startswith("smoke-17-")}
    a, e = arms["A"], arms["E"]

    fig, ax = plot()
    for name, times, shade in [
        ("LLM", [x["ms"] for x in a["answers"]], "white"),
        ("Reflex", [t[2] for t in e["ticks"] if t[2] is not None], "#aaaaaa"),
    ]:
        values = np.sort(np.maximum(times, 0.001))
        ax.step(values, (np.arange(len(values)) + 1) / len(values), label=name, color=shade)
    ax.set(xscale="log", xlabel="Measured latency (ms, log scale)", ylabel="Fraction of decisions")
    ax.legend(facecolor="black")
    save(fig, "lag")

    fig, ax = plot()
    rows = bins(e)
    ax.plot([0, 1], [0, 1], "--", color="#777777", lw=1)
    ax.scatter([r["p"] for r in rows], [r["agreement"] for r in rows],
               s=[15 + 3 * r["n"] for r in rows], marker="o", facecolors="none", edgecolors="white")
    for r in rows:
        ax.annotate(str(r["n"]), (r["p"], r["agreement"]), xytext=(6, 3), textcoords="offset points", color="white")
    ax.set(xlim=(0, 1.03), ylim=(0, 1.03), xlabel="Calibrated confidence", ylabel="Held-out agreement")
    save(fig, "agreement")

    fig, ax = plot()
    driver = [t[1] == "reflex" for t in e["ticks"]]
    trailing = [sum(driver[max(0, n - 89):n + 1]) / min(n + 1, 90) for n in range(len(driver))]
    ax.plot(np.arange(len(driver)) / FPS, trailing, lw=1.2, color="white")
    for flip in e["metrics"]["recovery"]:
        ax.axvline(flip["frame"] / FPS, ls="--", lw=1, color="#aaaaaa")
    ax.set(xlabel="Simulation time (s), dashed lines mark rule flips", ylim=(-.03, 1.04),
           ylabel="Reflex frame share\ntrailing 3 s")
    save(fig, "handoff")

    fig, ax = plot()
    for index, flip in enumerate(e["metrics"]["recovery"], 1):
        start = flip["frame"]
        answers = (a for a in e["answers"] if start <= a["frame"] <= a["at"] < start + e["flip_frames"])
        counts, agreement, audits = [], [], []
        for count, answer in enumerate(answers, 1):
            if answer["audit"]:
                audits.append(answer["agreed"])
                if len(audits) >= 10:
                    counts.append(count)
                    agreement.append(sum(audits[-10:]) / 10)
        shade = str(1 - 0.16 * (index - 1))
        ax.plot(counts, agreement, marker="o", ms=4 + index, color=shade,
                markerfacecolor="none", label=f"Flip {index}")
    ax.axhline(.9, color="#777777", ls="--", lw=1)
    ax.set(xlabel="Teacher answers since flip", ylabel="Agreement\nlast 10 audits", ylim=(-.03, 1.07))
    ax.legend(facecolor="black", ncol=4, loc="lower right")
    save(fig, "recovery")

    fig, ax = plot()
    scores = [r["metrics"]["score_per_min"] for r in arms.values()]
    bars = ax.bar(list(arms), scores, color=["white" if k == "E" else "#777777" for k in arms])
    ax.bar_label(bars, fmt="%.1f", color="white", padding=4)
    ax.margins(y=.17)
    ax.axhline(0, lw=1, color="#aaaaaa")
    ax.set(xlabel="Smoke seed 17 arms", ylabel="Score / minute")
    save(fig, "results")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--table-only", action="store_true")
    cli.add_argument("--study-summary", action="store_true")
    args = cli.parse_args()
    if args.study_summary:
        study_summary()
        return
    path = ROOT / "README.md"
    before, _, after = path.read_text().split(MARK)
    path.write_text(before + MARK + "\n\n" + table() + "\n\n" + MARK + after)
    if not args.table_only:
        figures()


if __name__ == "__main__":
    main()
