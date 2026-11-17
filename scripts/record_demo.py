"""Capture the actual replay UI as 1080p footage with a 1× simulation timebase."""
import argparse
import io
import json
import os
import subprocess
from pathlib import Path
from urllib.request import urlopen

from PIL import Image
from playwright.sync_api import sync_playwright

SIZES = {"16x9": (1920, 1080), "9x16": (1080, 1920)}
ROOT = Path(__file__).resolve().parents[1]


def capture(page, frame):
    page.evaluate("frame => window.captureAt(frame)", frame)
    return page.screenshot(type="jpeg", quality=91, animations="disabled")


def movie(page, out, fps, start, end, scoreboard):
    tmp = out.with_suffix(".tmp.mp4")
    with subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps),
        "-vcodec", "mjpeg", "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-threads", "2", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp),
    ], stdin=subprocess.PIPE) as ffmpeg:
        for frame in range(start, end):
            ffmpeg.stdin.write(capture(page, frame))
        card = capture(page, "results")
        for _ in range(round(scoreboard * fps)):
            ffmpeg.stdin.write(card)
        ffmpeg.stdin.close()
        if ffmpeg.wait():
            raise RuntimeError(f"ffmpeg failed while encoding {out.name}")
    tmp.replace(out)


def gif(movie_path, out):
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-filter_complex_threads", "1", "-i", str(movie_path),
        "-filter_complex", "fps=8,scale=688:-2:flags=lanczos,format=gray,split[s][p];"
        "[p]palettegen=max_colors=96[pal];[s][pal]paletteuse=dither=bayer:bayer_scale=4", str(out),
    ], check=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--base", default="http://127.0.0.1:8000")
    cli.add_argument("--run", default="smoke-17")
    cli.add_argument("--out", type=Path, default=ROOT / "docs/screenshots")
    cli.add_argument("--start", type=float, default=0, help="game time in seconds of first frame")
    cli.add_argument("--end", type=float, help="exclusive game time in seconds, defaults to end of run")
    cli.add_argument("--scoreboard", type=float, default=4, help="seconds to hold the full-run scoreboard")
    cli.add_argument("--aspect", choices=[*SIZES, "both"], default="both")
    cli.add_argument("--chrome", default=os.getenv("CHROME"), help="system Chrome path, or use Playwright Chromium")
    cli.add_argument("--stills", action="store_true", help="capture only representative grayscale dashboard images")
    cli.add_argument("--gif", action="store_true", help="make a grayscale GIF from the 16:9 recording")
    args = cli.parse_args()
    url = args.base.rstrip("/")
    with urlopen(url + "/api/replay/" + args.run, timeout=30) as response:
        data = json.load(response)
    fps, rows = data["fps"], data["rows"]
    start = round(args.start * fps)
    end = len(rows) if args.end is None else round(args.end * fps)
    if not (0 <= start < end <= len(rows)) or args.scoreboard <= 0:
        cli.error("Select a positive clip inside the run and a positive scoreboard hold")
    if args.gif and (args.stills or args.aspect == "9x16"):
        cli.error("--gif requires a 16:9 recording")
    args.out.mkdir(parents=True, exist_ok=True)
    orientations = list(SIZES) if args.aspect == "both" else [args.aspect]
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=args.chrome, args=["--disable-gpu"])
        for aspect in orientations:
            w, h = SIZES[aspect]
            page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
            page.goto(url + "/?film=1&run=" + args.run, wait_until="networkidle")
            page.wait_for_function("window.captureReady && document.fonts.status === 'loaded'")
            if args.stills:
                if aspect == "16x9":
                    onset = next((i for i, row in enumerate(rows) if row[1]["tick"][1] == "reflex"), data["flip"] - 1)
                    crash = next((i for i, row in enumerate(rows) if row[0]["event"] in ("crash", "wipeout")), 0)
                    times = {"hero": min(onset + fps, len(rows) - 1), "crash": crash,
                             "flip": min(data["flip"] + 6, len(rows) - 1), "scoreboard": "results"}
                    for name, frame in times.items():
                        image = Image.open(io.BytesIO(capture(page, frame))).convert("L")
                        image.thumbnail((1440, 1440))
                        image.save(args.out / f"{name}.png", optimize=True)
            else:
                out = args.out / f"{args.run}-{aspect}.mp4"
                movie(page, out, fps, start, end, args.scoreboard)
                print(f"{out.name}: ticks {start} to {end - 1} at {fps} fps, plus {args.scoreboard:g} s scoreboard")
                if args.gif and aspect == "16x9":
                    gif(out, args.out / "demo.gif")
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
