# Dashboard footage

Both files show the same seed-17 CPU smoke run. The capture plays simulation ticks 465–1319 (00:15.50–00:44.00 of the source) at **1×**, followed by four seconds on the full two-minute scoreboard. The browser renders every captured game tick at 30 fps. The files have sound-off captions and no audio.

| File | Frame size | Duration |
|---|---:|---:|
| [smoke-17-16x9.mp4](screenshots/smoke-17-16x9.mp4) | 1920×1080 | 32.50 s |
| [smoke-17-9x16.mp4](screenshots/smoke-17-9x16.mp4) | 1080×1920 | 32.50 s |
| [demo.gif](screenshots/demo.gif) | 688×388 | 32.50 s at 8 fps, 1× simulation time |

## Timeline in both MP4s

| Clip time | What happens |
|---|---|
| 00:00–00:03 | Both ships are already playing. LLM-only loses its last heart at 00:01.37. Twitch produces its first graphite tick at 00:01.60. |
| 00:03–00:09.50 | The comb fills. The three-second reflex share first reaches 80% at 00:03.97. |
| 00:09.50–00:15.63 | The rule changes to catch squares. The fresh comb stops, the meter falls, and the teacher pulses continue. |
| 00:15.63–00:28.50 | The reflex takes control again. The held-out agreement window reaches the recovery criterion at 00:26.63, 28 answers into that epoch. This is recovery of teacher agreement. |
| 00:28.50–00:32.50 | The full-run scoreboard. LLM-only scored 42.5 points/minute and twitch scored 23.0. The reflex drove 65.8% of all frames. The first flip took 17.2 seconds to reach the agreement criterion. |

Start `make demo`, then run `python -m scripts.record_demo --run smoke-17 --start 15.5 --end 44 --scoreboard 4 --gif`. Use `--chrome /path/to/chrome` for a system Chrome, or install Playwright Chromium. The source logs are [LLM-only](../results/smoke-17-A.json) and [twitch](../results/smoke-17-E.json).
