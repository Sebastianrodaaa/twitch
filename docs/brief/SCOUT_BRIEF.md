# twitch scout brief

Sebastian Roda gave the go at 18:46 Europe/Paris on 2026-11-17. This brief reconstructs the approved spec from the build request. The scout files under `/workspace/ai-frontier` were unavailable in the build environment.

A slow LLM can judge a frozen frame while play continues. A roughly one-million-parameter reflex evaluates typed game state every frame. A priority queue selects low-confidence or recent-failure snapshots for asynchronous option scoring. Teacher distributions update the reflex online with recent-epoch-weighted replay. Temperature scaling and a threshold from held-out agreement decide when the reflex controls the ship. Otherwise the last teacher target holds. The rule text goes to the teacher. The learner can observe epoch boundaries for replay and calibration.

The increment over SwiftSage and DAgger is the combination of real-time, time-dilated supervision, uncertainty-driven sampling, selective per-frame control, and measured recovery after rule changes. The artifact is a learner that visibly changes as it plays.

## Demo

Two ships share a seed in a deterministic five-lane game at 30 Hz. The LLM-only ship requests one action per second. The twitch ship combines stale teacher targets and fresh reflex decisions. The rule changes every 25 simulation seconds among four shape and color rules. A live WebSocket and the 1× replay both show the same game. Show real crashes, teacher pulses, graphite reflex ticks, changes in reflex share through a flip, and a scoreboard from the selected run. The approximate 30 s storyboard and example inference times were targets. Measured footage and logs decide what can be shown.

The dashboard uses the ice-lab palette, the split stage, two-lane trace, narrow counters rail, and results cards. See [the visual references](ui-refs/REFS.md). README media is monochrome. Supply 1920×1080 and 1080×1920 recordings of the dashboard, a grayscale README GIF, and a shot list. Content will cut the launch film separately. Publishing the launch film and social copy needs Sebastian’s approval.

## Protocol

Use only CPU PyTorch, CPU llama.cpp, public model weights and local resources. Use Qwen2.5-1.5B-Instruct GGUF option logprobs; cap the online teacher to at most four answers per second if necessary. Measure latency separately from any pacing. Verify upstream licenses. No paid services or API keys.

| Arm | Controller |
|---|---|
| A | LLM-only real-time |
| B | LLM with the game clock paused for each decision |
| C | Hand-coded, rule-aware heuristic |
| D | Offline-distilled frozen reflex |
| E | twitch online distillation |
| F | Beveridge-Large-C zero-shot typed questions |
| G | Random |

Time A/B/E for seed 17 at two minutes each before sizing the study. Target five fixed seeds at 10 minutes per arm, reducing to three seeds if the timed estimate exceeds 24 hours. Record score/min, frame driver share, median and p95 latency, teacher answers/1,000 frames, recovery answer count, and agreement. Report every run and 95% bootstrap intervals over flips.

The hypothesis calls for E to score ≥85% of B after the fifth flip with median latency <15 ms and teacher answers on <5% of frames, score ≥2× A, and recover to 90% teacher agreement within 40 answers (median over flips). D should drop below 60% after the first flip. Criteria 1 and 3 must hold in four of five seeds or the equivalent majority if reduced to three. Disclose negative or censored results and any limitations of the game and teacher.

## Ship

Use the [Ponytail implementation rules](PONYTAIL.md), committed real logs, model-free reproduction, CPU smoke CI, the [README format](readme-ref/README-FORMAT.md), MIT, and a third-party notice. Include methods, related work, all measured results and a short social summary in the report. Open a branch PR to main and leave it unmerged.
