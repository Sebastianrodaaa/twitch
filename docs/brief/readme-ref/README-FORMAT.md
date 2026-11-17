# README format

README (athanor format, black and white ONLY; also modelled on https://github.com/matiassingers/awesome-readme). Follow this format verbatim, and copy it into docs/brief/readme-ref/README-FORMAT.md. Model: https://github.com/ghgoodreau/athanor/blob/main/README.md

PALETTE: black and white only. Pure black background, white/grey type and art. No sepia, no color accents anywhere (banner, screenshots, diagrams, badges). The live dashboard may use the ice-lab palette; any README screenshot or GIF of it must be converted to black and white.

Banner (similar to athanor's):

- 1280x400 PNG at docs/screenshots/banner.png, first line of README: `![name](docs/screenshots/banner.png)`
- Centered project name in a large pixel blackletter/gothic bitmap font, lowercase: twitch
- Under it, a 1-2 line lowercase monospace tagline: "too slow to play. it grew a reflex."
- Left and right edges: Eadweard Muybridge, Animal Locomotion (1887, public domain) motion-sequence plates, dithered to two tones (1-bit/ordered or Floyd-Steinberg), fading into the black. Credit the source in NOTICE.md.

Structure:

1. Banner, then 2-3 plain sentences saying what it is.
2. One large hero screenshot of the real product/UI right away (B&W).
3. Short themed sections: ## The Lag, ## The Teacher, ## The Handoff, ## The Flip, ## The Results (negatives included), each a few short plain paragraphs followed by one real screenshot or plot. Inline `code` for commands.
4. A 2x2 markdown image grid where it helps.
5. ## Install: one sh block (clone, dry run, run), then what it touches and how to undo. Document model download and checksums. No API key is required.
6. ## The Name: a twitch is the muscle acting before the thought arrives.
7. ## Inspirations: short, links INSPIRATIONS.md / related work (SwiftSage, DAgger, Beveridge, Decision 1.0, others above).
8. ## License: MIT, third-party terms in NOTICE.md.

Voice: calm, literal, short sentences, a little wit, no hype, no emoji.
Under the tagline paragraph: the GitHub topics as hashtags, and a single small row of black-and-white shields.io badges (style=flat-square, color black/white). Demo GIF/video from a real run is required. Embed or link the 16:9 footage (or GIF cut from it) in the README.

GitHub topics (already set on the repo; also show them as hashtags in the README): ai-frontier-research, decision-models, distillation, online-learning, game-ai, real-time-agents, llm-agents, continual-learning, cpu-inference, pytorch
