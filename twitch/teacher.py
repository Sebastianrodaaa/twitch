"""Frozen-frame scoring and a bounded uncertainty queue."""
import heapq
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter, sleep
from urllib.request import Request, urlopen

from twitch.game import ACTIONS, COLORS, RULES, SHAPES

LETTERS = ("A", "B", "C")
BEVERIDGE = "sathiskumarjothi0/Beveridge-Large-C"


def frame_text(frame):
    groups = []
    hits = sorted({item.hit for item in frame.items})[:2]
    for label, hit in zip(("First drop", "Second drop"), hits):
        items = ", ".join(
            f"lane {item.lane + 1} {COLORS[item.color]} {SHAPES[item.shape]}"
            for item in sorted(frame.items, key=lambda item: item.lane)
            if item.hit == hit
        )
        groups.append(f"{label}: {items}.")
    # Vary the countdown at the end so neighboring frames can share the CPU prompt cache.
    eta = ", ".join(str(hit - frame.number) for hit in hits)
    return (f"Rule: {RULES[frame.rule]}. Ship: lane {frame.lane + 1} of 5. "
            + " ".join(groups) + f" Drop times: {eta} frames.")


def qwen(url):
    def score(document):
        prompt = (
            "<|im_start|>system\nYou control a ship in five lanes. Move toward the closest good "
            "item at the first drop, or toward an empty lane if there is no good item. "
            "A bad item crashes the ship. Output one letter. A = move one lane left "
            "(smaller number), B = stay, C = move one lane right (larger number).<|im_end|>\n"
            f"<|im_start|>user\n{document}<|im_end|>\n<|im_start|>assistant\n"
        )
        data = {
            "prompt": prompt, "n_predict": 1, "temperature": 1, "top_k": 0,
            "top_p": 1, "min_p": 0, "seed": 0, "n_probs": 3, "cache_prompt": True,
            # An equal shift keeps the relative option logits and excludes the rest
            # of the vocabulary from the three returned probabilities.
            "logit_bias": [[letter, 100] for letter in LETTERS], "post_sampling_probs": True,
        }
        request = Request(url.rstrip("/") + "/completion", json.dumps(data).encode(),
                          {"Content-Type": "application/json"})
        with urlopen(request, timeout=90) as response:
            reply = json.load(response)
        row, = reply["completion_probabilities"]
        scores = {item["token"]: item["prob"] for item in row["top_probs"]}
        probs = [scores.get(letter, 0.0) for letter in LETTERS]
        if not all(math.isfinite(p) and p > 0 for p in probs):
            raise ValueError(f"Expected three option probabilities, got {scores}")
        return [p / sum(probs) for p in probs]
    return score


def beveridge():
    import bev
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(BEVERIDGE, revision="7cb0d24cce8572b8803a738c083b2419cdc40a8a",
                                 allow_patterns=["*.json", "*.safetensors"])
    model = bev.load(snapshot, device="cpu")
    questions = {"move": {
        "type": "choice",
        "instructions": "Which one-lane move avoids a bad item and heads toward a good item at the first drop?",
        "criteria": dict(zip(ACTIONS, ("smaller lane number", "hold current lane", "larger lane number"))),
    }}

    def score(document):
        probabilities = model.decide(document, questions)["move"]["probabilities"]
        return [probabilities[name] for name in ACTIONS]
    return score


def timed_score(score, text, minimum=0.0):
    started = perf_counter()
    probs = score(text)
    elapsed = perf_counter() - started
    if minimum > elapsed:
        sleep(minimum - elapsed)
    return probs, round(elapsed * 1000, 3)


@dataclass(frozen=True)
class Query:
    frame: object
    text: str
    audit: bool


class Queue:
    def __init__(self, score, minimum=0.25, budget_frames=None):
        self.score = score
        self.minimum = minimum
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = []
        self.job = None
        self.issued = 0
        self.budget_frames = budget_frames
        self.epoch = -1
        self.issued_in_epoch = 0

    def offer(self, frame, confidence, failed):
        heapq.heappush(self.pending, (confidence - 2 * failed, -frame.number, frame))
        # At 30 Hz, filtering this small window costs less than a second heap.
        self.pending = [row for row in self.pending if -row[1] >= frame.number - 9
                        and row[2].epoch == frame.epoch]
        heapq.heapify(self.pending)

    def submit(self, current, targeted=False):
        if self.job:
            return
        if self.budget_frames is not None:
            if current.epoch != self.epoch:
                self.epoch, self.issued_in_epoch = current.epoch, 0
            limit = max(1, int(0.048 * self.budget_frames))
            burst = (limit + 2) // 3
            elapsed = current.number % self.budget_frames
            allowed = burst + (limit - burst) * (elapsed + 1) // self.budget_frames
            if self.issued_in_epoch >= allowed:
                return
            self.issued_in_epoch += 1
        audit = not targeted or self.issued % 3 == 0
        frame = current if audit else self.pending[0][2]
        # Older candidates cannot be reused after either path dispatches.
        self.pending.clear()
        query = Query(frame, frame_text(frame), audit)
        self.job = query, self.executor.submit(timed_score, self.score, query.text, self.minimum)
        self.issued += 1

    def poll(self):
        if self.job and self.job[1].done():
            query, future = self.job
            self.job = None
            return query, *future.result()
        return None

    def close(self):
        return self.executor.shutdown(cancel_futures=True)
