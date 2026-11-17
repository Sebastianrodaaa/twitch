"""Deterministic 30 Hz lane game. Controllers never change the spawn stream."""
from dataclasses import dataclass

import numpy as np

FPS = 30
LANES = 5
FLIP = 25 * FPS
LOOKAHEAD = 45
ACTIONS = ("left", "stay", "right")
SHAPES = ("circle", "square")
COLORS = ("blue", "mint")
RULES = (
    "Catch circles, dodge squares",
    "Catch squares, dodge circles",
    "Catch blue, dodge mint",
    "Catch mint, dodge blue",
)


@dataclass(frozen=True)
class Item:
    lane: int
    shape: int
    color: int
    hit: int
    travel: int

    def good(self, rule):
        return (self.shape, self.shape, self.color, self.color)[rule] == rule % 2


@dataclass(frozen=True)
class Frame:
    number: int
    lane: int
    cooldown: int
    items: tuple[Item, ...]
    epoch: int
    rule: int

    def target(self, action):
        return min(LANES - 1, max(0, self.lane + action - 1))

    def sense(self):
        # One row per lane. Three upcoming arrivals retain independent shape and color.
        state = np.zeros((LANES, 13), dtype=np.float32)
        state[self.lane, -1] = 1
        counts = [0] * LANES
        for item in self.items:
            j = counts[item.lane]
            if j < 3:
                start = 4 * j
                state[item.lane, start:start + 4] = (
                    1, 1 - (item.hit - self.number) / LOOKAHEAD,
                    1 - 2 * item.shape, 1 - 2 * item.color,
                )
                counts[item.lane] += 1
        return state


class Game:
    def __init__(self, seed, flip=FLIP):
        self.rng = np.random.default_rng(seed)
        self.flip = flip
        self.number = 0
        self.lane = 2
        self.cooldown = 0
        self.items = []
        self.score = self.catches = self.crashes = 0
        self.hearts = 3
        self.wipeouts = 0
        self.respawn = 0
        self.next_hit = 42
        self._spawn()

    def _spawn(self):
        while self.next_hit <= self.number + LOOKAHEAD:
            lanes = self.rng.choice(LANES, 3, replace=False)
            for lane in lanes:
                shape, color = map(int, self.rng.integers(0, 2, size=2))
                travel = int(self.rng.integers(28, LOOKAHEAD + 1))
                self.items.append(Item(int(lane), shape, color, self.next_hit, travel))
            self.next_hit += int(self.rng.integers(13, 23))

    def frame(self):
        epoch = self.number // self.flip
        return Frame(self.number, self.lane, self.cooldown, tuple(self.items), epoch, epoch % len(RULES))

    def step(self, target):
        frame = self.frame()
        event = ""
        if self.respawn:
            self.respawn -= 1
            if not self.respawn:
                self.hearts = 3
        else:
            if self.cooldown:
                self.cooldown -= 1
            elif target != self.lane:
                self.lane += 1 if target > self.lane else -1
                self.cooldown = 3
            for item in self.items:
                if item.hit != self.number or item.lane != self.lane:
                    continue
                if item.good(frame.rule):
                    self.score += 3
                    self.catches += 1
                    event = "catch"
                else:
                    self.score -= 2
                    self.crashes += 1
                    self.hearts -= 1
                    event = "crash"
                    if not self.hearts:
                        self.wipeouts += 1
                        self.respawn = FPS
                        event = "wipeout"
        self.items = [item for item in self.items if item.hit > self.number]
        self.number += 1
        self._spawn()
        return event


def heuristic(frame):
    """Prefer a reachable lane at the next drop, with deterministic ties."""
    hit = frame.items[0].hit
    turns = max(0, (hit - frame.number - frame.cooldown + 4) // 4)
    values = {item.lane: 3 if item.good(frame.rule) else -2 for item in frame.items if item.hit == hit}
    lanes = [lane for lane in range(LANES) if abs(lane - frame.lane) <= turns]
    target = max(lanes, key=lambda lane: (values.get(lane, 0), -abs(lane - frame.lane)))
    return 1 + int(target > frame.lane) - int(target < frame.lane)
