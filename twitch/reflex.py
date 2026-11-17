"""A CPU reflex with calibrated selective handoff and recency-weighted replay."""
from collections import deque

import numpy as np
import torch
from torch import nn

from twitch.game import LANES


class Reflex:
    def __init__(self, seed):
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(13 * LANES + 2, 512), nn.ReLU(),
            nn.Linear(512, 960), nn.ReLU(),
            nn.Linear(960, 512), nn.ReLU(), nn.Linear(512, 1),
        )
        self.optim = torch.optim.SGD(self.net.parameters(), lr=0.04, momentum=0.65)
        self.rng = np.random.default_rng(seed)
        self.replay = deque(maxlen=512)
        self.reset()

    def reset(self):
        self.audit = deque(maxlen=32)
        self.temperature = 1.0
        self.threshold = None

    @staticmethod
    def inputs(frame):
        sense = frame.sense()
        examples = np.zeros((3, LANES * 13 + 2), dtype=np.float32)
        for action in range(3):
            lane = frame.target(action)
            # Lane-centered encoding shares the same scorer across possible destinations.
            for dst in range(LANES):
                src = lane + dst - LANES // 2
                if 0 <= src < LANES:
                    examples[action, dst * 13:(dst + 1) * 13] = sense[src]
            examples[action, -2:] = (float(action == 1), frame.cooldown / 3)
        return torch.from_numpy(examples)

    def logits(self, inputs):
        return self.net(inputs).squeeze(-1)

    @torch.no_grad()
    def decide(self, inputs, frame):
        raw = self.logits(inputs)
        for action in (0, 2):
            if frame.target(action) == frame.lane:
                raw[action] = -1e6
        prob = torch.softmax(raw / self.temperature, dim=-1)
        return int(prob.argmax()), prob, raw

    def observe(self, inputs, teacher, raw, frame, epoch, audit):
        p = np.asarray(teacher, dtype=np.float32).copy()
        for action in (0, 2):
            if frame.target(action) == frame.lane:
                p[1] += p[action]
                p[action] = 0
        p /= p.sum()
        if audit and frame.epoch == epoch:
            # Prequential holdout: these answers never enter SGD.
            self.audit.append((raw, int(p.argmax())))
            self.calibrate()
        if audit:
            return
        self.replay.append((inputs, torch.from_numpy(p), frame.epoch))
        epochs = np.array([row[2] for row in self.replay])
        weights = 0.04 ** np.minimum(3, epoch - epochs)
        for _ in range(4):
            indices = self.rng.choice(len(self.replay), min(24, len(self.replay)), p=weights / weights.sum())
            batch = [self.replay[i] for i in indices]
            x = torch.stack([row[0] for row in batch])
            y = torch.stack([row[1] for row in batch])
            loss = -(y * self.logits(x).log_softmax(dim=-1)).sum(dim=-1).mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def calibrate(self):
        if len(self.audit) < 5:
            return
        logits = torch.stack([row[0] for row in self.audit])
        labels = torch.tensor([row[1] for row in self.audit])
        temperatures = torch.tensor([0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0])
        logs = (logits[None] / temperatures[:, None, None]).log_softmax(dim=-1)
        index = int((-logs[:, torch.arange(len(labels)), labels].mean(dim=1)).argmin())
        self.temperature = float(temperatures[index])
        confidence, predicted = logs[index].exp().max(dim=1)
        agree = labels == predicted
        self.threshold = None
        for low in (0.34, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
            chosen = confidence >= low
            if int(chosen.sum()) >= 4 and float(agree[chosen].float().mean()) >= 0.80:
                self.threshold = low
                break
