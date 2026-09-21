# FAKE MODULES so the game runs before anybody has written anything real.
#
# Every one of these is a placeholder for a teammate's task. They produce
# believable rubbish: faces that drift, keypoints that wobble and sometimes
# drop out, identities that could get swapped. That is on purpose. If the
# game only survives perfect input, it will fall apart on the demo day.
#
# To test YOUR module, pass its name on the command line and the stub for it
# is replaced:  python main.py --real face
from __future__ import annotations

import math
import random

import numpy as np

from game_types import (BUDGET_MS, DrawCommand, Face, GameEvent, GameState,
                       Keypoint, Module, Player)


# ---------------------------------------------------------------- TASK 1
class FaceStub(Module):
    """Stand-in for face tracking and augmented effects."""
    name = "face"

    def __init__(self, players: int = 2):
        self.players = players
        self.t = 0.0

    def process(self, frame, state: GameState) -> None:
        h, w = frame.shape[:2]
        self.t += 0.05
        for i in range(self.players):
            # Two faces, left and right, drifting so nothing is ever static.
            cx = w * (0.28 + 0.44 * i) + 18 * math.sin(self.t + i)
            cy = h * 0.32 + 12 * math.cos(self.t * 0.8 + i)
            size = 110 + 12 * math.sin(self.t * 0.5 + i)
            player = state.player(i)
            if player is None:
                player = Player(player_id=i,
                                colour=(80, 80, 240) if i == 0 else (240, 170, 60))
                state.players.append(player)
            player.face = Face(cx, cy, size, size * 1.25,
                               angle=6 * math.sin(self.t * 0.6 + i))
            player.frames_missing = 0

        # The augmented effect that has to stay glued to the face, plus the
        # health bar the brief asks for as a second, different effect.
        for player in state.players:
            if player.face is None:
                continue
            f = player.face
            state.draw.append(DrawCommand("circle", f.cx, f.cy, radius=f.w * 0.5,
                                          colour=player.colour, thickness=3, layer=2))
            # The bar goes UNDER the chin, not above the head. Above the head
            # it falls off the top of the frame as soon as a player stands up,
            # and HUD that leaves the screen is a bug, not a style choice.
            bar_w = f.w
            x0, y0 = f.cx - bar_w / 2, f.cy + f.h * 0.62
            state.draw.append(DrawCommand("rect", x0, y0, x0 + bar_w, y0 + 11,
                                          colour=(60, 60, 60), thickness=-1, layer=2))
            state.draw.append(DrawCommand("rect", x0, y0,
                                          x0 + bar_w * player.health / 100, y0 + 11,
                                          colour=player.colour, thickness=-1, layer=2))
            state.draw.append(DrawCommand("text", x0, y0 + 28,
                                          text=f"P{player.player_id + 1}",
                                          colour=player.colour, scale=0.6, layer=2))


# ---------------------------------------------------------------- TASK 2
class PoseStub(Module):
    """Stand-in for body pose estimation and motion tracking.

    Deliberately imperfect: keypoints jitter, and now and then one drops to
    low confidence so the rest of the system has to cope with it.
    """
    name = "pose"

    def __init__(self, dropout: float = 0.06, jitter: float = 3.0):
        self.dropout = dropout
        self.jitter = jitter
        self.t = 0.0

    def process(self, frame, state: GameState) -> None:
        h, w = frame.shape[:2]
        self.t += 0.05
        for player in state.players:
            if player.face is None:
                continue
            f = player.face
            swing = math.sin(self.t * 1.6 + player.player_id * 2.0)
            points = {
                "nose": (f.cx, f.cy),
                "left_shoulder": (f.cx - f.w * 0.55, f.cy + f.h * 0.8),
                "right_shoulder": (f.cx + f.w * 0.55, f.cy + f.h * 0.8),
                "left_wrist": (f.cx - f.w * (0.9 + 0.5 * swing), f.cy + f.h * (1.4 - 0.5 * swing)),
                "right_wrist": (f.cx + f.w * (0.9 - 0.5 * swing), f.cy + f.h * (1.4 + 0.5 * swing)),
                "left_hip": (f.cx - f.w * 0.42, f.cy + f.h * 1.9),
                "right_hip": (f.cx + f.w * 0.42, f.cy + f.h * 1.9),
            }
            for name, (x, y) in points.items():
                confidence = 0.2 if random.random() < self.dropout else 0.95
                player.keypoints[name] = Keypoint(
                    x + random.gauss(0, self.jitter),
                    y + random.gauss(0, self.jitter),
                    confidence)

            # Skeleton, drawn only from points we actually trust.
            for a, b in (("left_shoulder", "right_shoulder"),
                         ("left_shoulder", "left_wrist"),
                         ("right_shoulder", "right_wrist"),
                         ("left_shoulder", "left_hip"),
                         ("right_shoulder", "right_hip")):
                ka, kb = player.point(a), player.point(b)
                if ka and kb:
                    state.draw.append(DrawCommand("line", ka.x, ka.y, kb.x, kb.y,
                                                  colour=player.colour, thickness=2, layer=1))
            for hand in ("left_wrist", "right_wrist"):
                k = player.point(hand)
                if k:
                    state.draw.append(DrawCommand("circle", k.x, k.y, radius=11,
                                                  colour=player.colour, thickness=-1, layer=1))


# ---------------------------------------------------------------- TASK 3
class IdentityStub(Module):
    """Stand-in for multiplayer detection and identity tracking.

    The real version has to survive crossing, occlusion and re-entry. This one
    only keeps ids stable and counts how long a player has been missing, so
    the rest of the pipeline can be written against that behaviour.
    """
    name = "identity"

    def __init__(self, patience: int = 15):
        self.patience = patience

    def process(self, frame, state: GameState) -> None:
        for player in state.players:
            if player.face is None:
                player.frames_missing += 1
                if player.frames_missing == self.patience:
                    state.events.append(GameEvent("player_lost", player.player_id))
            elif player.frames_missing:
                state.events.append(GameEvent("player_returned", player.player_id))
                player.frames_missing = 0


# ---------------------------------------------------------------- TASK 4
class ControlStub(Module):
    """Stand-in for player interaction and game control.

    A placeholder game: a ball bounces around and players bat it with their
    hands. Enough to prove the loop works end to end and to give the renderer
    real events to react to. Task 4's owner replaces this with the real game.
    """
    name = "control"

    def __init__(self):
        self.pos = np.array([320.0, 120.0])
        self.vel = np.array([5.5, 4.0])
        self.cooldown = 0

    def reset(self) -> None:
        self.__init__()

    def process(self, frame, state: GameState) -> None:
        h, w = frame.shape[:2]
        self.pos += self.vel
        # Walls.
        if self.pos[0] < 15 or self.pos[0] > w - 15:
            self.vel[0] *= -1
            self.pos[0] = float(np.clip(self.pos[0], 15, w - 15))
        if self.pos[1] < 15 or self.pos[1] > h - 15:
            self.vel[1] *= -1
            self.pos[1] = float(np.clip(self.pos[1], 15, h - 15))

        # A cooldown stops one swing registering as twenty hits, which is the
        # "prevent accidental repeated actions" requirement in the brief.
        if self.cooldown:
            self.cooldown -= 1
        else:
            for player in state.players:
                for hand in ("left_wrist", "right_wrist"):
                    k = player.point(hand)
                    if k is None:
                        continue
                    if math.hypot(k.x - self.pos[0], k.y - self.pos[1]) < 45:
                        direction = self.pos - np.array([k.x, k.y])
                        norm = np.linalg.norm(direction) or 1.0
                        self.vel = direction / norm * 7.5
                        player.score += 1
                        self.cooldown = 8
                        state.events.append(GameEvent("hit", player.player_id, 1.0))
                        break

        state.draw.append(DrawCommand("circle", self.pos[0], self.pos[1], radius=15,
                                      colour=(255, 255, 255), thickness=-1, layer=1))
        scores = "   ".join(f"P{p.player_id + 1} {p.score}" for p in state.players)
        state.draw.append(DrawCommand("text", 12, frame.shape[0] - 16, text=scores,
                                      colour=(255, 255, 255), scale=0.8, layer=2))
