# TASK 5 - scene rendering. This part is mine.
#
# Two jobs here:
#   1. Draw the scene, and make it REACT to what happens in the game. The
#      brief is explicit that a background which just sits there is not
#      enough, so hits shake the frame and flash the edges.
#   2. Draw everything the other modules asked for, in the right order, so
#      nobody fights over the same pixels.
from __future__ import annotations

import cv2
import numpy as np

from game_types import DrawCommand, GameState


class Renderer:
    name = "render"

    def __init__(self):
        self.shake = 0.0        # decays every frame
        self.flash = 0.0        # decays every frame
        self.flash_colour = (255, 255, 255)

    def reset(self) -> None:
        self.shake = 0.0
        self.flash = 0.0

    # ---------------------------------------------------------------- scene
    def react(self, state: GameState) -> None:
        """Turn game events into scene changes. This is the task 5 deliverable."""
        for event in state.events:
            if event.kind == "hit":
                self.shake = min(1.0, self.shake + 0.6 * event.strength)
                self.flash = 0.8
                player = state.player(event.player_id)
                self.flash_colour = player.colour if player else (255, 255, 255)
            elif event.kind == "score":
                self.flash = 0.5
                self.flash_colour = (80, 255, 80)
            elif event.kind == "player_lost":
                self.flash = 0.4
                self.flash_colour = (60, 60, 255)

    def draw(self, frame: np.ndarray, state: GameState) -> np.ndarray:
        out = frame

        # Camera shake, done by rolling the image a few pixels. Cheap, and it
        # sells the impact far better than any overlay.
        if self.shake > 0.01:
            amount = int(round(10 * self.shake))
            dx = np.random.randint(-amount, amount + 1)
            dy = np.random.randint(-amount, amount + 1)
            out = np.roll(out, (dy, dx), axis=(0, 1))
        self.shake *= 0.82

        # Screen flash, blended rather than pasted so it reads as light.
        if self.flash > 0.01:
            layer = np.full_like(out, self.flash_colour, dtype=np.uint8)
            out = cv2.addWeighted(out, 1.0 - 0.35 * self.flash, layer, 0.35 * self.flash, 0)
        self.flash *= 0.85

        # Everything the modules asked for, background first, HUD last.
        for command in sorted(state.draw, key=lambda c: c.layer):
            self._apply(out, command)

        if state.message:
            self._banner(out, state.message)
        return out

    # ------------------------------------------------------------ primitives
    @staticmethod
    def _apply(img: np.ndarray, c: DrawCommand) -> None:
        colour = tuple(int(v) for v in c.colour)
        if c.kind == "circle":
            cv2.circle(img, (int(c.x), int(c.y)), int(c.radius), colour,
                       c.thickness, cv2.LINE_AA)
        elif c.kind == "rect":
            cv2.rectangle(img, (int(c.x), int(c.y)), (int(c.x2), int(c.y2)),
                          colour, c.thickness, cv2.LINE_AA)
        elif c.kind == "line":
            cv2.line(img, (int(c.x), int(c.y)), (int(c.x2), int(c.y2)),
                     colour, c.thickness, cv2.LINE_AA)
        elif c.kind == "text":
            Renderer._text(img, c.text, (int(c.x), int(c.y)), c.scale, colour, c.thickness)

    @staticmethod
    def _text(img, text, org, scale, colour, thickness=1) -> None:
        """White text with a dark outline so it survives any background.

        The outline is drawn at the SAME thickness, nudged around. A thicker
        pen widens every letter slightly and by the end of a long line the
        black copy has drifted enough to print a ghost character.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0),
                       (1, 0), (-1, 1), (0, 1), (1, 1)):
            cv2.putText(img, text, (org[0] + dx, org[1] + dy), font, scale,
                        (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.putText(img, text, org, font, scale, colour, thickness, cv2.LINE_AA)

    @staticmethod
    def _banner(img, text) -> None:
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 1.1, 2)
        x, y = (w - tw) // 2, (h + th) // 2
        band = img[max(0, y - th - 14):y + 14, :]
        band[:] = (0.4 * band).astype(band.dtype)
        Renderer._text(img, text, (x, y), 1.1, (255, 255, 255), 2)

    # ----------------------------------------------------------------- stats
    def stats_panel(self, img: np.ndarray, lines: list[str], failed: list[str]) -> None:
        """Performance numbers, top left. Press s to hide during the demo."""
        pad, line_h = 8, 18
        height = line_h * (len(lines) + len(failed)) + pad
        panel = img[0:height, 0:225]
        panel[:] = (0.35 * panel).astype(panel.dtype)
        for i, line in enumerate(lines):
            self._text(img, line, (pad, pad + line_h * (i + 1) - 5), 0.45, (230, 230, 230))
        for j, name in enumerate(failed):
            self._text(img, f"{name} OFFLINE", (pad, pad + line_h * (len(lines) + j + 1) - 5),
                       0.45, (80, 80, 255))
