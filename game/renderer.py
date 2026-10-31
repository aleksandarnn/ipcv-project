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

from arena import Arena
from game_types import DrawCommand, GameState


class Renderer:
    name = "render"

    def __init__(self):
        self.arena = Arena()
        self.shake = 0.0        # decays every frame
        self.flash = 0.0        # decays every frame
        self.flash_colour = (255, 255, 255)

    def reset(self) -> None:
        self.shake = 0.0
        self.flash = 0.0

    # ---------------------------------------------------------------- scene
    def react(self, state: GameState) -> None:
        """Turn game events into scene changes. This is the task 5 deliverable.

        A samurai duel needs the three outcomes to look completely different,
        so the player can tell at a glance whether a swing landed, was
        blocked, or missed. Colour and shake do that work.
        """
        for event in state.events:
            if event.kind == "damage":
                # A hit landed. Hard shake and a red wash, the strongest
                # reaction in the game.
                self.shake = min(1.0, self.shake + 0.8 * event.strength)
                self.flash = 0.85
                self.flash_colour = (40, 40, 235)
            elif event.kind == "blocked":
                # Steel on steel. A short white spark, almost no shake,
                # because nothing got through.
                self.shake = min(1.0, self.shake + 0.2)
                self.flash = 0.55
                self.flash_colour = (235, 235, 245)
            elif event.kind == "missed":
                # The swing hit air. Barely anything, so a miss never feels
                # as good as a hit.
                self.flash = 0.18
                self.flash_colour = (200, 200, 200)
            elif event.kind == "ko":
                self.shake = 1.0
                self.flash = 1.0
                self.flash_colour = (30, 30, 210)
            elif event.kind == "player_lost":
                self.flash = 0.4
                self.flash_colour = (60, 60, 255)

    def dual_view(self, raw: np.ndarray, game: np.ndarray,
                  fraction: float = 0.26) -> np.ndarray:
        """Put the untouched webcam picture inside the game output.

        The peer review video has one rule I cannot satisfy any other way:
        "The webcam input and corresponding game output must be visible."
        The game normally replaces the camera picture, so without this there
        is only one image on screen and the rule is not met.

        An inset rather than side by side, because side by side halves the
        game and the reviewers are judging the game.
        """
        out = game  # drawn into directly; draw() already returned a fresh array
        h, w = out.shape[:2]
        iw = max(80, int(w * fraction))
        ih = max(60, int(iw * raw.shape[0] / raw.shape[1]))
        inset = cv2.resize(raw, (iw, ih), interpolation=cv2.INTER_AREA)

        margin = 10
        x0, y0 = w - iw - margin, h - ih - margin
        out[y0:y0 + ih, x0:x0 + iw] = inset
        cv2.rectangle(out, (x0 - 1, y0 - 1), (x0 + iw, y0 + ih),
                      (235, 235, 240), 1, cv2.LINE_AA)
        self._text(out, "webcam input", (x0 + 4, y0 - 6), 0.42, (235, 235, 240))
        return out

    def draw(self, frame: np.ndarray, state: GameState) -> np.ndarray:
        # The arena goes down first: the webcam picture inside a dojo. The
        # players are passed in so the dojo can be cut away around them
        # instead of fading out the whole middle of the picture.
        out = self.arena.compose(frame, state.players)

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

        self.health_bars(out, state)
        if state.message:
            self._banner(out, state.message)
        return out

    # --------------------------------------------------------- the top bar
    def health_bars(self, img: np.ndarray, state: GameState) -> None:
        """Two health bars in the beam across the top, fighting-game style.

        Player 1 drains to the left, player 2 to the right, so the gap always
        opens towards the middle and you can read who is winning at a glance.
        """
        height, width = img.shape[:2]
        bar_h = max(12, int(height * 0.045))
        top = max(6, int(height * 0.025))
        margin = int(width * 0.075)
        bar_w = int(width * 0.34)

        for player in state.players[:2]:
            left_side = player.player_id == 0
            x0 = margin if left_side else width - margin - bar_w
            fraction = max(0.0, min(1.0, player.health / 100.0))
            filled = int(bar_w * fraction)

            cv2.rectangle(img, (x0, top), (x0 + bar_w, top + bar_h), (20, 18, 18), -1)
            if filled:
                # Drain outwards from the centre of the screen.
                fx0 = x0 + (bar_w - filled) if not left_side else x0
                cv2.rectangle(img, (fx0, top), (fx0 + filled, top + bar_h),
                              player.colour, -1)
            cv2.rectangle(img, (x0, top), (x0 + bar_w, top + bar_h),
                          (210, 210, 215), 2, cv2.LINE_AA)

            label = f"P{player.player_id + 1}"
            lx = x0 if left_side else x0 + bar_w - 34
            self._text(img, label, (lx, top + bar_h + 20), 0.62, player.colour, 2)

        if state.phase == "playing" and len(state.players) >= 2:
            self._text(img, "VS", (width // 2 - 20, top + bar_h - 4), 0.8,
                       (235, 235, 240), 2)

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
        """Performance numbers, bottom left. Press s to hide during the demo.

        Bottom rather than top: the health bars own the top of the screen, and
        up there the panel sat on player one's bar and covered its label.
        """
        pad, line_h = 8, 18
        rows = len(lines) + len(failed)
        height = line_h * rows + pad
        bottom = img.shape[0] - pad
        top = max(0, bottom - height)

        panel = img[top:bottom, 0:225]
        panel[:] = (0.35 * panel).astype(panel.dtype)
        for i, line in enumerate(lines):
            self._text(img, line, (pad, top + line_h * (i + 1)), 0.45, (230, 230, 230))
        for j, name in enumerate(failed):
            self._text(img, f"{name} OFFLINE",
                       (pad, top + line_h * (len(lines) + j + 1)), 0.45, (80, 80, 255))
