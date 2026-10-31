# TASK 5 - the arena. The samurai world the duel happens in.
#
# The webcam shows a student room, which does not look like a duel. This file
# turns it into a dojo WITHOUT hiding the players, because the players are the
# game.
#
# How it works
#   The dojo is drawn procedurally with OpenCV, so there are no image files to
#   ship in the ZIP and nothing to go missing on the graders' machine. If a
#   picture is dropped into game/assets/dojo.jpg it is used instead, which is
#   the only supported way to change the look without touching this file.
#
#   Either way the backdrop never changes, so it is built ONCE per frame size
#   and kept. Every frame after that is a single blend, which costs almost
#   nothing. Rebuilding it 25 times a second would be pointless work.
#
#   The players are not masked out - that would need segmentation on every
#   frame, which there is no budget for. Instead the artwork is strongest at
#   the edges and nearly transparent in the middle, where the players stand.
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Colours in blue, green, red order, which is what OpenCV uses.
INK = (34, 30, 28)           # near black, the posts and the frame
WOOD = (42, 68, 104)         # the beams and the lattice
WOOD_DARK = (28, 44, 68)     # shadowed wood
PAPER = (168, 196, 212)      # the shoji paper, warm off-white
TATAMI = (104, 142, 134)     # woven straw matting
TATAMI_EDGE = (46, 62, 60)   # the cloth border on each mat
BANNER = (26, 22, 22)        # the bar across the top

ASSET_DIR = Path(__file__).resolve().parent / "assets"


class Arena:
    def __init__(self):
        self._size = None       # the frame size the cached backdrop was built for
        self._overlay = None    # the arena artwork
        self._inv8 = None       # 255 - alpha, as 8-bit, for dimming the webcam
        self._pre8 = None       # the artwork already multiplied by alpha

    # ------------------------------------------------------------- building
    @staticmethod
    def _photo(width: int, height: int) -> np.ndarray | None:
        """A dojo photograph from game/assets, if one was provided."""
        for name in ("dojo.jpg", "dojo.jpeg", "dojo.png", "dojo.webp"):
            path = ASSET_DIR / name
            if not path.exists():
                continue
            image = cv2.imread(str(path))
            if image is None:
                print(f"[arena] {path.name} could not be read, drawing instead")
                continue
            # Fill the frame and crop, rather than squash: a stretched room
            # looks wrong immediately.
            scale = max(width / image.shape[1], height / image.shape[0])
            resized = cv2.resize(image, (int(image.shape[1] * scale + 0.5),
                                         int(image.shape[0] * scale + 0.5)),
                                 interpolation=cv2.INTER_AREA)
            y0 = (resized.shape[0] - height) // 2
            x0 = (resized.shape[1] - width) // 2
            print(f"[arena] backdrop from assets/{path.name}")
            return resized[y0:y0 + height, x0:x0 + width].copy()
        return None

    def _draw_dojo(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """The dojo interior: shoji screens, posts, beams and a tatami floor."""
        art = np.zeros((height, width, 3), np.uint8)
        weight = np.zeros((height, width), np.float32)

        beam_h = int(height * 0.12)
        horizon = int(height * 0.72)

        # ---- the back wall: panels of shoji paper in a wooden lattice
        art[beam_h:horizon, :] = PAPER
        weight[beam_h:horizon, :] = 0.93

        panels = 6
        panel_w = width / panels
        for i in range(panels + 1):
            x = int(i * panel_w)
            cv2.line(art, (x, beam_h), (x, horizon), WOOD, max(2, width // 150),
                     cv2.LINE_AA)
        # The lattice inside each panel. Thin bars, because thick ones read as
        # a prison window rather than a paper screen.
        rows = 4
        for r in range(1, rows):
            y = int(beam_h + (horizon - beam_h) * r / rows)
            cv2.line(art, (0, y), (width, y), WOOD, 1, cv2.LINE_AA)
        for i in range(panels):
            for c in range(1, 3):
                x = int((i + c / 3) * panel_w)
                cv2.line(art, (x, beam_h), (x, horizon), WOOD, 1, cv2.LINE_AA)

        # ---- the threshold where the wall meets the floor
        sill = max(3, int(height * 0.015))
        cv2.rectangle(art, (0, horizon - sill), (width, horizon + sill),
                      WOOD_DARK, -1)
        weight[horizon - sill:horizon + sill, :] = 0.95

        # ---- the tatami floor, in mats that get shorter towards the back
        art[horizon:, :] = TATAMI
        weight[horizon:, :] = 0.90
        y = horizon
        band = max(4, int((height - horizon) * 0.22))
        while y < height:
            cv2.line(art, (0, y), (width, y), TATAMI_EDGE, 2, cv2.LINE_AA)
            # The woven texture: short strokes, denser near the camera.
            for x in range(0, width, max(6, band // 2)):
                cv2.line(art, (x, y + 2), (x, min(height, y + band - 2)),
                         TATAMI_EDGE, 1, cv2.LINE_AA)
            y += band
            band = int(band * 1.45)
        # Mat seams running away from the camera, which give the floor depth.
        for i in (-2, -1, 1, 2):
            x_far = width // 2 + i * int(width * 0.11)
            x_near = width // 2 + i * int(width * 0.34)
            cv2.line(art, (x_far, horizon), (x_near, height), TATAMI_EDGE, 2,
                     cv2.LINE_AA)

        # ---- the posts down each side
        post_w = int(width * 0.06)
        for x0 in (0, width - post_w):
            cv2.rectangle(art, (x0, 0), (x0 + post_w, height), WOOD_DARK, -1)
            cv2.rectangle(weight, (x0, 0), (x0 + post_w, height), 0.97, -1)
            cv2.line(art, (x0 + post_w, 0), (x0 + post_w, height), INK, 2,
                     cv2.LINE_AA)

        # ---- the beam across the top, which also gives the health bars a home
        cv2.rectangle(art, (0, 0), (width, beam_h), BANNER, -1)
        cv2.rectangle(weight, (0, 0), (width, beam_h), 0.94, -1)
        cv2.line(art, (0, beam_h), (width, beam_h), WOOD, 2, cv2.LINE_AA)
        # Two rafters hanging below it, so the top is not a plain black bar.
        for x in (int(width * 0.22), int(width * 0.78)):
            cv2.rectangle(art, (x - post_w // 3, 0),
                          (x + post_w // 3, int(beam_h * 1.45)), WOOD_DARK, -1)
            cv2.rectangle(weight, (x - post_w // 3, 0),
                          (x + post_w // 3, int(beam_h * 1.45)), 0.88, -1)

        return art, weight

    def _build(self, width: int, height: int) -> None:
        """Draw the backdrop once. Called again only if the frame size changes."""
        photo = self._photo(width, height)
        art_weight = None
        if photo is not None:
            overlay = photo
            # A photograph has no idea where the players are, so it gets one
            # flat weight and relies entirely on the centre mask below.
            alpha = np.full((height, width), 0.93, np.float32)
        else:
            overlay, alpha = self._draw_dojo(width, height)
        art_weight = alpha.copy()

        # Keep the middle of the picture clear. Without segmentation the only
        # way to stop the dojo painting over the players is to fade it out
        # where they stand, so the artwork is a frame around them rather than
        # a wall in front of them.
        yy = np.linspace(-1, 1, height, dtype=np.float32).reshape(-1, 1)
        xx = np.linspace(-1, 1, width, dtype=np.float32).reshape(1, -1)
        radius = np.sqrt(xx * xx + yy * yy)
        centre = np.clip((radius - 0.42) * 1.45, 0.10, 1.0)
        alpha = alpha * centre

        # A thin warm wash over the whole artwork, which does more for the
        # mood than any big shape and costs nothing extra because it is baked
        # into the same cached overlay. Mixed in everywhere rather than
        # swapped in below a threshold: swapping drew a visible ring around
        # the clear middle, because the colour changed abruptly where the
        # weight crossed the threshold.
        warm = np.zeros_like(overlay)
        warm[:, :] = (38, 62, 112)
        overlay = cv2.addWeighted(overlay, 0.86, warm, 0.14, 0)
        alpha = np.maximum(alpha, 0.10)

        # Generously blurred, so the edge of the clear middle is a gradient
        # rather than a shape you can point at.
        alpha = cv2.GaussianBlur(alpha, (0, 0), max(6.0, width / 90.0))

        # Pre-chew the blend so each frame is cheap.
        #
        # The honest version of this blend is
        #     out = frame * (1 - alpha) + overlay * alpha
        # which in numpy builds two float64 copies of the whole frame every
        # time. Measured at 960x540 it cost 17.5 ms per frame, nearly three
        # times the entire render budget.
        #
        # Since the arena never changes, both weights can be worked out once,
        # stored as 8-bit, and applied with two OpenCV calls instead:
        #     out = frame * inv8 / 255 + pre8
        # Same picture to within 2 levels out of 255, invisible on screen,
        # and it measures 1.0 ms. That is a 17x saving for no visual cost.
        alpha8 = np.repeat((alpha * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        self._inv8 = 255 - alpha8
        self._pre8 = cv2.multiply(overlay, alpha8, scale=1 / 255.0)
        self._overlay = overlay
        self._size = (width, height)

        # For the version that cuts the players out instead of fading the
        # middle: the artwork at full strength, and its own weight map as a
        # single 8-bit channel to multiply the cut-out into.
        self._art = overlay
        self._full1 = (np.clip(art_weight, 0.0, 1.0) * 255).astype(np.uint8)
        self._small = (max(8, width // 4), max(8, height // 4))

    # -------------------------------------------------------------- drawing
    def compose(self, frame: np.ndarray, players: list | None = None) -> np.ndarray:
        """Put the webcam picture inside the arena.

        With no players known, the cached blend is used: the dojo is strong at
        the edges and fades out in the middle, because covering a player who
        has not been detected would erase them.

        With players known, the dojo is painted at nearly full strength over
        the whole frame and cut away only around each player's box, which is
        what task 3 is already computing. The room behind them disappears and
        they stay clear, without segmenting anything on this frame.
        """
        height, width = frame.shape[:2]
        if self._size != (width, height):
            self._build(width, height)

        boxes = [p.bbox for p in players or [] if getattr(p, "bbox", None)]
        if not boxes:
            # Two 8-bit operations: dim the webcam picture by (1 - alpha),
            # then add the arena already multiplied by alpha when it was built.
            return cv2.add(cv2.multiply(frame, self._inv8, scale=1 / 255.0),
                           self._pre8)

        # The cut-out is drawn at quarter size and blurred there, which is 16
        # times less work than blurring it at full size and is more than soft
        # enough once it is scaled back up.
        sw, sh = self._small
        keep = np.zeros((sh, sw), np.uint8)
        for box in boxes:
            cx = int((box.x0 + box.x1) * 0.5 * sw / width)
            cy = int((box.y0 + box.y1) * 0.5 * sh / height)
            ax = max(4, int(box.w * 0.62 * sw / width))
            ay = max(4, int(box.h * 0.62 * sh / height))
            cv2.ellipse(keep, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
        keep = cv2.GaussianBlur(keep, (0, 0), max(2.0, sw / 28.0))
        keep = cv2.resize(keep, (width, height), interpolation=cv2.INTER_LINEAR)

        # alpha = artwork weight, minus wherever a player is standing.
        a1 = cv2.multiply(self._full1, 255 - keep, scale=1 / 255.0)
        a3 = cv2.merge((a1, a1, a1))
        return cv2.add(cv2.multiply(frame, 255 - a3, scale=1 / 255.0),
                       cv2.multiply(self._art, a3, scale=1 / 255.0))

    def horizon(self, height: int) -> int:
        return int(height * 0.68)
