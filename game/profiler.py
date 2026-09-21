# Timing. This is where the numbers in my report section come from.
#
# I measure two different things and they are not the same:
#   - how long each module takes, which tells us who the bottleneck is
#   - end-to-end latency, from grabbing a frame to putting it on screen,
#     which is what the player actually feels
from __future__ import annotations

import time
from collections import deque


class Profiler:
    def __init__(self, window: int = 60):
        # A rolling window of the last N frames, so the numbers react to
        # what is happening now instead of averaging over the whole session.
        self.window = window
        self.times: dict[str, deque] = {}
        self.frame_times: deque = deque(maxlen=window)
        self.latency: deque = deque(maxlen=window)
        self._frame_start = None

    def start_frame(self) -> None:
        self._frame_start = time.perf_counter()

    def end_frame(self) -> None:
        if self._frame_start is not None:
            self.frame_times.append(time.perf_counter() - self._frame_start)

    def record(self, name: str, seconds: float) -> None:
        if name not in self.times:
            self.times[name] = deque(maxlen=self.window)
        self.times[name].append(seconds)

    def record_latency(self, seconds: float) -> None:
        self.latency.append(seconds)

    def ms(self, name: str) -> float:
        d = self.times.get(name)
        return 1000.0 * sum(d) / len(d) if d else 0.0

    @property
    def fps(self) -> float:
        if not self.frame_times:
            return 0.0
        mean = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / mean if mean > 0 else 0.0

    @property
    def latency_ms(self) -> float:
        return 1000.0 * sum(self.latency) / len(self.latency) if self.latency else 0.0

    def lines(self) -> list[str]:
        """Short strings for the on-screen stats panel."""
        out = [f"{self.fps:4.1f} fps   {self.latency_ms:5.1f} ms latency"]
        for name in sorted(self.times, key=lambda n: -self.ms(n)):
            out.append(f"{name:<9}{self.ms(name):5.1f} ms")
        return out

    def report(self) -> str:
        """The summary printed when the game exits. Paste into the report."""
        rows = [f"frames measured : {len(self.frame_times)}",
                f"average fps     : {self.fps:.1f}",
                f"average latency : {self.latency_ms:.1f} ms", "",
                "per module, slowest first:"]
        for name in sorted(self.times, key=lambda n: -self.ms(n)):
            rows.append(f"   {name:<10} {self.ms(name):6.2f} ms")
        return "\n".join(rows)
