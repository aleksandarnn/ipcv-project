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
    # The first frames are not representative: the models are still warming
    # up, the camera is still settling its exposure, and the first render
    # builds the arena cache. The report wants the rate during normal
    # gameplay, so these are counted but kept out of the statistics. The
    # number of skipped frames is stated in the summary, not hidden.
    WARMUP = 30

    def __init__(self, window: int = 60):
        # A rolling window of the last N frames, so the numbers react to
        # what is happening now instead of averaging over the whole session.
        self.window = window
        self.times: dict[str, deque] = {}
        self.frame_times: deque = deque(maxlen=window)
        self.latency: deque = deque(maxlen=window)
        self._frame_start = None

        # The rolling window above is for the live panel. The report needs the
        # whole run instead: an average hides the stalls, and the report asks
        # for a low percentile, which is exactly where the stalls show up.
        self.all_frames: list[float] = []
        self.all_latency: list[float] = []
        self.skipped = 0
        self.started_at = time.perf_counter()

    def start_frame(self) -> None:
        self._frame_start = time.perf_counter()

    def end_frame(self) -> None:
        if self._frame_start is not None:
            spent = time.perf_counter() - self._frame_start
            self.frame_times.append(spent)
            if self.skipped < self.WARMUP:
                self.skipped += 1
            else:
                self.all_frames.append(spent)

    def record(self, name: str, seconds: float) -> None:
        if name not in self.times:
            self.times[name] = deque(maxlen=self.window)
        self.times[name].append(seconds)

    def record_latency(self, seconds: float) -> None:
        self.latency.append(seconds)
        if self.skipped >= self.WARMUP:
            self.all_latency.append(seconds)

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

    # The report asks for "mean and low-percentile FPS". Low-percentile means
    # the slow end of the run: sort the frame durations, take the one 95 % of
    # the way up, and turn it into a frame rate. So "5 % low fps" is the rate
    # during the worst one frame in twenty, which is what a stutter feels like.
    def low_fps(self, percentile: float = 5.0) -> float:
        if not self.all_frames:
            return 0.0
        ordered = sorted(self.all_frames)
        index = min(len(ordered) - 1,
                    int(round((1.0 - percentile / 100.0) * (len(ordered) - 1))))
        slowest = ordered[index]
        return 1.0 / slowest if slowest > 0 else 0.0

    def mean_fps(self) -> float:
        """Over the whole run, not the rolling window."""
        if not self.all_frames:
            return 0.0
        mean = sum(self.all_frames) / len(self.all_frames)
        return 1.0 / mean if mean > 0 else 0.0

    def freezes(self, threshold: float = 0.25) -> int:
        """Frames that took longer than a quarter second, i.e. visible stalls."""
        return sum(1 for t in self.all_frames if t > threshold)

    def lines(self) -> list[str]:
        """Short strings for the on-screen stats panel."""
        out = [f"{self.fps:4.1f} fps   {self.latency_ms:5.1f} ms latency"]
        for name in sorted(self.times, key=lambda n: -self.ms(n)):
            out.append(f"{name:<9}{self.ms(name):5.1f} ms")
        return out

    def report(self, bootstrap: float = 0.0) -> str:
        """The summary printed when the game exits. Paste into the report."""
        run = time.perf_counter() - self.started_at
        mean_latency = (1000.0 * sum(self.all_latency) / len(self.all_latency)
                        if self.all_latency else 0.0)
        worst_latency = 1000.0 * max(self.all_latency) if self.all_latency else 0.0

        rows = [f"frames measured : {len(self.all_frames)} "
                f"(first {self.skipped} skipped as warm-up)",
                f"run length      : {run:.1f} s",
                f"mean fps        : {self.mean_fps():.1f}",
                f"5 % low fps     : {self.low_fps(5):.1f}",
                f"1 % low fps     : {self.low_fps(1):.1f}",
                f"mean latency    : {mean_latency:.1f} ms",
                f"worst latency   : {worst_latency:.1f} ms",
                f"stalls > 250 ms : {self.freezes()}"]
        if bootstrap:
            rows.append(f"bootstrap       : {bootstrap:.1f} s "
                        f"(launch to first frame on screen)")
        rows += ["", "per module, slowest first:"]
        for name in sorted(self.times, key=lambda n: -self.ms(n)):
            rows.append(f"   {name:<10} {self.ms(name):6.2f} ms")
        return "\n".join(rows)
