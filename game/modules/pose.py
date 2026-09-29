from __future__ import annotations


class Module:
    name = "pose"

    def __init__(self) -> None:
        pass

    def process(self, frame, state) -> None:
        # main.py calls this once per frame.
        pass

    def reset(self) -> None:
        pass
