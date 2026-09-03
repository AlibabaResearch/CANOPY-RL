"""Minimal timing utility for the AppWorld rollout integration."""

from __future__ import annotations

import time


class Timer:
    """Monotonic wall-clock timer with compact human-readable formatting."""

    def __init__(self, can_print: bool = True) -> None:
        self.start = time.monotonic()
        self.last = self.start
        self.can_print = can_print
        self.total = 0.0

    def tik(self) -> None:
        self.start = time.monotonic()
        self.last = self.start

    @staticmethod
    def get_info(milliseconds: float) -> str:
        if milliseconds < 5_000:
            return f"{int(milliseconds)}ms"
        seconds = milliseconds / 1_000
        if seconds < 300:
            return f"{seconds:.1f}s"
        minutes = seconds / 60
        if minutes < 120:
            return f"{minutes:.1f}min"
        hours = minutes / 60
        if hours < 72:
            return f"{hours:.1f}h"
        return f"{hours / 24:.1f}day"

    def get_print_info_by_seconds(self, seconds: float) -> str:
        return self.get_info(seconds * 1_000)

    def get_total_used_seconds(self) -> float:
        return round(time.monotonic() - self.start, 2)

    def tok(self, info: str) -> tuple[str, str]:
        end = time.monotonic()
        used = end - self.last
        total = end - self.start
        self.total = total
        self.last = end
        used_info = self.get_info(used * 1_000)
        total_info = self.get_info(total * 1_000)
        if self.can_print:
            print(f"{info}: used={used_info}, total={total_info}")
        return used_info, total_info
