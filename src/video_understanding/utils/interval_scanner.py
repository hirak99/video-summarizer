"""Efficient monotonic scanning of intervals.

For any given timestamp, find all (possibly overlapping) entries of intervals which contain the timestamp.

The timestamp is assumed to be monotonically increasing, which yields efficiency.
If this assumption did not hold, we'd need to use interval_tree data structure which is a little less efficient.
"""

import heapq
import itertools

from typing import Generic, Sequence, TypedDict, TypeVar

_EPSILON = 1e-5


# Each dict inside intervals must have "interval": [start, end].
class _IntervalT(TypedDict):
    interval: tuple[float, float]

    # There may be other keys in intervals, this type is not closed.


T = TypeVar("T", bound=_IntervalT)


class IntervalScanner(Generic[T]):
    def __init__(
        self,
        intervals: Sequence[T],
    ):
        self._intervals = sorted(intervals, key=lambda x: x["interval"][0])
        self._last_start_time = float("-inf")
        self.reset()

    def reset(self) -> None:
        self._index = 0
        self._last_start_time = float("-inf")
        # List of currently active intervals, i.e. start-time >= last scan time.
        # We will maintain this as a heap based on end time to pop at O(log(n)).
        # The first entry is then guaranteed to be the one with minimum end-time.
        self._active = []
        # Used to tie-break in case there are duplicate end-time in the heap.
        self._counter = itertools.count()

    def overlapping_intervals(self, start: float, end: float) -> list[T]:
        """Search for intervals overlapping with (start, end)."""
        if start < self._last_start_time - _EPSILON:
            raise ValueError(
                f"Scans must be monotonic, but {start} < last scan time {self._last_start_time}"
            )
        self._last_start_time = start
        # Mark intervals starting before the end as active (and remove for next scan).
        while self._index < len(self._intervals):
            this_interval = self._intervals[self._index]
            this_start, this_end = this_interval["interval"]
            if this_start > end + _EPSILON:
                break
            heapq.heappush(self._active, (this_end, next(self._counter), this_interval))
            self._index += 1

        # Drop intevals ending before the start time.
        while self._active and self._active[0][0] < start - _EPSILON:
            heapq.heappop(self._active)
        return [interval for _, _, interval in self._active]

    def containing_timestamp(self, time: float) -> list[T]:
        """Search for intervals which contain the given timestamp."""
        return self.overlapping_intervals(time, time)
