import unittest

from ..utils import interval_scanner


class TestIntervalScanner(unittest.TestCase):
    def test_simple_case(self):
        intervals = interval_scanner.IntervalScanner(
            [
                {"interval": (1, 2)},
            ]
        )
        self.assertEqual(intervals.containing_timestamp(0.5), [])
        self.assertEqual(intervals.containing_timestamp(1.5), [{"interval": (1, 2)}])
        self.assertEqual(intervals.containing_timestamp(2.5), [])

    def test_case2(self):
        intervals = interval_scanner.IntervalScanner(
            [
                {"interval": (10, 20)},
                {"interval": (0, 100)},
                {"interval": (11, 30)},
            ]
        )
        self.assertEqual(
            intervals.containing_timestamp(0.5),
            [
                {"interval": (0, 100)},
            ],
        )
        self.assertEqual(
            intervals.containing_timestamp(10.5),
            [
                {"interval": (10, 20)},
                {"interval": (0, 100)},
            ],
        )
        self.assertEqual(
            intervals.containing_timestamp(15),
            [
                {"interval": (10, 20)},
                {"interval": (0, 100)},
                {"interval": (11, 30)},
            ],
        )
        self.assertEqual(
            intervals.containing_timestamp(25),
            [
                {"interval": (11, 30)},
                {"interval": (0, 100)},
            ],
        )

    def test_non_monotonic(self):
        intervals = interval_scanner.IntervalScanner(
            [
                {"interval": (1, 2)},
            ]
        )
        self.assertEqual(intervals.containing_timestamp(1.5), [{"interval": (1, 2)}])
        self.assertEqual(intervals.containing_timestamp(1.5), [{"interval": (1, 2)}])
        with self.assertRaises(ValueError):
            intervals.containing_timestamp(0.5)

    def test_overlapping(self):
        intervals = interval_scanner.IntervalScanner(
            [
                {"interval": (10, 20)},
                {"interval": (0, 100)},
                {"interval": (11, 30)},
            ]
        )
        self.assertEqual(
            intervals.overlapping_intervals(9, 10.5),
            [
                {"interval": (10, 20)},
                {"interval": (0, 100)},
            ],
        )
        self.assertEqual(
            intervals.overlapping_intervals(10.5, 11.5),
            [
                {"interval": (10, 20)},
                {"interval": (0, 100)},
                {"interval": (11, 30)},
            ],
        )
        self.assertEqual(
            intervals.overlapping_intervals(15, 50),
            [
                {"interval": (10, 20)},
                {"interval": (0, 100)},
                {"interval": (11, 30)},
            ],
        )
        self.assertEqual(
            intervals.overlapping_intervals(55, 150),
            [
                {"interval": (0, 100)},
            ],
        )
