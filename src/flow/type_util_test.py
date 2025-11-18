import enum
import unittest

from . import type_util

from typing import Any, TypedDict


class TestDict(TypedDict):
    a: int
    b: str


class TestProcessNode(unittest.TestCase):
    def test_type_matching(self):
        self.assertTrue(type_util.matches(1, int))
        self.assertTrue(type_util.matches(1, float))
        self.assertTrue(type_util.matches(1, int | str))
        self.assertTrue(type_util.matches(1.0, float))
        self.assertTrue(type_util.matches([1, 2], list[int]))
        self.assertTrue(type_util.matches([1, "2"], list[int | str]))
        self.assertTrue(type_util.matches({"a": 1}, dict[str, int]))
        self.assertTrue(type_util.matches(1, Any))
        self.assertTrue(type_util.matches("1", Any))
        self.assertTrue(type_util.matches((1, "2"), tuple[int, str]))

        self.assertFalse(type_util.matches(1, bool))
        self.assertFalse(type_util.matches(1, str))
        self.assertFalse(type_util.matches(1, bool | str))
        self.assertFalse(type_util.matches([1, "2"], list[int]))
        self.assertFalse(type_util.matches({"a": "1"}, dict[str, int]))
        self.assertFalse(type_util.matches({1: 1}, dict[str, int]))
        self.assertFalse(type_util.matches((1, "2"), tuple[int, int]))
        self.assertFalse(type_util.matches((1,), tuple[int, int]))
        self.assertFalse(type_util.matches((1, 2, 3), tuple[int, int]))

        # Special case: lists can match as tuple (since JSON converts tuples to
        # list, and while loading tuple just acts as a more restricted type).
        self.assertTrue(type_util.matches([1, 2], tuple[int, int]))
        self.assertTrue(type_util.matches([1, "2"], tuple[int, str]))

        self.assertFalse(type_util.matches([1, "2"], tuple[int, int]))
        # The other way is not fine.
        self.assertFalse(type_util.matches((1, 2), list[int]))

    def test_enum(self):
        class Color(enum.Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        # Accepts Enum member.
        self.assertTrue(type_util.matches(Color.RED, Color))
        # Accepts Enum value.
        self.assertTrue(type_util.matches(1, Color))
        self.assertTrue(type_util.matches(2, Color))
        # Rejects invalid value.
        self.assertFalse(type_util.matches(4, Color))
        # Rejects wrong type.
        self.assertFalse(type_util.matches("RED", Color))
        self.assertFalse(type_util.matches(None, Color))

    def test_typed_dict(self):
        x = {"a": 1, "b": "hello"}
        self.assertTrue(type_util.matches(x, TestDict))
        x = {"a": 1, "b": "hello", "c": 3}
        self.assertTrue(type_util.matches(x, TestDict))
        x = [{"a": 1, "b": "hello"}]
        self.assertTrue(type_util.matches(x, list[TestDict]))

        type_util.matches(
            {"name": "John", "age": 30}, TypedDict("Person", {"name": str, "age": int})
        )

        x = {"a": 1, "b": 2}
        self.assertFalse(type_util.matches(x, TestDict))
        x = {"a": 1}
        self.assertFalse(type_util.matches(x, TestDict))
