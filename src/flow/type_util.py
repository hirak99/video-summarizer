import typing
from typing import Any, Union
import enum

def matches(obj, typ) -> bool:
    """
    Runtime type checking utility that should have been in `typing`.

    Recursively checks if an object matches a given type annotation.

    Supports standard types, Union, Any, list, set, dict, tuple, and TypedDict.
    Special handling:
    - int: Allows float values (but not bool).
    - tuple: Allows list instances if elements and length match (due to JSON).
    - TypedDict: Validates all keys and their types as defined in the TypedDict, allows extra keys.

    Examples:
        type_util.matches(1, int)  # True
        type_util.matches(1, float)  # True (special case)
        type_util.matches([1, "2"], list[int | str])  # True
        type_util.matches({"name": "John", "age": 30},
            TypedDict("Person", {"name": str, "age": int}))  # True

        type_util.matches(1.0, int)  # False
        type_util.matches({"name": ["John", "Doe"], "age": 30}, dict[str, int | str])  # False
    """
    origin = typing.get_origin(typ)
    args = typing.get_args(typ)

    # Special case: allow float where int is expected.
    if typ is float:
        return isinstance(obj, (int, float)) and not isinstance(obj, bool)

    if origin is None:
        if typ is Any:
            return True
        if isinstance(typ, typing._TypedDictMeta):  # type: ignore
            if not isinstance(obj, dict):
                return False
            # Handle TypedDict strictly.
            for key, key_typ in typ.__annotations__.items():
                if key not in obj or not matches(obj[key], key_typ):
                    return False
            return True
        # Enum support.
        if isinstance(typ, type) and issubclass(typ, enum.Enum):
            # Accept either an Enum member or a valid value for the Enum.
            if isinstance(obj, typ):
                return True
            # Accept the value if it matches any Enum member's value.
            return any(obj == member.value for member in typ)
        # Simple type, like int, str, etc.
        return isinstance(obj, typ)
    elif origin is Union:
        return any(matches(obj, arg) for arg in args)
    elif origin is tuple:
        # JSON convertgs tuple as list. Allow list to match against tuple.
        if not isinstance(obj, (list, tuple)):
            return False
        if len(obj) != len(args):
            return False
        return all(matches(item, arg) for item, arg in zip(obj, args))
    elif origin in {list, set}:
        if not isinstance(obj, origin):
            return False
        if not args:
            return True  # No inner type specified.
        return all(matches(item, args[0]) for item in obj)
    elif origin is dict:
        if not isinstance(obj, dict):
            return False
        key_type, val_type = args
        return all(
            matches(k, key_type) and matches(v, val_type) for k, v in obj.items()
        )

    # Fallback: Try normal isinstance.
    return isinstance(obj, typ)
