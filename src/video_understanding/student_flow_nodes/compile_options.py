import enum


class _CompilationType(enum.StrEnum):
    UNKNOWN = "unknown"
    HIRING = "hiring"
    RESUME = "resume"


# Should be set by main before proceeding to compile.
COMPILATION_TYPE = _CompilationType.UNKNOWN


def set_compilation_type(type: str):
    global COMPILATION_TYPE
    COMPILATION_TYPE = _CompilationType(  # pyright: ignore[reportConstantRedefinition]
        type
    )
