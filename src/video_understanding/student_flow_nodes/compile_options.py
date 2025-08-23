from ..video_flow_nodes import student_eval_type

# Should be set by main before proceeding to compile.
COMPILATION_TYPE = student_eval_type.CompilationType.UNKNOWN


def set_compilation_type(type: str):
    global COMPILATION_TYPE
    COMPILATION_TYPE = student_eval_type.CompilationType(  # pyright: ignore[reportConstantRedefinition]
        type
    )
    if COMPILATION_TYPE == student_eval_type.CompilationType.UNKNOWN:
        raise ValueError(f"Should not set compilation type to {type!r}")
