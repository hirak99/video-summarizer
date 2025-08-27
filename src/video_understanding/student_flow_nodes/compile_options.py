from ..utils import movie_compiler
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


def get_movie_options() -> movie_compiler.MovieOptions:
    global COMPILATION_TYPE

    # Default options.
    movie_options = movie_compiler.MovieOptions()

    if COMPILATION_TYPE == student_eval_type.CompilationType.STUDENT_RESUME:
        # Use a pleasant saturated blue color.
        movie_options.text_color = (77, 192, 255)

    return movie_options
