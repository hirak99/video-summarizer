from ..utils import movie_compiler
from ..video_flow_nodes import video_flow_types

# Should be set by main before proceeding to compile.
COMPILATION_TYPE = video_flow_types.CompilationType.UNKNOWN

_CAPTION_POSITIONING = movie_compiler.CaptionOptions(
    position_prop=(0.02, 0.6),  # Left, middle.
    caption_width_prop=0.3,  # 30% of the width.
    anchor="lm",  # See: https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html
    align="left",
    background_color=(0, 0, 0, 192),
)


def set_compilation_type(type: str):
    global COMPILATION_TYPE
    COMPILATION_TYPE = (  # pyright: ignore[reportConstantRedefinition]
        video_flow_types.CompilationType(type)
    )
    if COMPILATION_TYPE == video_flow_types.CompilationType.UNKNOWN:
        raise ValueError(f"Should not set compilation type to {type!r}")


def get_movie_options() -> movie_compiler.MovieOptions:
    global COMPILATION_TYPE

    # Default options.
    movie_options = movie_compiler.MovieOptions(caption=_CAPTION_POSITIONING)

    if COMPILATION_TYPE == video_flow_types.CompilationType.STUDENT_RESUME:
        # Use a pleasant saturated blue color.
        movie_options.text_color = (77, 192, 255)

    return movie_options
