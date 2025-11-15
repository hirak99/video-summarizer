from ..utils import movie_compiler
from ..video_flow_nodes import video_flow_types

_CAPTION_POSITIONING = movie_compiler.CaptionOptions(
    position_prop=(0.02, 0.6),  # Left, middle.
    caption_width_prop=0.3,  # 30% of the width.
    anchor="lm",  # See: https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html
    align="left",
    background_color=(0, 0, 0, 192),
)


def get_movie_options(
    movie_type: video_flow_types.CompilationType,
) -> movie_compiler.MovieOptions:

    # Default options.
    movie_options = movie_compiler.MovieOptions(
        resize_to=(1920, 1080),
        caption=_CAPTION_POSITIONING,
    )

    if movie_type == video_flow_types.CompilationType.STUDENT_RESUME:
        # Use a pleasant saturated blue color.
        movie_options.text_color = (77, 192, 255)

    return movie_options
