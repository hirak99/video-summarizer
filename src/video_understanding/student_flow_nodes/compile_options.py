from ..utils import movie_compiler
from ..video_flow_nodes import video_flow_types

_MOVIE_OPTIONS_BY_PROGRAM: dict[
    video_flow_types.ProgramType, movie_compiler.MovieOptions
] = {
    video_flow_types.ProgramType.PMA: movie_compiler.MovieOptions(
        resize_to=(1920, 1080),
        caption=movie_compiler.CaptionOptions(
            position_prop=(0.02, 0.6),  # Left, middle.
            caption_width_prop=0.3,  # 30% of the width.
            anchor="lm",  # See: https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html
            align="left",
            background_color=(0, 0, 0, 192),
        ),
        text_title_pos=(50, 50),
        text_desc_pos=(50, 100),
        text_color=(255, 165, 0),
    ),
    video_flow_types.ProgramType.FTP: movie_compiler.MovieOptions(
        resize_to=(1920, 1080),
        caption=movie_compiler.CaptionOptions(
            position_prop=(0.5, 0.99),
            caption_width_prop=0.95,
            anchor="md",
            align="center",
            background_color=(0, 0, 0, 192),
        ),
        text_title_pos=(10, 5),
        text_desc_pos=(10, 55),
        text_color=(255, 165, 0),
    ),
}


def get_movie_options(
    program: video_flow_types.ProgramType,
    movie_type: video_flow_types.CompilationType,
) -> movie_compiler.MovieOptions:

    movie_options = _MOVIE_OPTIONS_BY_PROGRAM[program]

    # Further specific tweaks.
    if movie_type == video_flow_types.CompilationType.STUDENT_RESUME:
        # Use a pleasant saturated blue color.
        movie_options.text_color = (77, 192, 255)

    return movie_options
