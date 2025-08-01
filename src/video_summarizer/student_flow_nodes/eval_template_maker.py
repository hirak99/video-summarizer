import json
import logging
import os

from . import hiring_highlight_curator as hhc
from .. import prompt_templates
from ...flow import process_node
from ..utils import misc_utils
from ..utils import prompt_utils
from ..utils import templater

from typing import override

# How much of the captions before and after section should be included.
_AUTO_EVAL_PRE_S = 15.0
_AUTO_EVAL_POST_S = 10.0


def _auto_eval_prompt(highlight: hhc.HighlightData) -> list[str]:
    return templater.fill(
        prompt_templates.AUTO_EVAL_PROMPT_TEMPLATE,
        {
            "movie_basename": os.path.basename(highlight.movie),
            "caption_lines": "\n".join(
                prompt_utils.caption_lines_for_prompt(
                    highlight.movie,
                    role_aware_summary=highlight.captions,
                    # TODO: Patch in the scene_understanding data if it exists.
                    scene_understanding=None,
                    start=highlight.evaluation["start"] - _AUTO_EVAL_PRE_S,
                    end=highlight.evaluation["end"] + _AUTO_EVAL_POST_S,
                ),
            ),
            "evaluation_json": json.dumps(highlight.evaluation, indent=2),
        },
    )


class EvalTermplateMaker(process_node.ProcessNode):

    @override
    def process(self, highlights_log_file: str, out_dir: str) -> str:
        with open(highlights_log_file, "r") as file:
            highlights_log = hhc.HighlightsLog.model_validate_json(file.read())

        eval_dir = os.path.join(
            out_dir,
            "auto_eval_workspace",
            os.path.basename(highlights_log.compiled_movie),
        )
        os.makedirs(eval_dir, exist_ok=True)

        # Delete all existing files in eval_dir. This is to ensure there is no
        # leftover if a movie is generated again after some changes.
        for file in os.listdir(eval_dir):
            os.remove(os.path.join(eval_dir, file))

        for index, highlight in enumerate(highlights_log.highlights):
            comment = misc_utils.slugify(highlight.evaluation["comment"])
            outfname = os.path.join(
                eval_dir, f"{index+1}.{highlight.fingerprint}.{comment}.txt"
            )
            prompt_lines = _auto_eval_prompt(highlight)
            with open(outfname, "w") as file:
                file.write("\n".join(prompt_lines))
            logging.info(f"Written to {outfname}")

        return eval_dir
