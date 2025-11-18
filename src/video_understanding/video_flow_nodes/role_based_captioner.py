import json
import logging

from . import word_caption_utils
from ...flow import process_node

from typing import override, TypedDict

FILE_SUFFIX = ".role_aware_captions.json"


# Probably should move to video_nodes.
class RoleAwareCaptionT(TypedDict):
    speaker: str
    text: str
    interval: tuple[float, float]


class RoleBasedCaptionsNode(process_node.ProcessNode):
    @override
    def process(
        self,
        word_captions_file: str,
        identified_roles: dict[str, str],
        out_file_stem: str,
    ) -> str:
        out_file = out_file_stem + FILE_SUFFIX
        with open(word_captions_file) as f:
            diarized_captions = json.load(f)

        logging.info(f"{identified_roles=}")
        captions = word_caption_utils.merge_word_captions(
            diarized_captions,
            speaker_aliases=identified_roles,
            unknown="Either",
        )
        with open(out_file, "w") as f:
            json.dump(captions, f)
        logging.info(f"Written to {out_file}")
        return out_file
