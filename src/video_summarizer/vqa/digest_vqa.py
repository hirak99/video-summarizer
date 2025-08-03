import json

from . import abstract_vqa
from .. import prompt_templates
from .. import video_flow_executor
from ..llm_service import abstract_llm
from ..llm_service import llm
from ..utils import prompt_utils
from ..utils import templater
from ..video_flow_nodes import role_based_captioner
from ..video_flow_nodes import vision_processor

from typing import override

_OPENAI_MODEL = "gpt-4.1"


class DigestVqa(abstract_vqa.AbstractVqa):
    def __init__(self, video_path: str, maintain_context: bool) -> None:
        self._video_path = video_path
        self._maintain_context = maintain_context

        executor = video_flow_executor.VideoFlowExecutor(makeviz=False, dry_run=True)
        executor.persist_graph_for(video_path)

        graph_results = executor.graph.results_dict
        if not graph_results:
            raise ValueError(f"Could not laod results for {video_path!r}")

        self._scene_understanding: vision_processor.SceneListT | None = None
        self._role_aware_caption: (
            list[role_based_captioner.RoleAwareCaptionT] | None
        ) = None

        # TODO: Make this easier to get without knowing details of process_graph.
        for _, item in graph_results.items():
            if item["name"] == "VisionProcess":
                vision_digest_file = item["output"]
                with open(vision_digest_file) as f:
                    self._scene_understanding = (
                        vision_processor.SceneListT.model_validate_json(f.read())
                    )
            elif item["name"] == "RoleBasedCaptionsNode":
                role_aware_caption_file = item["output"]
                with open(role_aware_caption_file) as f:
                    self._role_aware_caption = json.load(f)

        if self._scene_understanding is None:
            raise ValueError(f"Could not load scene understanding")
        if self._role_aware_caption is None:
            raise ValueError(f"Could not load role aware captions")

        # Use self._loaded_model instead of directly accessing this.
        self._model: abstract_llm.AbstractLlm | None = None

        # Keep a buffer of things that happened for context.
        self._history: list[tuple[float, str]] = []

    @property
    def _loaded_model(self) -> abstract_llm.AbstractLlm:
        if self._model is None:
            self._model = llm.OpenAiLlmInstance(_OPENAI_MODEL)
        return self._model

    @override
    def ask(self, time: float, question: str) -> str:
        assert self._scene_understanding is not None
        assert self._role_aware_caption is not None

        caption_for_prompt = prompt_utils.caption_lines_for_prompt(
            self._video_path,
            self._role_aware_caption,
            self._scene_understanding,
            end=time,
        )

        prompt: list[str] = templater.fill(
            prompt_templates.DIGEST_VQA_PROMPT_TEMPLATE,
            {
                "caption_lines": "\n".join(caption_for_prompt),
                "history": "\n".join(x[1] for x in self._history),
                "question": question,
            },
        )

        response: str = self._loaded_model.do_prompt_and_parse(
            "\n".join(prompt),
            transformers=[llm.remove_thinking],
            max_tokens=4096,
        )

        if self._maintain_context:
            # Remember previous questions and answers.

            if self._history and self._history[-1][0] != time:
                # Reset context if time changes.
                self._history = []

            self._history += [
                (time, f"Previous Question: {question}"),
                (time, f"Previous Answer: {response}"),
            ]

        return response
