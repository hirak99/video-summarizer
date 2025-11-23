from . import abstract_vqa
from .. import prompt_templates
from .. import video_flow_graph
from ..llm_service import abstract_llm
from ..llm_service import llm
from ..llm_service import llm_utils
from ..utils import prompt_utils
from ..utils import templater
from ..video_flow_nodes import video_flow_types

from typing import override

_OPENAI_MODEL = "gpt-4.1"


class DigestVqa(abstract_vqa.AbstractVqa):
    def __init__(self, video_path: str, maintain_context: bool) -> None:
        self._video_path = video_path
        self._maintain_context = maintain_context

        graph = video_flow_graph.VideoFlowGraph(
            program=video_flow_types.ProgramType.UNKNOWN, makeviz=False, dry_run=True
        )
        graph.persist_graph_for(video_path)

        scene_understanding = graph.scene_understanding_result()
        if scene_understanding is None:
            raise ValueError(
                "No scene_understanding data loaded. Is ENABLE_VISION == False?"
            )
        self._scene_understanding = scene_understanding

        self._role_aware_caption = graph.role_aware_captions()

        # Use self._loaded_model instead of directly accessing this.
        self._model: abstract_llm.AbstractLlm | None = None

        # Keep a buffer of things that happened for context.
        self._history: list[tuple[float, str]] = []

    @property
    def _loaded_model(self) -> abstract_llm.AbstractLlm:
        """Lazily instantiate and return the model."""
        if self._model is None:
            self._model = llm.OpenAiLlmInstance(_OPENAI_MODEL)
        return self._model

    @override
    def ask(self, time: float, question: str) -> str:
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
            transformers=[llm_utils.remove_thinking],
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
