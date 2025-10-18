import base64
import io
import os

import dotenv
import openai
from openai.types import chat
from openai.types import responses
from PIL import Image

from . import abstract_llm
from . import openai_utils

from typing import override


def to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    b64_image = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_image}"


class _ModelWithOpenAiApi(abstract_llm.AbstractLlm):
    def __init__(self, client: openai.OpenAI, model_id: str, system_prompt: str | None):
        self._client = client
        self._model_id = model_id
        self._system_prompt = system_prompt

    @override
    def model_description(self) -> str:
        return self._model_id

    @override
    def do_prompt(
        self,
        prompt: str,
        max_tokens: int,
        image_b64: str,
    ) -> str:
        messages: (
            list[chat.ChatCompletionMessageParam] | responses.ResponseInputParam
        ) = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_b64,
                            "detail": "auto",  # Default is "auto".
                        },
                    },
                ],
            },
        ]
        if self._system_prompt is not None:
            messages.insert(0, {"role": "system", "content": self._system_prompt})
        return openai_utils.streamed_openai_response(
            client=self._client,
            max_completion_tokens=max_tokens,
            model=self._model_id,
            messages=messages,
        )


class OpenAiVision(_ModelWithOpenAiApi):
    def __init__(self, model_id: str, system_prompt: str | None = None):
        # Really this needs to be done once, but it is idempotent, and for
        # simplicity and readability we can do it here.
        dotenv.load_dotenv()
        super().__init__(
            openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
            model_id,
            system_prompt=system_prompt,
        )


class OllamaVision(_ModelWithOpenAiApi):
    def __init__(self, model_id: str, system_prompt: str | None = None):
        super().__init__(
            openai.OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required, but unused
            ),
            model_id,
            system_prompt=system_prompt,
        )
