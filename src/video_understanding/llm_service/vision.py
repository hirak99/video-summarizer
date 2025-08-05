import base64
import io
import os

import dotenv
import openai
from PIL import Image

from . import abstract_llm
from . import openai_utils

from typing import override

_MODEL_ID = "gpt-4.1"


def to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    b64_image = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_image}"


class OpenAiVision(abstract_llm.AbstractLlm):
    def __init__(self):
        # Really this needs to be done once, but it is idempotent, and for
        # simplicity and readability we can do it here.
        dotenv.load_dotenv()

        self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @override
    def model_description(self) -> str:
        return _MODEL_ID

    @override
    def do_prompt(self, prompt: str, max_tokens: int, image_b64: str) -> str:
        return openai_utils.streamed_openai_response(
            client=self._client,
            max_completion_tokens=max_tokens,
            model=_MODEL_ID,
            messages=[
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
                }
            ],
        )
