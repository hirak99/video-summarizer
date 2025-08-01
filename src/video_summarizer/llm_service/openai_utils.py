import logging
import sys

import openai
from openai.types import chat

from typing import Iterable


def streamed_openai_response(
    *,
    client: openai.OpenAI,
    model: str,
    max_completion_tokens: int,
    messages: Iterable[chat.ChatCompletionMessageParam],
) -> str:
    """Replacement for client.responses.create.

    Except, it also logs the response to stderr in real-time.
    """
    try:
        stream = client.chat.completions.create(
            model=model,
            # messages=[{"role": "user", "content": prompt}],
            messages=messages,
            stream=True,
            max_completion_tokens=max_completion_tokens,
        )
    except openai.APIConnectionError as e:
        logging.warning(f"openai.APIConnectionError: {e}")
        return "(LLM server error)"
    tokens: list[str] = []
    logging.info(f"Streaming response to stderr:")
    try:
        for event in stream:
            token = event.choices[0].delta.content
            # Token will be None at the end.
            if token is not None:
                tokens.append(token)
                print(token, end="", flush=True)
    except openai.APIError as e:
        logging.warning(f"openai.APIError: {e}")
        # Rely on parser to catch the error, for example if JSON is expected.
        return "(LLM server error)"
    finally:
        print(file=sys.stderr)  # Newline.
    return "".join(tokens)
