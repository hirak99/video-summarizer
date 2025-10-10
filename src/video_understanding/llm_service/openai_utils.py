import logging
import sys

import openai
from openai.types import chat
from openai.types import responses

from typing import cast

# These model(s) raises an error saying that the organization must be verified if streaming is attempted.
_NON_STREAMABLE_MODELS = {"o3"}


def _non_streamed_openai_response(
    client: openai.OpenAI,
    model: str,
    messages: responses.ResponseInputParam,
) -> str:

    response = client.responses.create(
        model=model,
        input=messages,
        # reasoning={"effort": "high"},  # you can choose "low", "medium", "high"
        background=False,
    )
    logging.info(f"Query response: {response.output_text}")
    return response.output_text


def streamed_openai_response(
    *,
    client: openai.OpenAI,
    model: str,
    max_completion_tokens: int,
    messages: list[chat.ChatCompletionMessageParam] | responses.ResponseInputParam,
) -> str:
    """Replacement for client.responses.create.

    Except, it echoes the response to stderr in as it comes in real time.
    """
    if model in _NON_STREAMABLE_MODELS:
        # TODO: Implement compile-time checks if possible?
        # I think run-time checks will be done by OpenAI.
        # These two are very similar.
        messages = cast(responses.ResponseInputParam, messages)
        return _non_streamed_openai_response(client, model, messages)

    messages = cast(list[chat.ChatCompletionMessageParam], messages)
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_completion_tokens=max_completion_tokens,
        )
    except openai.BadRequestError as e:
        if "must be verified to stream" in e.message:
            logging.error("Note to developer: Try adding the model to _NON_STREAMABLE.")
        raise  # Re-raise.
    except openai.APIConnectionError as e:
        # TODO: This should be thrown as an exception, and handled appropriately.
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
