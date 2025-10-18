import logging
import sys

import httpx
import openai
from openai.types import chat
from openai.types import responses

from . import abstract_llm
from . import openai_type_helper

# Default is True, which uses streaming mode to echo responses to stderr.
_USE_STREAMING_ALWAYS = True

# These model(s) raise error, saying that the organization must be verified if streaming is attempted.
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
    messages: list[chat.ChatCompletionMessageParam],
) -> str:
    """Replacement for client.responses.create.

    Except, it echoes the response to stderr in as it comes in real time.
    """
    if not _USE_STREAMING_ALWAYS or model in _NON_STREAMABLE_MODELS:
        # TODO: Implement compile-time checks if possible?
        # I think run-time checks will be done by OpenAI.
        # These two are very similar.
        return _non_streamed_openai_response(
            client, model, openai_type_helper.chatcompletion_to_responseinput(messages)
        )

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
    except (openai.APIConnectionError, httpx.RemoteProtocolError) as e:
        logging.warning(f"Error {e}")
        raise abstract_llm.RetriableException() from e
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
        raise abstract_llm.RetriableException() from e
    finally:
        print(file=sys.stderr)  # Newline.
    return "".join(tokens)
