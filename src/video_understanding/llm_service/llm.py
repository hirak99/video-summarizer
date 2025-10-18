import json
import logging
import os
import sys

import dotenv
import openai
import requests

from . import abstract_llm
from . import local_server
from . import openai_utils

from typing import Final, override

# DO NOT SET THIS TO FALSE HERE.
# Use `--no-start-llm-server` command line argument to disable auto-starting.
AUTO_START_SERVER = True

_LLAMA_PORT = 8080


def _query_llama(
    prompt,
    max_tokens: int,
) -> str:
    server_url = f"http://localhost:{_LLAMA_PORT}"
    # Payload Documentation: https://platform.openai.com/docs/api-reference/responses-streaming/response/incomplete
    payload = {
        "prompt": prompt,
        "stream": True,
        # Note: OpenAI recommends max_completion_tokens, but llama.cpp still uses max_tokens.
        "max_tokens": max_tokens,
    }
    # try:
    #     response = requests.post(f"{server_url}/completion", json=payload)
    #     response.raise_for_status()
    #     return response.json().get("content", "")
    try:
        with requests.post(
            f"{server_url}/completion", json=payload, stream=True
        ) as response:
            response.raise_for_status()
            response_chunks: list[str] = []
            logging.info(f"Streaming response to stderr:")
            for line in response.iter_lines():
                if line:
                    assert line.startswith(
                        b"data: "
                    ), f"Unexpected line format: {line!r}"
                    json_str = line[len(b"data: ") :]
                    data = json.loads(json_str)
                    print(data["content"], file=sys.stderr, end="", flush=True)
                    response_chunks.append(data["content"])
        print(file=sys.stderr)  # Newline.
        return "".join(response_chunks)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return ""


# Returns None if the response is not valid JSON.
def _qwen_prompt(prompt: str) -> str:
    """Formats the prompt for Qwen."""
    # Qwen format:
    #
    # <|im_start|>system
    # You are a helpful assistant<|im_end|>
    # <|im_start|>user
    # Hello<|im_end|>
    # <|im_start|>assistant
    # Hi there<|im_end|>
    # <|im_start|>user
    # How are you?<|im_end|>
    # <|im_start|>assistant

    user_start_token = "<|im_start|>user"
    assistant_start_token = "<|im_start|>assistant"
    end_token = "<|im_end|>"
    prompt_lines = [user_start_token, prompt, end_token, assistant_start_token]
    return "\n".join(prompt_lines)


# Should make this instantiable with server address, and add convenience method
# for initializing with known servers.
class LocalLlmInstance(abstract_llm.AbstractLlm):
    def __init__(
        self,
        model_config: local_server.ModelConfig = local_server.MODEL_QWEN3_30B_A3B_Q4_K_M,
    ):
        # The server needs to be running before we can query it.
        self._server_instance = local_server.LocalServer(model_config)
        if AUTO_START_SERVER:
            self._server_instance.start(_LLAMA_PORT)

    @override
    def model_description(self) -> str:
        """Returns the model ID for the current instance."""
        return os.path.basename(self._server_instance.model_config.desc)

    def _decorate_prompt(self, prompt: str) -> str:
        """Decorates the prompt for the model."""
        if (
            self._server_instance.model_config
            == local_server.MODEL_QWEN3_30B_A3B_Q4_K_M
        ):
            return _qwen_prompt(prompt)
        raise NotImplementedError(
            f"Unknown how to decorate prompt for model: {self.model_description()}"
        )

    @override
    def do_prompt(self, prompt: str, **kwargs) -> str:
        response = _query_llama(self._decorate_prompt(prompt), **kwargs)
        return response

    @override
    def finalize(self):
        self._server_instance.terminate()


class OpenAiLlmInstance(abstract_llm.AbstractLlm):
    def __init__(self, model_id: str):
        # Really this needs to be done once, but it is idempotent, and for
        # simplicity and readability we can do it here.
        dotenv.load_dotenv()

        self.model_id: Final[str] = model_id
        self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @override
    def model_description(self) -> str:
        """Returns the model ID for the current instance."""
        return f"OpenAI {self.model_id}"

    # TODO: Implement @retry(exceptions=..., delay=..., tries=...)
    @override
    def do_prompt(self, prompt: str, max_tokens: int) -> str:
        return openai_utils.streamed_openai_response(
            client=self._client,
            max_completion_tokens=max_tokens,
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
        )


# Example usage
def _example_usage():
    # instance = LocalLlmInstance()
    instance = OpenAiLlmInstance("gpt-3.5-turbo")
    logging.info("Sending Query 1")
    prompt = "Explain the theory of relativity in one sentence."
    print("Response:\n", instance.do_prompt(prompt, max_tokens=1024))
    # logging.info("Sending Query 2")
    # prompt = "For a quick test, what did I just ask now?"
    # print("Response:\n", instance.do_prompt(prompt))
    instance.finalize()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _example_usage()
