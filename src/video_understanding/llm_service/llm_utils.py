import json
import logging

from . import abstract_llm

from typing import Any


def remove_thinking(response: str) -> str:
    # If there is "<think>" in a line, remove all lines until we encounter "</think>".
    lines = response.splitlines()
    result: list[str] = []
    in_thinking_block = False
    for line in lines:
        line = line.rstrip()
        if "<think>" == line.strip():
            in_thinking_block = True
            continue
        if "</think>" == line.strip():
            in_thinking_block = False
            continue
        if not in_thinking_block:
            result.append(line)
    logging.info(f"Removed {len(lines) - len(result)} thinking lines from response.")
    return "\n".join(result)


def parse_as_json(response: Any) -> Any:
    try:
        response_lines = (line.strip() for line in response.strip().splitlines())
        response = "\n".join(line for line in response_lines if line)
        if response.startswith("```json"):
            response = response[7:].strip()
        while response.endswith("`"):
            response = response[:-1]
        while response.startswith("`"):
            response = response[1:]
        logging.info(f"Parsing response as JSON: {response!r}")
        return json.loads(response)
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse response as JSON: {e}")
        raise abstract_llm.RetriableException() from e
