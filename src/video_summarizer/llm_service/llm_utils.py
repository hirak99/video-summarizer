import json
import logging

from typing import Any


def parse_as_json(response: Any) -> dict[str, Any] | None:
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
        return None
