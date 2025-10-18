"""OpenAI type conversions.

History
-------
The main reason for this to exist is that we started off with streaming mode
just to show the OpenAI calls more interactively, if anyone watches the log.

Later, it was discovered that some models - such as "o3" - require further
verification of the organization to allow streaming mode. We chose to fall back
for those calls to non-streaming mode. However, falling back requires type
conversion of some sort, since the message structures are different for
streaming and non-streaming query - something I had not anticipated earlier.
"""

from openai.types import chat
from openai.types import responses

from typing import Any, TypeAlias

# This is a subset of the list of unions defined in responses.ResponseInputParamItem.
_SimpleInputParamItem: TypeAlias = (
    responses.response_input_param.Message
    | responses.easy_input_message_param.EasyInputMessageParam
)
# This is a subset of the list of unions defined in responses.ResponseInputParam.
_SimpleInputParam: TypeAlias = list[_SimpleInputParamItem]


def _make_content(content: Any) -> responses.ResponseInputMessageContentListParam:
    # Example for image query -
    # "content": [
    #     {"type": "text", "text": prompt},
    #     {
    #         "type": "image_url",
    #         "image_url": {
    #             "url": image_b64,
    #             "detail": "auto",  # Default is "auto".
    #         },
    #     },
    # ],
    result: responses.ResponseInputMessageContentListParam = []
    for item in content:
        if item["type"] == "text":
            text: str = item["text"]
            result.append({"type": "input_text", "text": text})
        elif item["type"] == "image_url":
            image_url: str = item["image_url"]["url"]
            result.append(
                {
                    "type": "input_image",
                    "image_url": image_url,
                    "detail": item["image_url"]["detail"],
                }
            )
        else:
            raise ValueError(f"Unsupported type: {item['type']}")
    return result


def chatcompletion_to_responseinput(
    messages: list[chat.ChatCompletionMessageParam],
) -> responses.ResponseInputParam:
    result: _SimpleInputParam = []
    for message in messages:
        role = message["role"]
        if "content" not in message:
            raise ValueError('No "content" field in message.')

        content = message["content"]
        parsed_content: str | responses.ResponseInputMessageContentListParam
        if isinstance(content, str):
            parsed_content = content
        else:
            parsed_content = _make_content(content)
        # We don't need this anymore, prevent inadvertently using it below.
        del content

        if role == "user":
            new_message: _SimpleInputParamItem = {
                "role": "user",
                "content": parsed_content,
            }
        else:
            # Add more blocks above as we need more roles.
            raise ValueError(f"Unsupported role: {role}")
        result.append(new_message)
    # The no-op list below safely and implicitly casts all elements.
    # Without it we get a typing error since list cannot be cast as a whole.
    return [x for x in result]
