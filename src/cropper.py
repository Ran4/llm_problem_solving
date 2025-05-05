from typing import Any
from openai import OpenAI
import json
from base64 import b64encode
from openai.types.responses.response_input_param import FunctionCallOutput
from pydantic import BaseModel
from openai.types.responses import (
    FunctionToolParam,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseInputImageParam,
)

from config import config
from tools import format_tool_args_dict


class CropImageParams(BaseModel):
    pass

    class Config:
        extra = "forbid"


crop_image_tool = FunctionToolParam(
    name="crop_image",
    description="Crops an image",
    parameters=CropImageParams.model_json_schema(),
    strict=True,
    type="function",
)


def crop_image(
    in_: CropImageParams,
):
    return "such very cropped image"


def get_tools():
    return [
        crop_image_tool,
    ]


def call_tool(
    tool_call: ResponseFunctionToolCall,
) -> FunctionCallOutput:
    """
    Raises:
        CannotCallTool
    """
    arguments = tool_call.arguments
    call_id = tool_call.call_id
    name = tool_call.name

    _ = call_id

    args_dict: dict[str, Any] = json.loads(arguments)
    print(
        f"Calling tool {name}({format_tool_args_dict(args_dict)})",
    )

    if name == "crop_image":
        args = CropImageParams.model_validate(args_dict)
        result = crop_image(args)

    else:
        raise Exception(
            f"Could not call tool {name}({format_tool_args_dict(args_dict)}) as it was not found"  # noqa
        )

    return {
        "call_id": call_id,
        "output": result,
        "type": "function_call_output",
    }


def handle_tool_calls(
    tool_calls: list[ResponseFunctionToolCall],
):
    messages = []
    for tool_call in tool_calls:
        assert tool_call.id
        function_tool_call_param = ResponseFunctionToolCallParam(
            arguments=tool_call.arguments,
            call_id=tool_call.call_id,
            name=tool_call.name,
            type="function_call",
            id=tool_call.id,
            status="in_progress",
        )

        messages.append(function_tool_call_param)

        function_call_output = call_tool(tool_call)

        messages.append(function_call_output)

    return messages


def cropper(
    client: OpenAI,
):
    tools: list[FunctionToolParam] = get_tools()

    image_base64 = b64encode(
        open("data/two-doggos.webp", "rb").read(),
    ).decode("utf-8")

    messages = [
        {
            "content": [
                {"type": "input_text", "text": "Please crop out the right dog"},
            ],
            "role": "user",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": f"data:image/webp;base64,{image_base64}",
                }
            ],
        },
    ]

    response = client.responses.create(
        model="gpt-4o",
        instructions="You crop input images",
        input=messages,
        tools=tools,
        store=False,
    )

    tool_calls = [
        msg for msg in response.output if isinstance(msg, ResponseFunctionToolCall)
    ]

    if tool_calls:
        new_messages = handle_tool_calls(tool_calls)
        messages.extend(new_messages)

    print(response.output_text)


if __name__ == "__main__":
    client = OpenAI(
        api_key=config.OPENAI_API_KEY.get_secret_value(),
    )
    cropper(client)
