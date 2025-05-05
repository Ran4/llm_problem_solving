from pathlib import Path
from typing import Any, Optional, cast
import re
import io
from uuid import uuid4
import json
from base64 import b64encode, b64decode

from PIL import Image
from openai import OpenAI
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputContentParam,
)
from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    ResponseInputItemParam,
)
from pydantic import BaseModel, Field
from openai.types.responses import (
    FunctionToolParam,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
)

from config import config
from tools import format_tool_args_dict


class CropImageParams(BaseModel):
    file_id: str
    x: int
    y: int
    w: int
    h: int

    class Config:
        extra = "forbid"


crop_image_tool = FunctionToolParam(
    name="crop_image",
    description="Crops an image",
    parameters=CropImageParams.model_json_schema(),
    strict=True,
    type="function",
)


Metadata = dict[str, Any]


class LLMContext(BaseModel):
    messages_and_metadatas: list[
        tuple[ResponseInputItemParam, list[Optional[Metadata]]]
    ]
    new_messages: list[ResponseInputItemParam] = Field(
        default_factory=list, frozen=False
    )

    def find_image_by_file_id(
        self,
        file_id: str,
    ) -> str:
        """
        Find image in the previous messages.
        """
        images: list[str] = []
        for message, metadatas in self.messages_and_metadatas:
            if "content" not in message:
                continue

            content = cast(
                list[ResponseInputContentParam],
                message["content"],
            )

            for content_param, metadata in zip(content, metadatas):
                if (
                    content_param["type"] == "input_image"
                    and (metadata or {}).get("image_file_id") == file_id
                ):
                    assert (
                        "image_url" in content_param
                    ), "Found ResponseInputImageParam without image_url"
                    image = content_param["image_url"]
                    if image is None:
                        raise Exception(
                            "Found ResponseInputImageParam with empty image_url"
                        )
                    images.append(image)

        # There must always be exactly one image!
        match images:
            case [image]:
                return image

            case []:
                raise Exception(f"Could not find image with file_id {file_id}")

            case images:
                raise Exception(
                    f"Found multiple images with file_id {file_id}",
                )


def _crop_image(image_url: str, x: int, y: int, w: int, h: int) -> str:
    # Parse the header and base64 data
    match = re.match(r"data:(image/\w+);base64,(.*)", image_url)
    if not match:
        raise ValueError("Invalid image URL format")

    mime_type, base64_data = match.groups()
    image_format = mime_type.split("/")[1].upper()
    if image_format == "JPG":
        image_format = "JPEG"  # Pillow uses "JPEG" instead of "JPG"

    # Decode and load the image
    image_data = b64decode(base64_data)
    image = Image.open(io.BytesIO(image_data))

    # Crop the image
    cropped = image.crop((x, y, x + w, y + h))

    # Save the cropped image
    output_buffer = io.BytesIO()
    cropped.save(output_buffer, format=image_format)
    cropped_base64 = b64encode(output_buffer.getvalue()).decode("utf-8")

    return f"data:{mime_type};base64,{cropped_base64}"


def crop_image(
    in_: CropImageParams,
    ctx: LLMContext,
):
    print("In crop_image, in_:", in_.model_dump(mode="json"))

    image_url = ctx.find_image_by_file_id(in_.file_id)

    new_image = _crop_image(
        image_url=image_url,
        x=in_.x,
        y=in_.y,
        w=in_.w,
        h=in_.h,
    )

    ctx.new_messages.append(
        {
            "content": [
                {
                    "detail": "auto",
                    "type": "input_image",
                    "file_id": None,
                    "image_url": new_image,
                }
            ],
            "role": "user",
        }
    )
    return f"Cropped image {in_}"


def get_tools():
    return [
        crop_image_tool,
    ]


def call_tool(
    tool_call: ResponseFunctionToolCall,
    ctx: LLMContext,
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
        result = crop_image(args, ctx)

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
    messages_and_metadatas: list[
        tuple[ResponseInputItemParam, list[Optional[Metadata]]]
    ],
):
    messages = []

    ctx = LLMContext(
        messages_and_metadatas=messages_and_metadatas,
    )

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

        function_call_output = call_tool(tool_call, ctx)

        messages.append(function_call_output)

    return messages + ctx.new_messages


def save_image_url_to_file(image_url: str, output_path: Path | str) -> None:
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    # Parse MIME type and base64 data
    match = re.match(r"data:(image/\w+);base64,(.*)", image_url)
    if not match:
        raise ValueError("Invalid image URL format")

    mime_type, base64_data = match.groups()
    file_ext = mime_type.split("/")[1]
    full_output_path = output_path / f"image.{file_ext}"

    # Decode and save to file
    image_data = b64decode(base64_data)
    with open(full_output_path, "wb") as f:
        f.write(image_data)

    print(f"Image saved to {full_output_path}")


def cropper(
    client: OpenAI,
):
    tools: list[FunctionToolParam] = get_tools()

    image_base64 = b64encode(
        open("data/two-doggos.webp", "rb").read(),
    ).decode("utf-8")

    image_file_id = uuid4()
    messages: list[ResponseInputItemParam] = [
        {
            "content": [
                {"type": "input_text", "text": "Please crop out the right dog"},
            ],
            "role": "user",
        },
        {
            "content": [
                {
                    "type": "input_text",
                    "text": f"(The next image has file_id '{image_file_id}')",
                },
                {
                    "detail": "auto",
                    "type": "input_image",
                    "image_url": f"data:image/webp;base64,{image_base64}",
                },
            ],
            "role": "user",
        },
    ]
    metadatas = [[None], [None, {"image_file_id": str(image_file_id)}]]

    messages_and_metadatas = list(zip(messages, metadatas))

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
        new_messages = handle_tool_calls(tool_calls, messages_and_metadatas)
        messages.extend(new_messages)

    print(response.output_text)

    match messages:
        case *_, {"content": [{"image_url": image_url}]}:
            resulting_image = image_url

        case _:
            resulting_image = None

    if resulting_image:
        save_image_url_to_file(resulting_image, "/tmp")


if __name__ == "__main__":
    client = OpenAI(
        api_key=config.OPENAI_API_KEY.get_secret_value(),
    )
    cropper(client)
