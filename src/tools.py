from typing import Any
import json

from openai.types.responses import FunctionToolParam, ResponseFunctionToolCall
from openai.types.responses.response_input_param import FunctionCallOutput
from pydantic import BaseModel, Field


class ListCatsParams(BaseModel):
    pass

    class Config:
        extra = "forbid"


list_cats_tool = FunctionToolParam(
    name="list_cats",
    description="Retrieves a list of cats, including their names",
    parameters=ListCatsParams.model_json_schema(),
    strict=True,
    type="function",
)


class FeedCatsParams(BaseModel):
    """
    Feeds a cat.
    """

    cat_name: str = Field(description="The name of the cat to feed")

    class Config:
        extra = "forbid"


feed_cat_tool = FunctionToolParam(
    name="feed_cat",
    description="Feeds the cat",
    parameters=FeedCatsParams.model_json_schema(),
    strict=True,
    type="function",
)


def get_tools():
    return [
        list_cats_tool,
        feed_cat_tool,
    ]


cats = [
    {
        "name": "Alice",
        "fed": False,
    },
    {
        "name": "Bob",
        "fed": False,
    },
    {
        "name": "Charlie",
        "fed": False,
    },
]


def format_tool_args_dict(args_dict: dict[str, Any]) -> str:
    return ",".join([f"{key}={value}" for key, value in args_dict.items()])


def list_cats() -> str:
    return json.dumps(cats)


def feed_cat(
    feed_cats_params: FeedCatsParams,
) -> str:
    for cat in cats:
        if cat["name"] == feed_cats_params.cat_name:
            cat["fed"] = True
            return f"The cat named {feed_cats_params.cat_name} was fed"

    return "There is no cat named {feed_cats_params.cat_name}!"


def call_tool(
    tool_call: ResponseFunctionToolCall,
) -> FunctionCallOutput:
    arguments = tool_call.arguments
    call_id = tool_call.call_id
    name = tool_call.name

    _ = call_id

    args_dict: dict[str, Any] = json.loads(arguments)
    print(
        f"Calling tool {name}({format_tool_args_dict(args_dict)})",
    )

    if name == "list_cats":
        result = list_cats()

    elif name == "feed_cat":
        feed_cats_params = FeedCatsParams.model_validate_json(arguments)
        result = feed_cat(feed_cats_params)

    else:
        raise Exception(f"Could not call tool {name} as it was not found")

    return {
        "call_id": call_id,
        "output": result,
        "type": "function_call_output",
    }
