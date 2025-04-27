from typing import Any, Literal, Union
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


class ProblemSolvedArgs(BaseModel):
    explanation: str

    class Config:
        extra = "forbid"


problem_solved_tool = FunctionToolParam(
    name="problem_solved",
    description="Call this whenever you've finished the problem",
    parameters=ProblemSolvedArgs.model_json_schema(),
    strict=True,
    type="function",
)


class AskForMoreInformationArgs(BaseModel):
    """
    Either more information is needed or we do not know how to solve the problem
    """

    description: str = Field(
        description="Why we need to know this to solve the problem"
    )
    question: str = Field(
        description="""
A question or request that a user can answer to help solve the problem,
such as "What year is it today?" or "Give me an image of a dog"
""".strip(),
    )

    class Config:
        extra = "forbid"


ask_for_more_information_tool = FunctionToolParam(
    name="ask_for_more_information",
    description="Call this to ask for more information",
    parameters=AskForMoreInformationArgs.model_json_schema(),
    strict=True,
    type="function",
)


def get_solver_tools():
    return [
        list_cats_tool,
        feed_cat_tool,
        problem_solved_tool,
        ask_for_more_information_tool,
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


class CannotHandleTool(Exception):
    pass
    # def __init__(self, reason: Union[ProblemSolved, NeedMoreInformation]) -> None:
    #     self.reason: Union[ProblemSolved, NeedMoreInformation] = reason


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

    if name == "list_cats":
        result = list_cats()

    elif name == "feed_cat":
        feed_cats_params = FeedCatsParams.model_validate_json(arguments)
        result = feed_cat(feed_cats_params)

    elif name == "problem_solved":
        raise CannotHandleTool()

    elif name == "ask_for_more_information":
        raise CannotHandleTool()

    else:
        raise Exception(
            f"Could not call tool {name}({format_tool_args_dict(args_dict)}) as it was not found"  # noqa
        )

    return {
        "call_id": call_id,
        "output": result,
        "type": "function_call_output",
    }
