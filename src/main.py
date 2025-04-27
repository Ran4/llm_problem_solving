from typing import Any, Literal, cast
import json
from termcolor import colored
from openai import OpenAI
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseInputItemParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput
from pydantic import BaseModel, Field

from config import config
from tools import (
    CannotWorkMoreOnProblem,
    call_tool,
    format_tool_args_dict,
    get_solver_tools,
)


def _flip_role(msg: ResponseInputItemParam) -> ResponseInputItemParam:
    if msg.get("type") == "message":
        msg = cast(EasyInputMessageParam, msg)
        match msg["role"]:
            case "user":
                new_role = "assistant"
            case "assistant":
                new_role = "user"
            case other_role:
                new_role = other_role

        return EasyInputMessageParam(
            content=msg["content"],
            role=new_role,
            type="message",
        )

    else:
        return msg


def flip_roles(history: list[ResponseInputItemParam]) -> list[ResponseInputItemParam]:
    return [_flip_role(msg) for msg in history]


def yellow(s: str) -> str:
    return colored(s, "yellow")


def red(s: str) -> str:
    return colored(s, "red")


def blue(s: str) -> str:
    return colored(s, "blue")


def format_msg(msg: ResponseInputItemParam) -> str:
    match msg.get("type"):
        case "message":
            msg = cast(EasyInputMessageParam, msg)
            role = msg["role"]
            content = msg["content"]
            return f"{role}: {content}"

        case "function_call_output":
            msg = cast(FunctionCallOutput, msg)
            output = msg["output"]
            return yellow(f"--> {output}")

        case "function_call":
            msg = cast(ResponseFunctionToolCallParam, msg)

            arguments = msg["arguments"]
            name = msg["name"]

            args_dict: dict[str, Any] = json.loads(arguments)
            return red(f"{name}({format_tool_args_dict(args_dict)})")

        case type:
            return f"{type} {msg}"


class Agent(BaseModel):
    name: str
    history: list[ResponseInputItemParam] = Field(default_factory=list)
    instructions: str = ""
    tools: list[FunctionToolParam] = Field(default_factory=list)


class SolveProblemResult(BaseModel):
    result: Literal[
        "problem_solved",
        "need_more_information_or_does_not_know_how_to_solve_problem",
    ]
    description: str


def solve_problem(
    client: OpenAI,
    problem_description: str = """
Mata alla katterna.
""".strip(),
):
    solver = Agent(
        name="Solver",
        history=[],
        instructions="You are a problem solver",
        tools=get_solver_tools(),
    )
    asker = Agent(
        name="Asker",
        history=[
            EasyInputMessageParam(
                content=f"""
Lös följande problem: {problem_description}.

När du är klar, eller om du inte kan lösa problemet på annat sätt, anropa
verktyget cannot_work_more_on_problem
""".strip(),
                role="user",
                type="message",
            )
        ],
        instructions="You ask another agent if they finished their mission",
        tools=[],
    )

    caller, receiver = asker, solver

    while True:
        print(
            "\n\ninput:\n"
            + "\n".join([f"  * {format_msg(msg)}" for msg in caller.history])
        )
        response = client.responses.create(
            model="gpt-4o",
            instructions=caller.instructions,
            input=caller.history,
            tools=receiver.tools,
            store=False,
        )

        tool_calls = [
            msg for msg in response.output if isinstance(msg, ResponseFunctionToolCall)
        ]
        if tool_calls:
            for tool_call in tool_calls:
                try:
                    function_call_output = call_tool(tool_call)
                except CannotWorkMoreOnProblem as e:
                    return SolveProblemResult(
                        result=e.reason,
                        description=e.description,
                    )

                assert tool_call.id
                function_tool_call_param = ResponseFunctionToolCallParam(
                    arguments=tool_call.arguments,
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    type="function_call",
                    id=tool_call.id,
                    status="in_progress",
                )

                for history in [caller.history, receiver.history]:
                    history.append(function_tool_call_param)
                    history.append(function_call_output)
        else:
            # print("-" * 20)
            # print(f"{receiver.name}: {response.output_text}")

            agent_and_roles: list[tuple[Agent, Literal["assistant", "user"]]] = [
                (caller, "assistant"),
                (receiver, "user"),
            ]
            for agent, role in agent_and_roles:
                agent.history.append(
                    EasyInputMessageParam(
                        content=response.output_text,
                        role=role,
                        type="message",
                    )
                )

            agent_and_roles: list[tuple[Agent, Literal["assistant", "user"]]] = [
                (receiver, "assistant"),
                (caller, "user"),
            ]
            for agent, role in agent_and_roles:
                agent.history.append(
                    EasyInputMessageParam(
                        content="Har du löst problemet än?",
                        role=role,
                        type="message",
                    )
                )

            # They switch roles
            caller, receiver = receiver, caller


def main():
    client = OpenAI(
        api_key=config.OPENAI_API_KEY.get_secret_value(),
    )

    result = solve_problem(
        client,
    )
    print("Solve problem result:", result)


if __name__ == "__main__":
    main()
