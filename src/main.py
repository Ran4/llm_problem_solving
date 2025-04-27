from typing import Any, Literal, Optional, cast
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
    AskForMoreInformationArgs,
    CannotHandleTool,
    ProblemSolvedArgs,
    call_tool,
    format_tool_args_dict,
    get_solver_tools,
)


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


class SolveProblemInterrupt(BaseModel):
    asker_history: list[ResponseInputItemParam] = Field(default_factory=list)
    solver_history: list[ResponseInputItemParam] = Field(default_factory=list)


def handle_tool_calls(
    tool_calls: list[ResponseFunctionToolCall],
) -> tuple[list[ResponseInputItemParam], Optional[CannotHandleTool]]:
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

        try:
            function_call_output = call_tool(tool_call)

        except CannotHandleTool as e:
            return messages, e

        messages.append(function_call_output)

    return messages, None


def solve_problem(
    client: OpenAI,
    problem_description: str,
    interrupt: Optional[SolveProblemInterrupt] = None,
):
    """
    Tries to solve a problem by having the AI repeatedly talk to itself (
    using two agent roles: the "asker" and the "solver") until the
    problem has been solved or more information is needed.

    Returns an object that contains conversation(s) between the asker and the caller,
    and that conversation will always have a hanging function call (a message of type
    "function call" that does not have a corresponding "function_call_output"):

    * If it's of name "problem_solved", the problem has been solved.

    * If it's of name "ask_for_more_information", then you need to
    call `solve_problem` again with an answer.
    """
    solver = Agent(
        name="Solver",
        history=interrupt.solver_history if interrupt is not None else [],
        instructions="You are a problem solver",
        tools=get_solver_tools(),
    )
    asker = Agent(
        name="Asker",
        history=interrupt.asker_history
        if interrupt is not None
        else [
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
            assert caller is asker
            new_messages, cannot_handle_tool = handle_tool_calls(tool_calls)

            caller.history.extend(new_messages)
            receiver.history.extend(new_messages)

            if cannot_handle_tool:
                return SolveProblemInterrupt(
                    asker_history=asker.history,
                    solver_history=solver.history,
                )

        else:
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

    # problem_description = "Mata alla katterna"
    problem_description = "Mata alla katterna, men bara om det är söndag idag"

    interrupt = None
    while True:
        interrupt = solve_problem(
            client,
            problem_description=problem_description,
            interrupt=interrupt,
        )

        tool_call: ResponseFunctionToolCallParam = cast(
            ResponseFunctionToolCallParam,
            interrupt.solver_history[-1],
        )
        assert tool_call.get("type") == "function_call"

        if tool_call["name"] == "problem_solved":
            args = ProblemSolvedArgs.model_validate_json(tool_call["arguments"])
            print("Problem was solved!", args.explanation)
            break

        elif tool_call["name"] == "ask_for_more_information":
            args = AskForMoreInformationArgs.model_validate_json(tool_call["arguments"])
            print(
                f"More information was needed to solve the problem: {args.description}"
            )
            answer = input(f"Question: {args.question}\n> ")

            function_call_output: FunctionCallOutput = {
                "call_id": tool_call["call_id"],
                "output": answer,
                "type": "function_call_output",
            }
            interrupt.asker_history.append(function_call_output)
            interrupt.solver_history.append(function_call_output)

        else:
            raise Exception(interrupt)


if __name__ == "__main__":
    main()
