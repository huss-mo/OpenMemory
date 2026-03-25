"""
OpenAI adapter - converts groundmemory tool schemas into the format expected by
the OpenAI Chat Completions ``tools`` parameter (function-calling API).

Usage
-----
>>> from openai import OpenAI
>>> from groundmemory.session import MemorySession
>>> from groundmemory.adapters.openai import get_openai_tools, handle_tool_calls
>>>
>>> session = MemorySession.create("my_project")
>>> client  = OpenAI()
>>>
>>> response = client.chat.completions.create(
...     model="gpt-4o",
...     messages=[{"role": "system", "content": session.bootstrap()},
...               {"role": "user",   "content": "What do you remember about Alice?"}],
...     tools=get_openai_tools(),
... )
>>> messages = handle_tool_calls(session, response, messages)
"""
from __future__ import annotations

import json
from typing import Any

from groundmemory.tools import ALL_TOOLS, TOOL_RUNNERS


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------


def _to_openai_tool(schema: dict) -> dict:
    """Convert a single groundmemory SCHEMA dict to an OpenAI tool definition."""
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["parameters"],
        },
    }


def get_openai_tools(names: list[str] | None = None) -> list[dict]:
    """
    Return OpenAI-formatted tool definitions for all (or a subset of) tools.

    Parameters
    ----------
    names : list[str] | None
        If provided, only include tools whose names are in this list.
        Useful when you want to expose only a subset of memory tools.

    Returns
    -------
    list[dict]
        Ready to pass as ``tools=...`` in ``client.chat.completions.create()``.
    """
    result = []
    for schema, _ in ALL_TOOLS:
        if names is None or schema["name"] in names:
            result.append(_to_openai_tool(schema))
    return result


# ---------------------------------------------------------------------------
# Tool-call handling
# ---------------------------------------------------------------------------


def handle_tool_calls(
    session: Any,
    response: Any,
    messages: list[dict],
) -> list[dict]:
    """
    Process all tool calls in an OpenAI ``ChatCompletion`` response.

    Executes each tool call against *session*, appends the assistant message
    and all tool result messages to *messages*, and returns the updated list.

    Parameters
    ----------
    session  : MemorySession
    response : openai.types.chat.ChatCompletion
    messages : list[dict]   The current conversation message list (mutated in place).

    Returns
    -------
    list[dict]  The same *messages* list with new entries appended.
    """
    choice = response.choices[0]
    assistant_msg = choice.message

    # Append the assistant turn (includes tool_calls)
    messages.append(assistant_msg.model_dump(exclude_unset=True))

    if not assistant_msg.tool_calls:
        return messages

    for tc in assistant_msg.tool_calls:
        fn = tc.function
        try:
            kwargs = json.loads(fn.arguments) if fn.arguments else {}
        except json.JSONDecodeError:
            kwargs = {}

        result = session.execute_tool(fn.name, **kwargs)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            }
        )

    return messages


def run_agent_loop(
    session: Any,
    client: Any,
    messages: list[dict],
    model: str = "gpt-4o",
    max_iterations: int = 10,
    **create_kwargs: Any,
) -> list[dict]:
    """
    Run a simple agentic loop that keeps calling the model until it stops
    producing tool calls (or ``max_iterations`` is reached).

    Parameters
    ----------
    session        : MemorySession
    client         : openai.OpenAI
    messages       : list[dict]    Initial message list (system + user turns).
    model          : str
    max_iterations : int           Safety cap to prevent infinite loops.
    **create_kwargs                Extra kwargs forwarded to ``client.chat.completions.create``.

    Returns
    -------
    list[dict]  Final message history.
    """
    from groundmemory.bootstrap.compaction import should_flush, get_compaction_prompts

    tools = get_openai_tools()
    compaction_cfg = session.config.compaction
    _flushed = False  # only flush once per loop

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            **create_kwargs,
        )
        choice = response.choices[0]
        messages = handle_tool_calls(session, response, messages)

        # Check compaction threshold and inject a flush turn if needed
        if not _flushed and compaction_cfg.enabled:
            usage = response.usage
            used = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
            if should_flush(used, compaction_cfg):
                prompts = get_compaction_prompts(compaction_cfg)
                messages.append({"role": "user", "content": prompts["user"]})
                # One dedicated flush turn - let the model write to memory
                flush_response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": prompts["system"]}] + messages,
                    tools=tools,
                    **create_kwargs,
                )
                messages = handle_tool_calls(session, flush_response, messages)
                _flushed = True

        # Stop when the model is done calling tools
        if choice.finish_reason != "tool_calls":
            break

    return messages
