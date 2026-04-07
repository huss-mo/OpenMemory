"""
Anthropic adapter - converts groundmemory tool schemas into the format expected
by the Anthropic Messages API (``tools`` parameter).

Usage
-----
>>> import anthropic
>>> from groundmemory.session import MemorySession
>>> from groundmemory.adapters.anthropic import get_anthropic_tools, handle_tool_calls
>>>
>>> session = MemorySession.create("my_project")
>>> client  = anthropic.Anthropic()
>>>
>>> response = client.messages.create(
...     model="claude-opus-4-5",
...     max_tokens=4096,
...     system=session.bootstrap(),
...     messages=[{"role": "user", "content": "What do you remember about Alice?"}],
...     tools=get_anthropic_tools(),
... )
>>> messages, tool_results = handle_tool_calls(session, response)
"""
from __future__ import annotations

import json
from typing import Any

from groundmemory.tools import ALL_TOOLS


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------


def _to_anthropic_tool(schema: dict) -> dict:
    """Convert a single groundmemory SCHEMA dict to an Anthropic tool definition."""
    return {
        "name": schema["name"],
        "description": schema["description"],
        "input_schema": schema["parameters"],
    }


def get_anthropic_tools(names: list[str] | None = None) -> list[dict]:
    """
    Return Anthropic-formatted tool definitions for all (or a subset of) tools.

    Parameters
    ----------
    names : list[str] | None
        If provided, only include tools whose names are in this list.

    Returns
    -------
    list[dict]
        Ready to pass as ``tools=...`` in ``client.messages.create()``.
    """
    result = []
    for schema, _ in ALL_TOOLS:
        if names is None or schema["name"] in names:
            result.append(_to_anthropic_tool(schema))
    return result


# ---------------------------------------------------------------------------
# Tool-call handling
# ---------------------------------------------------------------------------


def handle_tool_calls(
    session: Any,
    response: Any,
) -> tuple[dict, dict]:
    """
    Process all tool_use blocks in an Anthropic ``Message`` response.

    Parameters
    ----------
    session  : MemorySession
    response : anthropic.types.Message

    Returns
    -------
    (assistant_turn, tool_result_turn)
        Two message dicts ready to be appended to your messages list.
        ``assistant_turn``   - the assistant message containing tool_use blocks.
        ``tool_result_turn`` - a user message containing all tool_result blocks.

    Example
    -------
    >>> asst, results = handle_tool_calls(session, response)
    >>> messages += [asst, results]
    """
    # Build the assistant turn from the raw response content blocks
    content_blocks = []
    tool_use_blocks = []

    for block in response.content:
        block_dict = block.model_dump()
        content_blocks.append(block_dict)
        if block.type == "tool_use":
            tool_use_blocks.append(block)

    assistant_turn = {"role": "assistant", "content": content_blocks}

    if not tool_use_blocks:
        return assistant_turn, {}

    # Execute each tool and collect results
    tool_results = []
    for block in tool_use_blocks:
        kwargs = block.input if isinstance(block.input, dict) else {}
        result = session.execute_tool(block.name, **kwargs)
        tool_results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            }
        )

    tool_result_turn = {"role": "user", "content": tool_results}
    return assistant_turn, tool_result_turn


def run_agent_loop(
    session: Any,
    client: Any,
    messages: list[dict],
    model: str = "claude-opus-4-5",
    max_tokens: int = 4096,
    system: str = "",
    max_iterations: int = 10,
    **create_kwargs: Any,
) -> list[dict]:
    """
    Run a simple agentic loop that keeps calling the model until it stops
    producing tool_use blocks (or ``max_iterations`` is reached).

    Parameters
    ----------
    session        : MemorySession
    client         : anthropic.Anthropic
    messages       : list[dict]    Initial message list (user turns).
    model          : str
    max_tokens     : int
    system         : str           System prompt (use session.bootstrap() here).
    max_iterations : int           Safety cap.
    **create_kwargs                Extra kwargs forwarded to ``client.messages.create``.

    Returns
    -------
    list[dict]  Final message history.
    """
    tools = get_anthropic_tools()
    # Auto-inject bootstrap if system is empty
    if not system:
        system = session.bootstrap()

    for _ in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            **create_kwargs,
        )

        asst_turn, result_turn = handle_tool_calls(session, response)
        messages.append(asst_turn)

        # Stop when the model finished without requesting tools
        if response.stop_reason != "tool_use":
            break

        if result_turn:
            messages.append(result_turn)

    return messages