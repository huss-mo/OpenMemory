"""
examples/anthropic_agent.py
===========================
Minimal example of an Anthropic Claude-powered agent that uses groundmemory to
persist information across sessions.

Run
---
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run python examples/anthropic_agent.py
"""
from __future__ import annotations

import os
import sys

try:
    import anthropic
except ImportError:
    print("anthropic package not installed.  Run: uv add anthropic")
    sys.exit(1)

from groundmemory.session import MemorySession
from groundmemory.adapters.anthropic import get_anthropic_tools, handle_tool_calls, run_agent_loop

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

session = MemorySession.create("anthropic_example")

# Sync the workspace on startup so the index is up to date.
stats = session.sync()
print(f"[groundmemory] Synced workspace: {stats}")

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ---------------------------------------------------------------------------
# System prompt includes the bootstrap memory context
# ---------------------------------------------------------------------------

system_prompt = (
    "You are a helpful assistant with access to a persistent memory system. "
    "Use memory_write to remember important facts, memory_read to recall "
    "information, and memory_relate to record relationships between entities.\n\n"
    + session.bootstrap()
)

messages: list[dict] = []

# ---------------------------------------------------------------------------
# Interactive chat loop
# ---------------------------------------------------------------------------

print("groundmemory + Anthropic demo.  Type 'quit' to exit.\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        break

    if not user_input:
        continue
    if user_input.lower() in {"quit", "exit", "q"}:
        break

    messages.append({"role": "user", "content": user_input})

    messages = run_agent_loop(
        session=session,
        client=client,
        messages=messages,
        model="claude-opus-4-6",
        max_tokens=4096,
        system=system_prompt,
        max_iterations=5,
    )

    # Find the last assistant message and print any text blocks
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                print(f"\nAssistant: {content}\n")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        print(f"\nAssistant: {block['text']}\n")
                        break
            break

session.close()