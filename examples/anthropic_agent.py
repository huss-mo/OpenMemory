"""
examples/anthropic_agent.py
===========================
Minimal example of an Anthropic Claude-powered agent that uses OpenMemory to
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

from openmemory.session import MemorySession
from openmemory.adapters.anthropic import get_anthropic_tools, handle_tool_calls, run_agent_loop

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

session = MemorySession.create("anthropic_example")

# Sync the workspace on startup so the index is up to date.
stats = session.sync()
print(f"[OpenMemory] Synced workspace: {stats}")

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ---------------------------------------------------------------------------
# System prompt includes the bootstrap memory context
# ---------------------------------------------------------------------------

system_prompt = (
    "You are a helpful assistant with access to a persistent memory system. "
    "Use memory_write to remember important facts, memory_search to recall "
    "information, and memory_relate to record relationships between entities.\n\n"
    + session.bootstrap()
)

messages: list[dict] = []

# ---------------------------------------------------------------------------
# Interactive chat loop
# ---------------------------------------------------------------------------

print("OpenMemory + Anthropic demo.  Type 'quit' to exit.\n")

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

    # Check if we should compact before calling the model
    approx_tokens = sum(len(str(m.get("content", ""))) // 4 for m in messages)
    if session.should_compact(approx_tokens, 200_000):
        prompts = session.compaction_prompts()
        print("[OpenMemory] Context window approaching limit — triggering memory flush.")
        # Inject a compact request as a user turn (Anthropic doesn't support system mid-stream)
        messages.append({"role": "user", "content": prompts["user"]})

    messages = run_agent_loop(
        session=session,
        client=client,
        messages=messages,
        model="claude-opus-4-5",
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