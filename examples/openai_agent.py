"""
examples/openai_agent.py
========================
Minimal example of an OpenAI-powered agent that uses groundmemory to persist
information across sessions.

Run
---
    export OPENAI_API_KEY=sk-...
    uv run python examples/openai_agent.py
"""
from __future__ import annotations

import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("openai package not installed.  Run: uv add openai")
    sys.exit(1)

from groundmemory.session import MemorySession
from groundmemory.adapters.openai import get_openai_tools, handle_tool_calls, run_agent_loop

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

session = MemorySession.create("openai_example")

# Sync the workspace on startup so the index is up to date.
stats = session.sync()
print(f"[groundmemory] Synced workspace: {stats}")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# System prompt includes the bootstrap memory context
# ---------------------------------------------------------------------------

system_prompt = (
    "You are a helpful assistant with access to a persistent memory system. "
    "Use memory_write to remember important facts, memory_search to recall "
    "information, and memory_relate to record relationships between entities.\n\n"
    + session.bootstrap()
)

messages: list[dict] = [{"role": "system", "content": system_prompt}]

# ---------------------------------------------------------------------------
# Interactive chat loop
# ---------------------------------------------------------------------------

print("groundmemory + OpenAI demo.  Type 'quit' to exit.\n")

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
    # (token counts are approximate here - use tiktoken for production)
    approx_tokens = sum(len(m.get("content", "") or "") // 4 for m in messages)
    if session.should_compact(approx_tokens, 128_000):
        prompts = session.compaction_prompts()
        print("[groundmemory] Context window approaching limit - triggering memory flush.")
        messages.append({"role": "system", "content": prompts["system"]})
        messages.append({"role": "user", "content": prompts["user"]})

    messages = run_agent_loop(
        session=session,
        client=client,
        messages=messages,
        model="gpt-4.1",
        max_iterations=5,
    )

    # Find the last assistant message and print it
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                print(f"\nAssistant: {content}\n")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        print(f"\nAssistant: {block['text']}\n")
                        break
            break

session.close()