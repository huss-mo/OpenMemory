"""
groundmemory - local-first, model-agnostic persistent memory for AI agents.

Quick start
-----------
>>> from groundmemory import MemorySession
>>> session = MemorySession.create("my_project")
>>> session.sync()
>>> print(session.bootstrap())          # inject into system prompt
>>> result = session.execute_tool("memory_write", content="Alice loves Python.")
>>> result = session.execute_tool("memory_search", query="Alice")
>>> session.close()

OpenAI adapter
--------------
>>> from groundmemory.adapters.openai import get_openai_tools, run_agent_loop

Anthropic adapter
-----------------
>>> from groundmemory.adapters.anthropic import get_anthropic_tools, run_agent_loop
"""
from groundmemory.session import MemorySession
from groundmemory.config import groundmemoryConfig

__all__ = [
    "MemorySession",
    "groundmemoryConfig",
]

__version__ = "0.1.0"