"""
OpenMemory — local-first, model-agnostic persistent memory for AI agents.

Quick start
-----------
>>> from openmemory import MemorySession
>>> session = MemorySession.create("my_project")
>>> session.sync()
>>> print(session.bootstrap())          # inject into system prompt
>>> result = session.execute_tool("memory_write", content="Alice loves Python.")
>>> result = session.execute_tool("memory_search", query="Alice")
>>> session.close()

OpenAI adapter
--------------
>>> from openmemory.adapters.openai import get_openai_tools, run_agent_loop

Anthropic adapter
-----------------
>>> from openmemory.adapters.anthropic import get_anthropic_tools, run_agent_loop
"""
from openmemory.session import MemorySession
from openmemory.config import OpenMemoryConfig

__all__ = [
    "MemorySession",
    "OpenMemoryConfig",
]

__version__ = "0.1.0"