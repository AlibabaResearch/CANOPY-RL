"""Small parsing helpers used by the AppWorld agent loop.

No task trajectories or benchmark-specific examples are stored in this file.
"""

from __future__ import annotations

import re


_PYTHON_FENCE = re.compile(
    r"```(?:python|py)?[ \t]*\r?\n(.*?)\r?\n```",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_action_only_first_sync(llm_output: str) -> str:
    """Return the first fenced Python block, or the stripped raw response."""
    matches = _PYTHON_FENCE.findall(llm_output)
    return matches[0].strip() if matches else llm_output.strip()


async def parse_action_only_first(llm_output: str) -> str:
    """Async-compatible wrapper for the agent-loop state machine."""
    return parse_action_only_first_sync(llm_output)


async def parse_action_merge_all(llm_output: str) -> str:
    """Return all fenced Python blocks joined in their original order."""
    matches = _PYTHON_FENCE.findall(llm_output)
    return "\n".join(block.strip() for block in matches) if matches else llm_output.strip()
