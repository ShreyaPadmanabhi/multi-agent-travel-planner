"""Client side of the MCP integration.

`MCPToolbox` launches the travel MCP server as a stdio subprocess, discovers its
tools, and exposes them in the OpenAI function-calling format so an LLM can
choose among them in a ReAct loop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Observations are fed back into the model context, so keep them bounded.
MAX_OBSERVATION_CHARS = 2500


class MCPToolbox:
    def __init__(self, server: Any | None = None) -> None:
        # Tests may pass an in-process MCPServer; production launches a subprocess.
        self._target = server or StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.mcp.server"],
            cwd=str(PROJECT_ROOT),
        )
        self._client: Client | None = None
        self.tools: list = []

    async def __aenter__(self) -> "MCPToolbox":
        self._client = Client(self._target)
        await self._client.__aenter__()
        self.tools = (await self._client.list_tools()).tools
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._client is not None:
            await self._client.__aexit__(*exc_info)
            self._client = None

    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in self.tools]

    def openai_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            }
            for tool in self.tools
        ]

    async def call(self, name: str, arguments: dict) -> str:
        """Invoke a tool and return its result as text suitable for an observation."""
        if self._client is None:
            raise RuntimeError("MCPToolbox used outside its context manager.")
        if name not in self.tool_names:
            return f"Error: unknown tool {name!r}. Available: {', '.join(self.tool_names)}"

        result = await self._client.call_tool(name, arguments)
        if result.structured_content is not None:
            text = json.dumps(result.structured_content, default=str)
        else:
            text = "\n".join(getattr(block, "text", "") for block in result.content)
            try:
                # Tool results arrive pretty-printed; compact JSON spends fewer tokens.
                text = json.dumps(json.loads(text), separators=(",", ":"), default=str)
            except ValueError:
                pass
        if result.is_error:
            text = f"Error: {text}"
        return text[:MAX_OBSERVATION_CHARS]
