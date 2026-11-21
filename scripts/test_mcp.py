"""Launch the MCP server over stdio, list its tools and call a few of them."""

import asyncio
import pathlib
import sys
from datetime import date, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.mcp import MCPToolbox


async def main() -> None:
    async with MCPToolbox() as toolbox:
        print("TOOLS:", ", ".join(toolbox.tool_names))
        print("\ngeocode:", await toolbox.call("geocode", {"query": "Kyoto"}))
        print("\nfind_route:", await toolbox.call("find_route", {"origin": "Paris", "destination": "Lyon"}))
        start = date.today() + timedelta(days=3)
        print(
            "\nget_weather:",
            await toolbox.call(
                "get_weather",
                {"destination": "Rome", "start_date": str(start), "end_date": str(start + timedelta(days=2))},
            ),
        )


asyncio.run(main())
