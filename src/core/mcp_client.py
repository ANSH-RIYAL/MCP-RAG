import json
import asyncio
from typing import Any, List, Dict, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """
    Enhanced MCP client for connecting to multiple MCP servers.
    Supports both local and remote MCP servers.
    """

    def __init__(self, server_params: StdioServerParameters):
        """Initialize the MCP client with server parameters"""
        self.server_params = server_params
        self.session = None
        self._client = None
        self.tools_cache = {}

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def connect(self):
        """Establishes connection to MCP server"""
        self._client = stdio_client(self.server_params)
        self.read, self.write = await self._client.__aenter__()
        session = ClientSession(self.read, self.write)
        self.session = await session.__aenter__()
        await self.session.initialize()

    async def get_available_tools(self) -> List[Any]:
        """Retrieve a list of available tools from the MCP server."""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        tools = await self.session.list_tools()
        _, tools_list = tools
        _, tools_list = tools_list
        return tools_list

    async def get_available_resources(self) -> List[Any]:
        """Retrieve a list of available resources from the MCP server."""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        resources = await self.session.list_resources()
        _, resources_list = resources
        return resources_list

    def call_tool(self, tool_name: str) -> Any:
        """
        Create a callable function for a specific tool.
        Returns an async function that executes the specified tool.
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        async def callable(*args, **kwargs):
            try:
                response = await self.session.call_tool(tool_name, arguments=kwargs)
                return response.content[0].text
            except Exception as e:
                return f"Error calling tool {tool_name}: {str(e)}"

        return callable

    async def read_resource(self, uri: str) -> str:
        """Read a resource from the MCP server."""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        try:
            response = await self.session.read_resource(uri)
            return response.contents[0].text
        except Exception as e:
            return f"Error reading resource {uri}: {str(e)}"

    def get_tools_dict(self) -> Dict[str, Dict]:
        """Get tools in a format suitable for LLM function calling."""
        tools_dict = {}
        
        for tool_name, tool_info in self.tools_cache.items():
            tools_dict[tool_name] = {
                "name": tool_name,
                "callable": self.call_tool(tool_name),
                "schema": {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_info.get("description", ""),
                        "parameters": tool_info.get("inputSchema", {}),
                    },
                },
            }
        
        return tools_dict

    async def initialize_tools_cache(self):
        """Initialize the tools cache by fetching available tools."""
        tools = await self.get_available_tools()
        for tool in tools:
            if tool.name:
                self.tools_cache[tool.name] = {
                    "description": tool.description,
                    "inputSchema": tool.inputSchema,
                } 