import json
import asyncio
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
from mcp import StdioServerParameters

from .mcp_client import MCPClient
from .llm_client import LLMClient


class FlexibleRAGAgent:
    """
    RAG Agent that combines MCP tools with flexible LLM backend (Gemini or custom localhost API).
    """

    def __init__(
        self,
        mode: str = "gemini",  # "gemini" or "custom"
        gemini_api_key: str = None,
        gemini_model: str = "gemini-2.0-flash-exp",
        custom_api_url: str = "http://localhost:8000",
        custom_api_key: str = None,
        mcp_servers: List[Dict[str, Any]] = None
    ):
        """
        Initialize Flexible RAG Agent.
        
        Args:
            mode: "gemini" for Google Gemini API, "custom" for localhost API
            gemini_api_key: API key for Google Gemini (required if mode="gemini")
            gemini_model: Model name to use
            custom_api_url: URL for custom localhost API (required if mode="custom")
            custom_api_key: API key for custom API (optional)
            mcp_servers: List of MCP server configurations
        """
        self.mode = mode
        
        # Initialize LLM client
        self.llm_client = LLMClient(
            mode=mode,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            custom_api_url=custom_api_url,
            custom_api_key=custom_api_key
        )
        
        # Initialize MCP clients
        self.mcp_clients = []
        self.all_tools = {}
        self.mcp_servers_config = mcp_servers or []

    async def initialize(self):
        """Initialize MCP servers asynchronously."""
        for server_config in self.mcp_servers_config:
            await self.add_mcp_server(server_config)

    async def add_mcp_server(self, server_config: Dict[str, Any]):
        """Add an MCP server to the agent."""
        server_params = StdioServerParameters(
            command=server_config["command"],
            args=server_config.get("args", []),
            env=server_config.get("env", None)
        )
        
        mcp_client = MCPClient(server_params)
        await mcp_client.__aenter__()
        await mcp_client.initialize_tools_cache()
        
        self.mcp_clients.append(mcp_client)
        
        # Add tools to the global tools dictionary
        tools_dict = mcp_client.get_tools_dict()
        self.all_tools.update(tools_dict)

    async def close(self):
        """Close all MCP client connections and LLM client."""
        for client in self.mcp_clients:
            await client.__aexit__(None, None, None)
        await self.llm_client.close()

    async def chat(
        self, 
        message: str, 
        conversation_history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Chat with the RAG agent using the selected LLM.
        
        Args:
            message: User message
            conversation_history: Previous conversation messages
            
        Returns:
            Dictionary with response and metadata
        """
        if conversation_history is None:
            conversation_history = []

        # Convert tools to the format expected by the LLM client
        tools = []
        for tool in self.all_tools.values():
            tools.append({
                "name": tool["name"],
                "description": tool["schema"]["function"]["description"],
                "parameters": tool["schema"]["function"]["parameters"]
            })

        # Prepare messages
        messages = conversation_history.copy()
        messages.append({"role": "user", "content": message})

        # Generate response using the LLM client
        result = await self.llm_client.generate_content(
            messages=messages,
            tools=tools,
            temperature=0.1
        )

        # Handle tool calls if present
        tool_calls_made = []
        if result.get("tool_calls"):
            for tool_call in result["tool_calls"]:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                tool_calls_made.append(tool_name)
                
                # Call the tool with arguments
                tool_result = await self.all_tools[tool_name]["callable"](**tool_args)
                
                # Add tool result to conversation
                messages.append({
                    "role": "assistant", 
                    "content": f"I'll use the {tool_name} tool to help answer your question."
                })
                messages.append({
                    "role": "user", 
                    "content": f"Tool result: {tool_result}"
                })
                
                # Generate follow-up response with tool results
                follow_up_result = await self.llm_client.generate_content(
                    messages=messages,
                    tools=tools,
                    temperature=0.1
                )
                
                result = follow_up_result

        return {
            "response": result["response"],
            "tool_calls": tool_calls_made if tool_calls_made else None,
            "model": result.get("model", "unknown"),
            "mode": self.mode
        }

    async def upload_document(self, file_path: str) -> Dict[str, Any]:
        """Upload a document using the RAG MCP server."""
        if "upload_document" not in self.all_tools:
            raise ValueError("RAG MCP server not connected. No upload_document tool available.")
        
        result = await self.all_tools["upload_document"]["callable"](file_path=file_path)
        return json.loads(result) if isinstance(result, str) else result

    async def search_documents(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search documents using the RAG MCP server."""
        if "search_documents" not in self.all_tools:
            raise ValueError("RAG MCP server not connected. No search_documents tool available.")
        
        result = await self.all_tools["search_documents"]["callable"](query=query, top_k=top_k)
        return json.loads(result) if isinstance(result, str) else result

    async def list_documents(self) -> Dict[str, Any]:
        """List all indexed documents."""
        if "list_documents" not in self.all_tools:
            raise ValueError("RAG MCP server not connected. No list_documents tool available.")
        
        result = await self.all_tools["list_documents"]["callable"]()
        return json.loads(result) if isinstance(result, str) else result

    def _create_system_prompt(self) -> str:
        """Create system prompt with available tools."""
        tools_description = ""
        if self.all_tools:
            tools_description = "\n\nAvailable tools:\n" + "\n".join([
                f"- {name}: {tool['schema']['function']['description']}"
                for name, tool in self.all_tools.items()
            ])
        
        return f"""You are a helpful AI assistant that can use various tools to answer questions and perform tasks.

When a user asks a question, you can use the available tools to gather information and provide accurate answers.

{tools_description}

Always be helpful, accurate, and use the available tools when they would be useful to answer the user's question."""


# Keep the old class name for backward compatibility
GeminiRAGAgent = FlexibleRAGAgent 