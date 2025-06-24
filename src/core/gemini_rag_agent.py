import json
import asyncio
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
from mcp import StdioServerParameters

from .mcp_client import MCPClient


class GeminiRAGAgent:
    """
    RAG Agent that combines MCP tools with Google Gemini for document search and Q&A.
    """

    def __init__(
        self,
        gemini_api_key: str = "AIzaSyB-d7vpvd2W8kXyVmfjn7XJNiZmDNP6hHM",
        gemini_model: str = "gemini-2.0-flash-exp",
        mcp_servers: List[Dict[str, Any]] = None
    ):
        """
        Initialize Gemini RAG Agent.
        
        Args:
            gemini_api_key: API key for Google Gemini
            gemini_model: Model name to use
            mcp_servers: List of MCP server configurations
        """
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        
        # Initialize Gemini client
        self.gemini_client = genai.Client(api_key=gemini_api_key)
        
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
        """Close all MCP client connections."""
        for client in self.mcp_clients:
            await client.__aexit__(None, None, None)

    async def chat(
        self, 
        message: str, 
        conversation_history: List[types.Content] = None
    ) -> Dict[str, Any]:
        """
        Chat with the RAG agent using Gemini.
        
        Args:
            message: User message
            conversation_history: Previous conversation messages
            
        Returns:
            Dictionary with response and metadata
        """
        if conversation_history is None:
            conversation_history = []

        # Convert tools to Gemini function declarations format
        tool_declarations = []
        for tool in self.all_tools.values():
            # Convert OpenAI format to Gemini format
            parsed_parameters = json.loads(
                json.dumps(tool["schema"]["function"]["parameters"])
                .replace("object", "OBJECT")
                .replace("string", "STRING")
                .replace("number", "NUMBER")
                .replace("boolean", "BOOLEAN")
                .replace("array", "ARRAY")
                .replace("integer", "INTEGER")
            )
            declaration = types.FunctionDeclaration(
                name=tool["name"],
                description=tool["schema"]["function"]["description"],
                parameters=parsed_parameters,
            )
            tool_declarations.append(declaration)

        # Create system instruction
        system_instruction = self._create_system_prompt()

        # Initialize generation config
        generation_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
            tools=[types.Tool(function_declarations=tool_declarations)] if tool_declarations else None,
        )

        # Prepare conversation
        contents = conversation_history.copy()
        contents.append(types.Content(role="user", parts=[types.Part(text=message)]))

        # Generate response
        response = self.gemini_client.models.generate_content(
            model=self.gemini_model,
            config=generation_config,
            contents=contents,
        )

        # Handle tool calls if present
        tool_calls_made = []
        for part in response.candidates[0].content.parts:
            contents.append(types.Content(role="model", parts=[part]))
            
            if part.function_call:
                function_call = part.function_call
                tool_calls_made.append(function_call.name)
                
                # Call the tool with arguments
                tool_result = await self.all_tools[function_call.name]["callable"](
                    **function_call.args
                )
                
                # Build the response parts
                function_response_part = types.Part.from_function_response(
                    name=function_call.name,
                    response={"result": tool_result},
                )
                contents.append(types.Content(role="user", parts=[function_response_part]))
                
                # Send follow-up with tool results
                func_gen_response = self.gemini_client.models.generate_content(
                    model=self.gemini_model, 
                    config=generation_config, 
                    contents=contents
                )
                contents.append(types.Content(role="model", parts=[func_gen_response.candidates[0].content.parts]))

        # Extract final response text
        final_response_text = ""
        for part in contents[-1].parts:
            if hasattr(part, 'text') and part.text:
                final_response_text += part.text

        return {
            "response": final_response_text,
            "tool_calls": tool_calls_made if tool_calls_made else None,
            "conversation_history": contents
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

        return f"""You are a helpful AI assistant with access to document search and retrieval capabilities. You can help users find information in their documents and answer questions based on the content.

{tools_description}

Instructions:
1. When users ask questions about documents, use the search_documents tool to find relevant information
2. When users want to upload documents, use the upload_document tool
3. When users want to see what documents are available, use the list_documents tool
4. Always provide helpful, accurate responses based on the available information
5. If you don't have access to the information requested, let the user know
6. Be conversational and helpful in your responses

Remember to use the appropriate tools when needed to provide the best possible assistance.""" 