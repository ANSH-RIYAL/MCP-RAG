import asyncio
import json
import os
from typing import Any, Sequence
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# Initialize server
server = Server("rag-server")

# Load business knowledge
def load_business_knowledge():
    try:
        with open("data/business_knowledge.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "Business knowledge file not found."

business_knowledge = load_business_knowledge()

def search_knowledge(query: str) -> str:
    """Simple text-based search in business knowledge."""
    query_lower = query.lower()
    knowledge_lower = business_knowledge.lower()
    
    # Simple keyword matching
    lines = business_knowledge.split('\n')
    relevant_lines = []
    
    for line in lines:
        if any(keyword in line.lower() for keyword in query_lower.split()):
            relevant_lines.append(line.strip())
    
    if relevant_lines:
        return "Relevant information:\n" + "\n".join(relevant_lines[:5])  # Limit to 5 lines
    else:
        return "No specific information found for your query. Here's general business knowledge:\n" + business_knowledge[:500] + "..."

@server.list_tools()
async def handle_list_tools() -> ListToolsResult:
    """List available RAG tools."""
    return ListToolsResult(
        tools=[
            Tool(
                name="search_business_knowledge",
                description="Search for business terms, definitions, policies, and guidelines",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for business knowledge"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_business_terms",
                description="Get definitions of common business terms",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "term": {
                            "type": "string",
                            "description": "Specific business term to look up"
                        }
                    },
                    "required": ["term"]
                }
            ),
            Tool(
                name="get_company_policies",
                description="Get information about company policies and procedures",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "policy_type": {
                            "type": "string",
                            "description": "Type of policy (e.g., 'budget', 'hiring', 'expense')"
                        }
                    },
                    "required": ["policy_type"]
                }
            )
        ]
    )

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> CallToolResult:
    """Handle tool calls for RAG functionality."""
    if arguments is None:
        arguments = {}
    
    try:
        if name == "search_business_knowledge":
            query = arguments["query"]
            result = search_knowledge(query)
            return CallToolResult(
                content=[TextContent(type="text", text=result)]
            )
        
        elif name == "get_business_terms":
            term = arguments["term"].lower()
            result = search_knowledge(f"definition of {term}")
            return CallToolResult(
                content=[TextContent(type="text", text=result)]
            )
        
        elif name == "get_company_policies":
            policy_type = arguments["policy_type"].lower()
            result = search_knowledge(f"policy {policy_type}")
            return CallToolResult(
                content=[TextContent(type="text", text=result)]
            )
        
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")]
            )
    
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")]
        )

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rag-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 