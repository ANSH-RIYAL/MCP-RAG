#!/usr/bin/env python3
"""
MCP Business Analytics + RAG Demo
Demonstrates MCP servers for business data analysis and knowledge retrieval
"""

import asyncio
import json
import os
from mcp import ClientSession, StdioServerParameters, stdio_client

# LLM Configuration
LLM_MODE = os.getenv("LLM_MODE", "gemini")  # "gemini" or "custom"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CUSTOM_API_URL = os.getenv("CUSTOM_API_URL", "http://localhost:8000")
CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY")

async def demo_business_analytics():
    """Demo the business analytics server."""
    print("=== Business Analytics Server Demo ===\n")
    
    server_params = StdioServerParameters(
        command="python3",
        args=["src/servers/business_analytics_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # List available tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
            print()
            
            # Get data info
            print("1. Getting dataset information...")
            result = await session.call_tool("get_data_info", {})
            print(result.content[0].text)
            print()
            
            # Calculate mean earnings for Q1
            print("2. Calculating average earnings for Q1-2024...")
            result = await session.call_tool("calculate_mean", {
                "column": "earnings",
                "filter_column": "quarter",
                "filter_value": "Q1-2024"
            })
            print(result.content[0].text)
            print()
            
            # Calculate correlation between sales and earnings
            print("3. Calculating correlation between sales and earnings...")
            result = await session.call_tool("calculate_correlation", {
                "column1": "sales",
                "column2": "earnings"
            })
            print(result.content[0].text)
            print()
            
            # Linear regression to predict earnings from sales and employees
            print("4. Creating linear regression model to predict earnings...")
            result = await session.call_tool("linear_regression", {
                "target_column": "earnings",
                "feature_columns": ["sales", "employees"]
            })
            print(result.content[0].text)
            print()

async def demo_rag_server():
    """Demo the RAG server."""
    print("=== RAG Server Demo ===\n")
    
    server_params = StdioServerParameters(
        command="python3",
        args=["src/servers/rag_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # List available tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
            print()
            
            # Search for business terms
            print("1. Searching for 'earnings' definition...")
            result = await session.call_tool("get_business_terms", {"term": "earnings"})
            print(result.content[0].text)
            print()
            
            # Get company policies
            print("2. Getting budget allocation policy...")
            result = await session.call_tool("get_company_policies", {"policy_type": "budget"})
            print(result.content[0].text)
            print()
            
            # Search business knowledge
            print("3. Searching for profit margin information...")
            result = await session.call_tool("search_business_knowledge", {"query": "profit margin calculation"})
            print(result.content[0].text)
            print()

async def demo_llm_integration():
    """Demo LLM integration with MCP tools."""
    print("=== LLM Integration Demo ===\n")
    
    # Check LLM configuration
    print(f"LLM Mode: {LLM_MODE}")
    if LLM_MODE == "gemini":
        if not GEMINI_API_KEY:
            print("⚠️  GEMINI_API_KEY not set. Skipping LLM demo.")
            return
        print(f"Using Gemini API with model: gemini-2.0-flash-exp")
    elif LLM_MODE == "custom":
        print(f"Using Custom API at: {CUSTOM_API_URL}")
    print()
    
    try:
        from src.core.gemini_rag_agent import FlexibleRAGAgent
        
        # Configure MCP servers
        mcp_servers = [
            {
                "command": "python3",
                "args": ["src/servers/business_analytics_server.py"]
            },
            {
                "command": "python3", 
                "args": ["src/servers/rag_server.py"]
            }
        ]
        
        # Initialize RAG agent
        agent = FlexibleRAGAgent(
            mode=LLM_MODE,
            gemini_api_key=GEMINI_API_KEY,
            custom_api_url=CUSTOM_API_URL,
            custom_api_key=CUSTOM_API_KEY,
            mcp_servers=mcp_servers
        )
        
        await agent.initialize()
        
        # Demo conversation
        print("Demo conversation:")
        print("User: 'What's the average earnings from Q1 and what does earnings mean?'")
        
        response = await agent.chat("What's the average earnings from Q1 and what does earnings mean?")
        
        print(f"Agent ({response['mode']} mode): {response['response']}")
        if response.get('tool_calls'):
            print(f"Tools used: {response['tool_calls']}")
        
        await agent.close()
        
    except ImportError:
        print("⚠️  LLM integration not available. Install required dependencies.")
    except Exception as e:
        print(f"⚠️  LLM demo failed: {e}")

async def main():
    """Run the demos."""
    print("MCP Business Analytics + RAG Demo")
    print("=" * 50)
    print(f"LLM Mode: {LLM_MODE}")
    print("=" * 50)
    
    try:
        await demo_business_analytics()
        print("\n" + "=" * 50 + "\n")
        await demo_rag_server()
        print("\n" + "=" * 50 + "\n")
        await demo_llm_integration()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nYou can now ask questions like:")
        print("- 'What's the average earnings from the first quarter?'")
        print("- 'What's the correlation between sales and expenses?'")
        print("- 'What does profit margin mean?'")
        print("- 'What are the budget allocation policies?'")
        
        print(f"\nLLM Configuration:")
        print(f"- Mode: {LLM_MODE}")
        if LLM_MODE == "gemini":
            print(f"- API Key: {'Set' if GEMINI_API_KEY else 'Not set'}")
        elif LLM_MODE == "custom":
            print(f"- API URL: {CUSTOM_API_URL}")
            print(f"- API Key: {'Set' if CUSTOM_API_KEY else 'Not set'}")
        
    except Exception as e:
        print(f"Error during demo: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 