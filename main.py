#!/usr/bin/env python3
"""
MCP Business Analytics + RAG Demo
Demonstrates MCP servers for business data analysis and knowledge retrieval
"""

import asyncio
import json
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

async def demo_business_analytics():
    """Demo the business analytics server."""
    print("=== Business Analytics Server Demo ===\n")
    
    async with stdio_client(["python", "src/servers/business_analytics_server.py"]) as (read, write):
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
    
    async with stdio_client(["python", "src/servers/rag_server.py"]) as (read, write):
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

async def main():
    """Run both demos."""
    print("MCP Business Analytics + RAG Demo")
    print("=" * 50)
    
    try:
        await demo_business_analytics()
        print("\n" + "=" * 50 + "\n")
        await demo_rag_server()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nYou can now ask questions like:")
        print("- 'What's the average earnings from the first quarter?'")
        print("- 'What's the correlation between sales and expenses?'")
        print("- 'What does profit margin mean?'")
        print("- 'What are the budget allocation policies?'")
        
    except Exception as e:
        print(f"Error during demo: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 