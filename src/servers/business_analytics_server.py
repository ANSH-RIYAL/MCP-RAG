import asyncio
import json
import pandas as pd
import numpy as np
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
server = Server("business-analytics")

# Load data
try:
    df = pd.read_csv("data/sample_business_data.csv")
except FileNotFoundError:
    df = pd.DataFrame()

@server.list_tools()
async def handle_list_tools() -> ListToolsResult:
    """List available business analytics tools."""
    return ListToolsResult(
        tools=[
            Tool(
                name="get_data_info",
                description="Get information about the business dataset including column names and data types",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            Tool(
                name="calculate_mean",
                description="Calculate the mean (average) of a numeric column",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "Name of the column to calculate mean for"
                        },
                        "filter_column": {
                            "type": "string",
                            "description": "Optional: Column to filter by (e.g., 'quarter')"
                        },
                        "filter_value": {
                            "type": "string",
                            "description": "Optional: Value to filter for (e.g., 'Q1-2024')"
                        }
                    },
                    "required": ["column"]
                }
            ),
            Tool(
                name="calculate_std",
                description="Calculate the standard deviation of a numeric column",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "Name of the column to calculate standard deviation for"
                        },
                        "filter_column": {
                            "type": "string",
                            "description": "Optional: Column to filter by"
                        },
                        "filter_value": {
                            "type": "string",
                            "description": "Optional: Value to filter for"
                        }
                    },
                    "required": ["column"]
                }
            ),
            Tool(
                name="calculate_correlation",
                description="Calculate correlation between two numeric columns",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "column1": {
                            "type": "string",
                            "description": "First column name"
                        },
                        "column2": {
                            "type": "string",
                            "description": "Second column name"
                        },
                        "filter_column": {
                            "type": "string",
                            "description": "Optional: Column to filter by"
                        },
                        "filter_value": {
                            "type": "string",
                            "description": "Optional: Value to filter for"
                        }
                    },
                    "required": ["column1", "column2"]
                }
            ),
            Tool(
                name="linear_regression",
                description="Create a linear regression model to predict one column from others",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "target_column": {
                            "type": "string",
                            "description": "Column to predict (dependent variable)"
                        },
                        "feature_columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of columns to use as features (independent variables)"
                        },
                        "filter_column": {
                            "type": "string",
                            "description": "Optional: Column to filter by"
                        },
                        "filter_value": {
                            "type": "string",
                            "description": "Optional: Value to filter for"
                        }
                    },
                    "required": ["target_column", "feature_columns"]
                }
            )
        ]
    )

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> CallToolResult:
    """Handle tool calls for business analytics."""
    if arguments is None:
        arguments = {}
    
    try:
        if name == "get_data_info":
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Dataset Info:\nColumns: {list(df.columns)}\nShape: {df.shape}\nData Types:\n{df.dtypes.to_string()}\n\nSample Data:\n{df.head().to_string()}"
                    )
                ]
            )
        
        elif name == "calculate_mean":
            column = arguments["column"]
            filter_column = arguments.get("filter_column")
            filter_value = arguments.get("filter_value")
            
            data = df.copy()
            if filter_column and filter_value:
                data = data[data[filter_column] == filter_value]
            
            if column not in data.columns:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: Column '{column}' not found")]
                )
            
            if not pd.api.types.is_numeric_dtype(data[column]):
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: Column '{column}' is not numeric")]
                )
            
            mean_val = data[column].mean()
            return CallToolResult(
                content=[TextContent(type="text", text=f"Mean of {column}: {mean_val:.2f}")]
            )
        
        elif name == "calculate_std":
            column = arguments["column"]
            filter_column = arguments.get("filter_column")
            filter_value = arguments.get("filter_value")
            
            data = df.copy()
            if filter_column and filter_value:
                data = data[data[filter_column] == filter_value]
            
            if column not in data.columns:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: Column '{column}' not found")]
                )
            
            if not pd.api.types.is_numeric_dtype(data[column]):
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: Column '{column}' is not numeric")]
                )
            
            std_val = data[column].std()
            return CallToolResult(
                content=[TextContent(type="text", text=f"Standard deviation of {column}: {std_val:.2f}")]
            )
        
        elif name == "calculate_correlation":
            column1 = arguments["column1"]
            column2 = arguments["column2"]
            filter_column = arguments.get("filter_column")
            filter_value = arguments.get("filter_value")
            
            data = df.copy()
            if filter_column and filter_value:
                data = data[data[filter_column] == filter_value]
            
            if column1 not in data.columns or column2 not in data.columns:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: One or both columns not found")]
                )
            
            if not pd.api.types.is_numeric_dtype(data[column1]) or not pd.api.types.is_numeric_dtype(data[column2]):
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: One or both columns are not numeric")]
                )
            
            corr = data[column1].corr(data[column2])
            return CallToolResult(
                content=[TextContent(type="text", text=f"Correlation between {column1} and {column2}: {corr:.3f}")]
            )
        
        elif name == "linear_regression":
            target_column = arguments["target_column"]
            feature_columns = arguments["feature_columns"]
            filter_column = arguments.get("filter_column")
            filter_value = arguments.get("filter_value")
            
            data = df.copy()
            if filter_column and filter_value:
                data = data[data[filter_column] == filter_value]
            
            # Check if all columns exist
            all_columns = [target_column] + feature_columns
            missing_columns = [col for col in all_columns if col not in data.columns]
            if missing_columns:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: Columns not found: {missing_columns}")]
                )
            
            # Check if all columns are numeric
            non_numeric = [col for col in all_columns if not pd.api.types.is_numeric_dtype(data[col])]
            if non_numeric:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: Non-numeric columns: {non_numeric}")]
                )
            
            # Simple linear regression using numpy
            X = data[feature_columns].values
            y = data[target_column].values
            
            # Add intercept
            X = np.column_stack([np.ones(X.shape[0]), X])
            
            # Calculate coefficients using normal equation
            try:
                coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
                intercept = coefficients[0]
                feature_coeffs = coefficients[1:]
                
                # Calculate R-squared
                y_pred = X @ coefficients
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                result = f"Linear Regression Results:\n"
                result += f"Target: {target_column}\n"
                result += f"Features: {feature_columns}\n"
                result += f"Intercept: {intercept:.2f}\n"
                for i, feature in enumerate(feature_columns):
                    result += f"{feature} coefficient: {feature_coeffs[i]:.2f}\n"
                result += f"R-squared: {r_squared:.3f}"
                
                return CallToolResult(
                    content=[TextContent(type="text", text=result)]
                )
            
            except np.linalg.LinAlgError:
                return CallToolResult(
                    content=[TextContent(type="text", text="Error: Cannot perform linear regression (singular matrix)")]
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
                server_name="business-analytics",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 