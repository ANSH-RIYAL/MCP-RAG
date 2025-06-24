import json
import asyncio
import httpx
from typing import Any, Dict, List, Optional, Union
from google import genai
from google.genai import types

class LLMClient:
    """
    Flexible LLM client that supports both Gemini API and custom localhost API.
    """
    
    def __init__(
        self,
        mode: str = "gemini",  # "gemini" or "custom"
        gemini_api_key: str = None,
        gemini_model: str = "gemini-2.0-flash-exp",
        custom_api_url: str = "http://localhost:8000",
        custom_api_key: str = None
    ):
        """
        Initialize LLM client.
        
        Args:
            mode: "gemini" for Google Gemini API, "custom" for localhost API
            gemini_api_key: API key for Google Gemini (required if mode="gemini")
            gemini_model: Gemini model name
            custom_api_url: URL for custom localhost API (required if mode="custom")
            custom_api_key: API key for custom API (optional)
        """
        self.mode = mode
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.custom_api_url = custom_api_url
        self.custom_api_key = custom_api_key
        
        # Initialize appropriate client
        if mode == "gemini":
            if not gemini_api_key:
                raise ValueError("gemini_api_key is required when mode='gemini'")
            self.gemini_client = genai.Client(api_key=gemini_api_key)
        elif mode == "custom":
            self.http_client = httpx.AsyncClient()
        else:
            raise ValueError("mode must be either 'gemini' or 'custom'")
    
    async def generate_content(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Generate content using the selected LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: List of tool definitions (optional)
            temperature: Generation temperature
            
        Returns:
            Dictionary with response and metadata
        """
        if self.mode == "gemini":
            return await self._generate_gemini(messages, tools, temperature)
        elif self.mode == "custom":
            return await self._generate_custom(messages, tools, temperature)
    
    async def _generate_gemini(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """Generate content using Gemini API."""
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            if msg["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=msg["content"])]))
            elif msg["role"] == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=msg["content"])]))
        
        # Convert tools to Gemini format if provided
        tool_declarations = []
        if tools:
            for tool in tools:
                # Convert OpenAI format to Gemini format
                parsed_parameters = json.loads(
                    json.dumps(tool.get("parameters", {}))
                    .replace("object", "OBJECT")
                    .replace("string", "STRING")
                    .replace("number", "NUMBER")
                    .replace("boolean", "BOOLEAN")
                    .replace("array", "ARRAY")
                    .replace("integer", "INTEGER")
                )
                declaration = types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=parsed_parameters,
                )
                tool_declarations.append(declaration)
        
        # Create generation config
        generation_config = types.GenerateContentConfig(
            temperature=temperature,
            tools=[types.Tool(function_declarations=tool_declarations)] if tool_declarations else None,
        )
        
        # Generate response
        response = self.gemini_client.models.generate_content(
            model=self.gemini_model,
            config=generation_config,
            contents=contents,
        )
        
        # Extract response
        response_text = ""
        tool_calls = []
        
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                response_text += part.text
            if hasattr(part, 'function_call') and part.function_call:
                tool_calls.append({
                    "name": part.function_call.name,
                    "arguments": part.function_call.args
                })
        
        return {
            "response": response_text,
            "tool_calls": tool_calls if tool_calls else None,
            "model": self.gemini_model
        }
    
    async def _generate_custom(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """Generate content using custom localhost API."""
        # Prepare request payload (assuming DeepSeek R1 format)
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1000,
            "stream": False
        }
        
        # Add tools if provided
        if tools:
            payload["tools"] = tools
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json"
        }
        if self.custom_api_key:
            headers["Authorization"] = f"Bearer {self.custom_api_key}"
        
        # Make request
        try:
            response = await self.http_client.post(
                f"{self.custom_api_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract response (assuming OpenAI-compatible format)
            choice = result["choices"][0]
            message = choice["message"]
            
            response_text = message.get("content", "")
            tool_calls = message.get("tool_calls", []) if message.get("tool_calls") else None
            
            return {
                "response": response_text,
                "tool_calls": tool_calls,
                "model": result.get("model", "custom-model")
            }
            
        except httpx.HTTPError as e:
            raise Exception(f"Custom API request failed: {e}")
        except Exception as e:
            raise Exception(f"Error calling custom API: {e}")
    
    async def close(self):
        """Close HTTP client if using custom mode."""
        if self.mode == "custom" and hasattr(self, 'http_client'):
            await self.http_client.aclose() 