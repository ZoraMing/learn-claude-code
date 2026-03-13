"""
Provider utilities for multi-provider AI agent support.

This module provides a unified interface for multiple AI providers (Anthropic, OpenAI, Gemini),
allowing the existing agent code (v0-v4) to run unchanged.

It uses the Adapter Pattern to make OpenAI-compatible clients look exactly like
Anthropic clients to the consuming code.
"""

import os
import json
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# Data Structures (Mimic Anthropic SDK)
# =============================================================================

class ResponseWrapper:
    """Wrapper to make OpenAI responses look like Anthropic responses."""
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason

class ContentBlock:
    """Wrapper to make content blocks look like Anthropic content blocks."""
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"ContentBlock({attrs})"

# =============================================================================
# Adapters
# =============================================================================

class OpenAIAdapter:
    """
    Adapts the OpenAI client to look like an Anthropic client.
    
    Key Magic:
    self.messages = self 
    
    This allows the agent code to call:
    client.messages.create(...)
    
    which resolves to:
    adapter.create(...)
    """
    def __init__(self, openai_client):
        self.client = openai_client
        self.messages = self  # Duck typing: act as the 'messages' resource

    def create(self, model: str, system: str, messages: List[Dict], tools: List[Dict], max_tokens: int = 8000):
        """
        The core translation layer. 
        Converts Anthropic inputs -> OpenAI inputs -> OpenAI API -> Anthropic outputs.
        """
        # 1. Convert Messages (Anthropic -> OpenAI)
        openai_messages = [{"role": "system", "content": system}]
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                if isinstance(content, str):
                    # Simple text message
                    openai_messages.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    # Tool results (User role in Anthropic, Tool role in OpenAI)
                    for part in content:
                        if part.get("type") == "tool_result":
                            openai_messages.append({
                                "role": "tool",
                                "tool_call_id": part["tool_use_id"],
                                "content": part["content"] or "(no output)"
                            })
                        # Note: Anthropic user messages can also contain text+image, 
                        # but v0-v4 agents don't use that yet.

            elif role == "assistant":
                if isinstance(content, str):
                    # Simple text message
                    openai_messages.append({"role": "assistant", "content": content})
                elif isinstance(content, list):
                    # Tool calls (Assistant role)
                    # Anthropic splits thought (text) and tool_use into blocks
                    # OpenAI puts thought in 'content' and tools in 'tool_calls'
                    text_parts = []
                    tool_calls = []
                    
                    for part in content:
                        # Handle both dicts and objects (ContentBlock)
                        if isinstance(part, dict):
                            part_type = part.get("type")
                            part_text = part.get("text")
                            part_id = part.get("id")
                            part_name = part.get("name")
                            part_input = part.get("input")
                        else:
                            part_type = getattr(part, "type", None)
                            part_text = getattr(part, "text", None)
                            part_id = getattr(part, "id", None)
                            part_name = getattr(part, "name", None)
                            part_input = getattr(part, "input", None)

                        if part_type == "text":
                            text_parts.append(part_text)
                        elif part_type == "tool_use":
                            tool_calls.append({
                                "id": part_id,
                                "type": "function",
                                "function": {
                                    "name": part_name,
                                    "arguments": json.dumps(part_input)
                                }
                            })
                    
                    assistant_msg = {"role": "assistant"}
                    if text_parts:
                        assistant_msg["content"] = "\n".join(text_parts)
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls
                    
                    openai_messages.append(assistant_msg)

        # 2. Convert Tools (Anthropic -> OpenAI)
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            })

        # 3. Call OpenAI API
        # Note: Gemini/OpenAI handle max_tokens differently, but usually support the param
        response = self.client.chat.completions.create(
            model=model,
            messages=openai_messages,
            tools=openai_tools if openai_tools else None,
            max_tokens=max_tokens
        )

        # 4. Convert Response (OpenAI -> Anthropic)
        message = response.choices[0].message
        content_blocks = []

        # Extract text content
        if message.content:
            content_blocks.append(ContentBlock("text", text=message.content))

        # Extract tool calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                content_blocks.append(ContentBlock(
                    "tool_use",
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=json.loads(tool_call.function.arguments)
                ))

        # Map stop reasons: OpenAI "stop"/"tool_calls" -> Anthropic "end_turn"/"tool_use"
        # OpenAI: stop, length, content_filter, tool_calls
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif finish_reason == "stop":
            stop_reason = "end_turn"
        else:
            stop_reason = finish_reason # Fallback

        return ResponseWrapper(content_blocks, stop_reason)

# =============================================================================
# Factory Functions
# =============================================================================

def get_provider():
    """Get the current AI provider from environment variable."""
    return os.getenv("AI_PROVIDER", "anthropic").lower()

def get_client():
    """
    Return a client that conforms to the Anthropic interface.
    
    If AI_PROVIDER is 'anthropic', returns the native Anthropic client.
    Otherwise, returns an OpenAIAdapter wrapping an OpenAI-compatible client.
    """
    provider = get_provider()

    if provider == "anthropic":
        from anthropic import Anthropic
        base_url = os.getenv("ANTHROPIC_BASE_URL")
        # Return native client - guarantees 100% behavior compatibility
        return Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url=base_url
        )
    
    else:
        # For OpenAI/Gemini, we wrap the client to mimic Anthropic
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        elif provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            # Gemini OpenAI-compatible endpoint
            base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        else:
            # Generic OpenAI-compatible provider
            api_key = os.getenv(f"{provider.upper()}_API_KEY")
            base_url = os.getenv(f"{provider.upper()}_BASE_URL")

        if not api_key:
            raise ValueError(f"API Key for {provider} is missing. Please check your .env file.")

        raw_client = OpenAI(api_key=api_key, base_url=base_url)
        return OpenAIAdapter(raw_client)

def get_model():
    """Return model name from environment variable."""
    model = os.getenv("MODEL_NAME")
    if not model:
        raise ValueError("MODEL_NAME environment variable is missing. Please set it in your .env file.")
    return model