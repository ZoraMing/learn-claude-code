"""
多提供商 AI 代理支持的提供商工具库。

本模块为多个 AI 提供商（Anthropic, OpenAI, Gemini）提供统一接口，
允许现有的代理代码（v0-v4）无需修改即可运行。

它使用适配器模式（Adapter Pattern），使兼容 OpenAI 协议的客户端
在调用代码面前表现得与 Anthropic 客户端完全一致。
"""

import os
import json
from typing import Dict, List
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# =============================================================================
# 数据结构（模拟 Anthropic SDK）
# =============================================================================

class ResponseWrapper:
    """包装器，使 OpenAI 的响应看起来像 Anthropic 的响应。"""
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason

class ContentBlock:
    """包装器，使内容块（Content Block）看起来像 Anthropic 的内容块。"""
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"ContentBlock({attrs})"

# =============================================================================
# 适配器
# =============================================================================

class OpenAIAdapter:
    """
    将 OpenAI 客户端适配为 Anthropic 客户端的形式。
    
    关键魔法：
    self.messages = self 
    
    这允许代理代码调用：
    client.messages.create(...)
    
    实际上会解析为调用：
    adapter.create(...)
    """
    def __init__(self, openai_client):
        self.client = openai_client
        self.messages = self  # 鸭子类型：充当 'messages' 资源

    def create(self, model: str, system: str, messages: List[Dict], tools: List[Dict], max_tokens: int = 8000):
        """
        核心转换层。
        转换流程：Anthropic 输入 -> OpenAI 输入 -> OpenAI API -> Anthropic 输出。
        """
        # 1. 转换消息格式（Anthropic -> OpenAI）
        openai_messages = [{"role": "system", "content": system}]
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                if isinstance(content, str):
                    # 普通文本消息
                    openai_messages.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    # 工具结果（Anthropic 中角色为 User，OpenAI 中角色为 Tool）
                    for part in content:
                        if part.get("type") == "tool_result":
                            openai_messages.append({
                                "role": "tool",
                                "tool_call_id": part["tool_use_id"],
                                "content": part["content"] or "(no output)"
                            })
                        # 注意：Anthropic 用户消息也可以包含文本+图像，
                        # 但 v0-v4 代理暂未使用该功能。

            elif role == "assistant":
                if isinstance(content, str):
                    # 普通文本消息
                    openai_messages.append({"role": "assistant", "content": content})
                elif isinstance(content, list):
                    # 工具调用（助手角色）
                    # Anthropic 将思考（文本）和工具使用（tool_use）拆分为不同的块
                    # OpenAI 将思考放在 'content' 中，工具调用放在 'tool_calls' 中
                    text_parts = []
                    tool_calls = []
                    
                    for part in content:
                        # 处理字典（dict）和对象（ContentBlock）两种情况
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

        # 2. 转换工具定义（Anthropic -> OpenAI）
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

        # 3. 调用 OpenAI API
        # 注意：Gemini/OpenAI 处理 max_tokens 的方式不同，但通常都支持该参数
        response = self.client.chat.completions.create(
            model=model,
            messages=openai_messages,
            tools=openai_tools if openai_tools else None,
            max_tokens=max_tokens
        )

        # 4. 转换响应格式（OpenAI -> Anthropic）
        message = response.choices[0].message
        content_blocks = []

        # 提取文本内容
        if message.content:
            content_blocks.append(ContentBlock("text", text=message.content))

        # 提取工具调用
        if message.tool_calls:
            for tool_call in message.tool_calls:
                content_blocks.append(ContentBlock(
                    "tool_use",
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=json.loads(tool_call.function.arguments)
                ))

        # 映射停止原因：OpenAI "stop"/"tool_calls" -> Anthropic "end_turn"/"tool_use"
        # OpenAI 可能的状态：stop, length, content_filter, tool_calls
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif finish_reason == "stop":
            stop_reason = "end_turn"
        else:
            stop_reason = finish_reason # 备选方案

        return ResponseWrapper(content_blocks, stop_reason)

# =============================================================================
# 工厂函数
# =============================================================================

def get_provider():
    """从环境变量获取当前的 AI 提供商。"""
    return os.getenv("AI_PROVIDER", "anthropic").lower()

def get_client():
    """
    返回一个符合 Anthropic 接口标准的客户端。
    
    如果 AI_PROVIDER 是 'anthropic'，返回原生的 Anthropic 客户端。
    否则，返回一个封装了兼容 OpenAI 协议客户端的 OpenAIAdapter。
    """
    provider = get_provider()

    if provider == "anthropic":
        from anthropic import Anthropic
        base_url = os.getenv("ANTHROPIC_BASE_URL")
        # 返回原生客户端 - 保证 100% 的行为兼容性
        return Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url=base_url
        )
    
    else:
        # 对于 OpenAI/Gemini，我们封装客户端以模仿 Anthropic
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai 库：pip install openai")

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        elif provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            # Gemini 的 OpenAI 兼容端点
            base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        else:
            # 通用的兼容 OpenAI 协议的提供商
            api_key = os.getenv(f"{provider.upper()}_API_KEY")
            base_url = os.getenv(f"{provider.upper()}_BASE_URL")

        if not api_key:
            raise ValueError(f"缺少 {provider} 的 API Key。请检查您的 .env 文件。")

        raw_client = OpenAI(api_key=api_key, base_url=base_url)
        return OpenAIAdapter(raw_client)

def get_model():
    """从环境变量获取模型名称。"""
    model = os.getenv("MODEL_NAME")
    if not model:
        raise ValueError("缺少 MODEL_NAME 环境变量。请在您的 .env 文件中设置它。")
    return model