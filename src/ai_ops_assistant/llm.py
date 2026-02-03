"""DeepSeek API 接入：对话与工具调用。"""
from openai import OpenAI

from .config import DeepSeekConfig


def create_client(config: DeepSeekConfig) -> OpenAI:
    return OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
    )


# 供 AI 调用的“在指定资产上执行命令”工具
EXECUTE_COMMAND_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_command",
        "description": "在指定的 Linux 资产（服务器）上执行 shell 命令。用于巡检、部署、查看日志等。",
        "parameters": {
            "type": "object",
            "properties": {
                "asset_name": {
                    "type": "string",
                    "description": "资产名称，必须在已知资产列表中，例如 web-server-01",
                },
                "command": {
                    "type": "string",
                    "description": "要在目标服务器上执行的完整 shell 命令，例如: df -h",
                },
            },
            "required": ["asset_name", "command"],
        },
    },
}

# 列出可用资产
LIST_ASSETS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_assets",
        "description": "列出当前配置的所有 Linux 资产（服务器）名称与连接信息，用于确认要对哪台机器执行操作。",
        "parameters": {"type": "object", "properties": {}},
    },
}


SYSTEM_PROMPT = """你是一个专业的 Linux 运维助手。用户会给你自然语言指令，你需要：

1. 理解用户意图（例如：巡检某台服务器、部署应用、查看磁盘/内存/进程等）。
2. 通过 list_assets 了解当前有哪些 Linux 资产。
3. 通过 execute_command 在对应资产上执行具体命令，根据输出继续执行或给出结论。

【重要】资产范围规则：
- 若用户**未明确指定**要对哪台/哪些资产操作（例如只说「查看资产的内存」「做一次巡检」等），你必须先 list_assets，然后**只回复询问用户**，不要对任何资产执行命令。询问示例：「当前有资产：linux_222、linux_111。请指定要操作的资产名称（如 linux_222），或说明「全部」以对所有资产执行。」
- 仅当用户**明确指定**单台资产（如「对 linux_222 做巡检」）或**明确表示全部**（如「全部」「所有资产」「所有服务器」「每台都查」）时，才可对相应资产执行 execute_command。
- 一次只在一个资产上执行一条命令，根据结果再决定下一步。

其他规则：
- 只执行用户明确要求或合理推断出的操作，不要执行危险命令（如 rm -rf /、格式化磁盘等）。
- 巡检时可按：系统信息、CPU/内存/磁盘、关键进程、日志错误等顺序执行。
- 回复简洁专业，必要时用中文总结执行结果。"""


def chat_with_tools(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict],
    tool_handlers: dict,
) -> tuple[str, list[dict]]:
    """带 function calling 的对话：若 AI 返回 tool_calls 则执行并继续，否则返回最终回复。"""
    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            return (msg.content or "").strip(), messages

        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            if name not in tool_handlers:
                result = f"未知工具: {name}"
            else:
                import json
                try:
                    args = json.loads(tc.function.arguments)
                except Exception as e:
                    result = f"参数解析失败: {e}"
                else:
                    result = tool_handlers[name](**args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                }
            )
    return "", messages
