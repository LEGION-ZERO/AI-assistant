"""任务编排：用户指令 → AI 规划 → 执行（SSH）→ 反馈 AI → 循环直到完成。"""
from pathlib import Path
from typing import Callable, Optional

from .config import AppConfig, load_config
from .llm import (
    EXECUTE_COMMAND_SCHEMA,
    LIST_ASSETS_SCHEMA,
    SYSTEM_PROMPT,
    create_client,
    chat_with_tools,
)
from .ssh_executor import execute_on_asset, get_asset_by_name, list_assets_display


def run_instruction(
    user_instruction: str,
    config_path: Optional[Path] = None,
    on_command: Optional[Callable[[str, str, str], None]] = None,
    on_command_start: Optional[Callable[[str, str], None]] = None,
    asset_names: Optional[list[str]] = None,
) -> str:
    """
    执行用户的一条自然语言指令。
    - on_command_start: 可选，在每条命令**开始执行前**调用 (asset_name, command)，用于实时展示「执行中」
    - on_command: 可选，在每条命令**执行完成后**调用 (asset_name, command, result)
    - asset_names: 可选，指定本次仅对这些资产执行，AI 不会使用其他资产
    返回 AI 的最终回复文本。
    """
    config = load_config(config_path)
    if not config.deepseek.api_key:
        return "错误：未配置 DeepSeek API Key，请在 config.yaml 的 deepseek.api_key 中填写。"

    client = create_client(config.deepseek)
    tools = [
        {"type": "function", "function": LIST_ASSETS_SCHEMA["function"]},
        {"type": "function", "function": EXECUTE_COMMAND_SCHEMA["function"]},
    ]

    def list_assets() -> str:
        return list_assets_display(config)

    def execute_command(asset_name: str, command: str) -> str:
        if on_command_start:
            on_command_start(asset_name, command)
        asset = get_asset_by_name(config, asset_name)
        if not asset:
            return f"未找到资产 '{asset_name}'。可用资产：\n{list_assets_display(config)}"
        result = execute_on_asset(asset, command)
        if on_command:
            on_command(asset_name, command, result)
        return result

    tool_handlers = {
        "list_assets": list_assets,
        "execute_command": execute_command,
    }

    effective_instruction = user_instruction
    if asset_names:
        prefix = "【本次指定运维对象：仅限以下资产：" + "、".join(asset_names) + "。请只对这些资产执行命令，不要使用其他资产。】\n\n"
        effective_instruction = prefix + user_instruction

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": effective_instruction},
    ]

    final_reply, _ = chat_with_tools(
        client,
        config.deepseek.model,
        messages,
        tools,
        tool_handlers,
    )
    return final_reply
