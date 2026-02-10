"""任务编排：用户指令 → AI 规划 → 执行（SSH）→ 反馈 AI → 循环直到完成。"""
from __future__ import annotations

import json
import logging
import re
import threading
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from .config import AppConfig, load_config
from .llm import (
    EXECUTE_COMMAND_SCHEMA,
    LIST_ASSETS_SCHEMA,
    PROMPT_TOOLS_INSTRUCTION,
    SELF_CODED_SYSTEM,
    SYSTEM_PROMPT,
    create_client,
    chat_once,
    chat_with_prompt_tools,
    chat_with_self_coded_fc,
    chat_with_tools,
)
from .ssh_executor import execute_on_asset, get_asset_by_name, list_assets_display

logger = logging.getLogger("ai_ops_assistant.orchestrator")


def _call_mcp_tool(url: str, name: str, arguments: dict, timeout: int = 120) -> str:
    """通过 MCP Tool HTTP 调用工具（POST /tool），返回 result 文本。"""
    try:
        data = json.dumps({"name": name, "arguments": arguments}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read().decode("utf-8"))
            return (out.get("result") or "").strip() or "(无输出)"
    except Exception as e:
        return f"MCP 工具调用失败: {type(e).__name__}: {e}"


def _extract_json(text: str) -> dict:
    """从模型输出中尽量提取一个 JSON 对象。"""
    text = (text or "").strip()
    if not text:
        raise ValueError("空响应")
    try:
        return json.loads(text)
    except Exception:
        pass
    # 容错：截取第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("无法解析 JSON")


_DANGEROUS_PATTERNS = [
    r"\brm\b.*\s-rf\b",
    r"\bmkfs(\.|)\b",
    r"\bdd\b",
    r":\s*\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\binit\s+0\b",
    r">\s*/dev/sd[a-z]\b",
    r"\b:\s*>\s*/\b",
]


def _looks_dangerous_command(cmd: str) -> bool:
    s = (cmd or "").strip().lower()
    if not s:
        return True
    for pat in _DANGEROUS_PATTERNS:
        if re.search(pat, s):
            return True
    return False


def _resolve_target_assets(config: AppConfig, instruction: str) -> tuple[Optional[list[str]], bool]:
    """
    返回 (资产列表 or None, 是否明确指定)：
    - None 表示未明确指定，需要询问用户
    - list[str] 为需要执行的资产（单台或多台）
    """
    instruction = (instruction or "").strip()
    asset_names = [a.name for a in config.assets]
    if not asset_names:
        return [], True

    if any(k in instruction for k in ("全部", "所有", "每台")):
        return asset_names, True

    mentioned = [name for name in asset_names if name in instruction]
    if mentioned:
        # 只取第一台，符合“一次只在一个资产上执行一条命令”的安全意图；
        # 若用户明确写了多个资产名，后续可再扩展为多台。
        return [mentioned[0]], True
    return None, False


# 无工具模式多轮对话最大轮数，防止死循环
_NO_TOOLS_MAX_ROUNDS = 30


def _run_no_tools_mode(
    client,
    model: str,
    config: AppConfig,
    user_instruction: str,
    on_command: Optional[Callable[[str, str, str], None]] = None,
    on_command_start: Optional[Callable[[str, str], None]] = None,
    on_model_reply: Optional[Callable[[int, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> str:
    """
    降级模式：当模型不支持 tools/function calling 时，采用多轮对话：
    每轮模型根据「用户指令 + 已执行结果」输出下一步（一条命令或结论），
    执行后把结果反馈给模型，循环直到模型给出结论或达到最大轮数。
    """
    logger.info("无工具模式开始（多轮） instruction=%r", user_instruction)
    target_assets, explicit = _resolve_target_assets(config, user_instruction)
    if not explicit:
        logger.info("无工具模式需用户明确指定资产")
        return (
            "你还没有指定要操作的资产。\n\n"
            "当前可用资产：\n"
            f"{list_assets_display(config)}\n\n"
            "请回复要操作的资产名称（如 web-server-01），或回复“全部”。"
        )
    if target_assets is None:
        target_assets = []

    if not target_assets:
        logger.warning("无工具模式: 未配置任何资产")
        return "错误：未配置任何资产，请先在 config.yaml 的 assets 中添加服务器。"

    asset_name = target_assets[0]
    asset = get_asset_by_name(config, asset_name)
    if not asset:
        logger.warning("无工具模式: 未找到资产 %s", asset_name)
        return f"未找到资产 '{asset_name}'。可用资产：\n{list_assets_display(config)}"

    is_deploy_intent = any(k in (user_instruction or "") for k in ("部署", "安装", "upgrade", "install", "nginx"))

    system_content = (
        "你是一个专业的 Linux 运维助手。你只能输出 JSON，不要输出 markdown 或其它文字。\n"
        "根据用户指令和已执行命令的结果，每次只做以下两种之一：\n"
        "1) 需要继续执行：输出 {\"commands\":[{\"command\":\"<一条 shell 命令>\",\"purpose\":\"<目的>\"}]}，每次只给一条命令。\n"
        "2) 可以结束：输出 {\"done\":true,\"conclusion\":\"<用中文给出最终结论与建议>\"}。\n"
        "禁止危险命令：rm -rf、mkfs、dd、shutdown、覆盖磁盘等。命令必须非交互式。\n"
        + (
            "本次是【部署/安装】意图：允许 apt/yum/dnf、systemctl 等，并根据结果继续排查直到成功或明确失败。\n"
            if is_deploy_intent
            else "本次是【巡检/只读】意图：尽量只用只读命令（df、free、ps、journalctl 等）。\n"
        )
    )

    first_user = (
        f"目标资产：{asset.get_display()}\n"
        f"用户指令：{user_instruction}\n\n"
        "请给出第一步要执行的命令（仅一条）。只输出 JSON："
        "{\"commands\":[{\"command\":\"...\",\"purpose\":\"...\"}]}。若无需执行可直接给结论：{\"done\":true,\"conclusion\":\"...\"}。"
    )

    messages: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": first_user},
    ]

    for round_index in range(_NO_TOOLS_MAX_ROUNDS):
        if cancel_event and cancel_event.is_set():
            return "已按用户请求停止。"
        logger.info("无工具模式: 第 %s 轮请求模型", round_index + 1)
        plan_text = chat_once(client, model, messages, max_tokens=1024)
        if not (plan_text or "").strip():
            logger.warning("无工具模式: 第 %s 轮模型返回空，重试一次", round_index + 1)
            plan_text = chat_once(client, model, messages, max_tokens=1024)
        logger.info("无工具模式: 第 %s 轮回复长度=%s", round_index + 1, len(plan_text or ""))
        if plan_text:
            logger.info("无工具模式: 第 %s 轮模型输出: %s", round_index + 1, (plan_text.strip()[:500] + ("..." if len(plan_text) > 500 else "")))
        if on_model_reply and plan_text:
            on_model_reply(round_index + 1, plan_text)

        if not (plan_text or "").strip():
            logger.warning("无工具模式: 第 %s 轮模型仍无有效输出，结束", round_index + 1)
            return "模型本轮未返回有效内容（可能服务偶发空响应）。请重试本次指令，或检查模型服务是否正常。"

        try:
            plan = _extract_json(plan_text)
        except Exception as e:
            logger.warning("无工具模式: 第 %s 轮 JSON 解析失败: %s", round_index + 1, e)
            return f"模型返回的 JSON 无法解析。\n\n原始输出：\n{(plan_text or '').strip() or '(空)'}\n\n解析错误：{e}"

        # 模型选择结束并给出结论
        if plan.get("done") is True and (plan.get("conclusion") or "").strip():
            return (plan.get("conclusion") or "").strip()

        # 模型只给了 conclusion 没有 done，也视为结束
        if (plan.get("conclusion") or "").strip() and not (plan.get("commands")):
            return (plan.get("conclusion") or "").strip()

        commands = plan.get("commands") or []
        if not isinstance(commands, list):
            commands = []
        # 每轮只执行第一条命令，便于模型根据结果决定下一步
        item = commands[0] if commands else None
        if not item or not isinstance(item, dict):
            messages.append(
                {"role": "user", "content": "你没有输出有效命令。请输出 {\"commands\":[{\"command\":\"...\",\"purpose\":\"...\"}]} 或 {\"done\":true,\"conclusion\":\"...\"}。只输出 JSON。"}
            )
            continue

        cmd = (item.get("command") or "").strip()
        if not cmd:
            messages.append({"role": "user", "content": "command 为空，请重新输出一条有效命令的 JSON。"})
            continue

        if _looks_dangerous_command(cmd):
            logger.warning("无工具模式: 拒绝危险命令 cmd=%r", cmd)
            return (
                "为保证安全，已拒绝执行疑似危险命令。\n\n"
                f"命令：{cmd}\n\n"
                "请改写指令或避免危险操作（如 rm -rf、mkfs、dd 等）。"
            )

        if on_command_start:
            on_command_start(asset_name, cmd)
        logger.info("无工具模式: 执行 asset=%s cmd=%r", asset_name, cmd)
        result = execute_on_asset(asset, cmd)
        if on_command:
            on_command(asset_name, cmd, result)
        logger.info("无工具模式: 执行完成 result_len=%s", len(result or ""))

        follow_up = (
            f"已执行命令：{cmd}\n\n"
            f"执行结果：\n{result or '(无输出)'}\n\n"
            "请根据上述结果决定下一步。若问题已解决或可给出最终结论，请输出 {\"done\":true,\"conclusion\":\"...\"}；"
            "否则请输出 {\"commands\":[{\"command\":\"...\",\"purpose\":\"...\"}]}（仅一条命令）。只输出 JSON。"
        )
        messages.append({"role": "user", "content": follow_up})

    # 达到最大轮数仍未结束
    return (
        "已达到多轮对话上限，任务未在限定轮数内完成。请缩小指令范围或分步执行。"
    )


def run_instruction(
    user_instruction: str,
    config_path: Optional[Path] = None,
    on_command: Optional[Callable[[str, str, str], None]] = None,
    on_command_start: Optional[Callable[[str, str], None]] = None,
    on_model_reply: Optional[Callable[[int, str], None]] = None,
    asset_names: Optional[list[str]] = None,
    trace_id: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
    interaction_log_path: Optional[Path] = None,
) -> str:
    """
    执行用户的一条自然语言指令。
    - on_command_start: 可选，在每条命令**开始执行前**调用 (asset_name, command)，用于实时展示「执行中」
    - on_command: 可选，在每条命令**执行完成后**调用 (asset_name, command, result)
    - on_model_reply: 可选，无工具模式每轮模型输出后调用 (round_index, content)，用于日志/前端展示「模型思考」
    - asset_names: 可选，指定本次仅对这些资产执行，AI 不会使用其他资产
    - interaction_log_path: 可选，若指定则把每轮 assistant/user 交互追加到该文件，便于排查中间过程
    返回 AI 的最终回复文本。
    """
    if trace_id:
        logger.info("[%s] run_instruction 开始 instruction=%r asset_names=%s", trace_id, user_instruction, asset_names)
    else:
        logger.info("run_instruction 开始 instruction=%r asset_names=%s", user_instruction, asset_names)
    config = load_config(config_path)
    # 兼容本地 OpenAI 接口（如 Ollama）：即使不填 api_key 也可运行
    # 若 base_url 仍是 DeepSeek 官方且未配置 key，则给出明确提示
    if (not (config.deepseek.api_key or "").strip()) and (config.deepseek.base_url or "").strip() in (
        "",
        "https://api.deepseek.com",
    ):
        return "错误：未配置 DeepSeek API Key。若使用 DeepSeek 官方接口，请在 config.yaml 的 deepseek.api_key 中填写；若使用本地 Ollama，请将 deepseek.base_url 设为如 http://192.168.1.205:11434/v1（可不填 api_key）。"

    client = create_client(config.deepseek)
    tools = [
        {"type": "function", "function": LIST_ASSETS_SCHEMA["function"]},
        {"type": "function", "function": EXECUTE_COMMAND_SCHEMA["function"]},
    ]

    def list_assets() -> str:
        logger.info("%s调用 list_assets()", (f"[{trace_id}] " if trace_id else ""))
        return list_assets_display(config)

    def execute_command(asset_name: str, command: str) -> str:
        if cancel_event and cancel_event.is_set():
            return "已按用户请求停止。"
        command = (command or "").replace("\\n", "\n").replace("\\t", "\t")
        logger.info("%s调用 execute_command asset=%s cmd=%r", (f"[{trace_id}] " if trace_id else ""), asset_name, command)
        if on_command_start:
            on_command_start(asset_name, command)
        asset = get_asset_by_name(config, asset_name)
        if not asset:
            return f"未找到资产 '{asset_name}'。可用资产：\n{list_assets_display(config)}"
        result = execute_on_asset(asset, command)
        if on_command:
            on_command(asset_name, command, result)
        logger.info("%sexecute_command 完成 asset=%s cmd=%r result_len=%s", (f"[{trace_id}] " if trace_id else ""), asset_name, command, len(result or ""))
        return result

    tool_handlers = {
        "list_assets": list_assets,
        "execute_command": execute_command,
    }

    effective_instruction = user_instruction
    if asset_names:
        prefix = (
            "【本次指定运维对象：仅限以下资产：" + "、".join(asset_names) + "。"
            "用户已在界面选定资产，请直接对上述资产调用 execute_command 执行用户指令，不要回复「请指定要操作的资产名称」，也不要仅用文字描述将要执行的操作而不调用 execute_command。】\n\n"
        )
        effective_instruction = prefix + user_instruction
    else:
        # 用户未选资产且指令含「所有/全部」时，要求对 list_assets 返回的每台资产都执行，不要遗漏
        if any(
            k in (user_instruction or "")
            for k in ("所有服务器", "全部服务器", "检查所有", "所有主机", "每台", "全部主机", "所有资产")
        ):
            prefix = (
                "【用户要求检查/操作「所有」或「全部」服务器。请先 list_assets，然后对返回的**每一台**资产依次执行相应命令，不要遗漏任何一台，再输出 final 总结。】\n\n"
            )
            effective_instruction = prefix + user_instruction

    system_content = SYSTEM_PROMPT
    if getattr(config.deepseek, "use_prompt_tools", False):
        system_content = SYSTEM_PROMPT + PROMPT_TOOLS_INSTRUCTION
    if getattr(config.deepseek, "use_self_coded_fc", False):
        system_content = SELF_CODED_SYSTEM
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": effective_instruction},
    ]

    def _write_interaction(path: Path, line: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            logger.warning("写入交互日志失败 path=%s err=%s", path, e)

    on_turn = None
    if interaction_log_path:
        try:
            interaction_log_path.parent.mkdir(parents=True, exist_ok=True)
            with interaction_log_path.open("w", encoding="utf-8") as f:
                f.write(f"=== trace_id: {trace_id or '-'} ===\n")
                f.write(f"=== 用户指令 ===\n{effective_instruction}\n\n")
                f.write("--- system (前 500 字) ---\n")
                f.write((system_content or "")[:500] + "\n\n")
                f.write("--- user (首条) ---\n")
                f.write(effective_instruction + "\n\n")
        except Exception as e:
            logger.warning("写入交互日志头失败 path=%s err=%s", interaction_log_path, e)
        else:
            def on_turn(role: str, content: str) -> None:
                body = (content or "").strip()
                if role == "assistant" and not body:
                    body = "(模型返回为空)"
                _write_interaction(interaction_log_path, f"--- {role} ---\n{body or ''}\n\n")

    if getattr(config.deepseek, "use_self_coded_fc", False):
        use_mcp = getattr(config.deepseek, "use_mcp_for_tools", False)
        mcp_url = (getattr(config.deepseek, "mcp_tool_url", None) or "").strip()
        if use_mcp and mcp_url:
            logger.info("%s使用自研 Agent + MCP 工具执行（%s）", (f"[{trace_id}] " if trace_id else ""), mcp_url)
        else:
            logger.info("%s使用自研 Agent（纯 JSON action 调度）", (f"[{trace_id}] " if trace_id else ""))

        # 注入给模型的 tool_result 最大长度，避免 4w+ 字输出淹没上下文导致模型不输出 action（DeepSeek R1 等）
        _TOOL_RESULT_MAX_CHARS = 12_000

        def _wrap_tool_result(text: str, asset: str | None = None) -> str:
            """用 <tool_result> 包裹；可选 asset 便于多机汇总时按资产区分，避免模型混淆。"""
            s = text or "(无输出)"
            if len(s) > _TOOL_RESULT_MAX_CHARS:
                s = (
                    s[:_TOOL_RESULT_MAX_CHARS]
                    + f"\n\n(以上为命令输出前 {_TOOL_RESULT_MAX_CHARS} 字，已截断；完整共 {len(text)} 字。请根据上述信息输出下一个 action：继续执行命令或 final 总结。)"
                )
            if asset:
                return f"<tool_result asset=\"{asset}\">\n{s}\n</tool_result>"
            return f"<tool_result>\n{s}\n</tool_result>"

        def dispatch(action: dict) -> tuple[str, bool]:
            act = (action.get("action") or "").strip()
            if act == "list_assets":
                if use_mcp and mcp_url:
                    result = _call_mcp_tool(mcp_url, "list_assets", {}, timeout=30)
                    return _wrap_tool_result(result), False
                return _wrap_tool_result(list_assets_display(config)), False
            if act == "execute_command":
                if cancel_event and cancel_event.is_set():
                    return "已按用户请求停止。", True
                asset_name = (action.get("asset") or "").strip()
                command = (action.get("command") or "").strip()
                # 模型在 JSON 中用 \n 表示换行，解析后为字面量 \\n，需转为真实换行再执行
                command = command.replace("\\n", "\n").replace("\\t", "\t")
                if not asset_name or not command:
                    return _wrap_tool_result("action execute_command 缺少 asset 或 command。"), False
                # 拦截占位符，避免首轮误发「资产名称」「shell 命令」导致未找到资产
                if asset_name == "资产名称" or command.strip() == "shell 命令":
                    return _wrap_tool_result(
                        "请使用用户已选定的资产名称（如 linux_222）和具体命令（如 getent passwd），不要使用占位符「资产名称」「shell 命令」。"
                    ), False
                if use_mcp and mcp_url:
                    if on_command_start:
                        on_command_start(asset_name, command)
                    result = _call_mcp_tool(
                        mcp_url, "execute_command",
                        {"asset_name": asset_name, "command": command},
                        timeout=120,
                    )
                    if on_command:
                        on_command(asset_name, command, result)
                    return _wrap_tool_result(result or "(无输出)", asset=asset_name), False
                asset = get_asset_by_name(config, asset_name)
                if not asset:
                    return _wrap_tool_result(f"未找到资产 '{asset_name}'。可用资产：\n{list_assets_display(config)}"), False
                if on_command_start:
                    on_command_start(asset_name, command)
                result = execute_on_asset(asset, command)
                if on_command:
                    on_command(asset_name, command, result)
                return _wrap_tool_result(result or "(无输出)", asset=asset_name), False
            if act == "final":
                return (action.get("message") or "").strip(), True
            return _wrap_tool_result(f"未知 action: {act}，请输出 execute_command / list_assets / final 之一。"), False

        final_reply = chat_with_self_coded_fc(
            client,
            config.deepseek.model,
            messages,
            dispatch,
            trace_id=trace_id,
            on_turn=on_turn,
        )
        logger.info("%srun_instruction 完成 reply_len=%s", (f"[{trace_id}] " if trace_id else ""), len(final_reply or ""))
        if interaction_log_path:
            _write_interaction(interaction_log_path, f"--- final_reply ---\n{final_reply or ''}\n")
        return final_reply

    if getattr(config.deepseek, "use_prompt_tools", False):
        logger.info("%s使用提示词函数调用（不依赖 API tool_calls）", (f"[{trace_id}] " if trace_id else ""))
        default_asset = (asset_names[0] if asset_names and len(asset_names) == 1 else None)
        final_reply, _ = chat_with_prompt_tools(
            client,
            config.deepseek.model,
            messages,
            tool_handlers,
            trace_id=trace_id,
            default_asset_name=default_asset,
            on_turn=on_turn,
        )
        logger.info("%srun_instruction 完成 reply_len=%s", (f"[{trace_id}] " if trace_id else ""), len(final_reply or ""))
        if interaction_log_path:
            _write_interaction(interaction_log_path, f"--- final_reply ---\n{final_reply or ''}\n")
        return final_reply

    try:
        final_reply, _ = chat_with_tools(
            client,
            config.deepseek.model,
            messages,
            tools,
            tool_handlers,
            trace_id=trace_id,
            on_turn=on_turn,
        )
        logger.info("%srun_instruction 完成 reply_len=%s", (f"[{trace_id}] " if trace_id else ""), len(final_reply or ""))
        if interaction_log_path:
            _write_interaction(interaction_log_path, f"--- final_reply ---\n{final_reply or ''}\n")
        return final_reply
    except Exception as e:
        err_str = str(e)
        # Ollama 部分模型会报 “does not support tools”；vLLM 会返回 400 “Extra inputs are not permitted” (tools/tool_choice)
        tools_unsupported = (
            "does not support tools" in err_str
            or ("Extra inputs are not permitted" in err_str and "tool" in err_str.lower())
        )
        if tools_unsupported:
            logger.warning("%s接口不支持 tools/tool_choice，降级为无工具模式", (f"[{trace_id}] " if trace_id else ""))
            final_reply = _run_no_tools_mode(
                client,
                config.deepseek.model,
                config,
                effective_instruction,
                on_command=on_command,
                on_command_start=on_command_start,
                on_model_reply=on_model_reply,
                cancel_event=cancel_event,
            )
            if interaction_log_path:
                _write_interaction(interaction_log_path, "--- final_reply (无工具模式) ---\n" + (final_reply or "") + "\n")
            return final_reply
        raise
