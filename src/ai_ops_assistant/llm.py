"""DeepSeek API 接入：对话与工具调用。"""
from __future__ import annotations

import json
import logging
import re
from typing import Callable
from urllib.parse import urlparse

from openai import OpenAI

from .config import DeepSeekConfig

logger = logging.getLogger("ai_ops_assistant.llm")

# 模型多次「说要执行但未调工具」时，最多提醒几轮（避免无限提醒）
MAX_NUDGES = 2

# 部分本地模型（如 Qwen2.5 7B via Ollama）不在 API 层返回 tool_calls，而是在正文里写 <tool_call>...</tool_call>
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{[^<]*\})\s*</tool_call>", re.DOTALL | re.IGNORECASE)
# 兜底：仅出现 </tool_call> 或整段里只有一处 {"name":..., "arguments":...}（arguments 可为 {} 或单层对象）
_JSON_TOOL_RE = re.compile(
    r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{(?:[^{}]|\{[^{}]*\})*\})\s*\}', re.DOTALL
)
# 一些模型会在 JSON 里加行内注释（// xxx），需先去掉再做 json.loads；避免误伤 http:// 用负向前瞻
_JSON_LINE_COMMENT_RE = re.compile(r"(?<!:)//.*$", re.MULTILINE)
# DeepSeek R1 等推理模型常在正文前输出 <think>...</think>
_THINK_BLOCK_RE = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)


def _strip_think_blocks(text: str) -> str:
    """去掉 <think>...</think>
 块，便于从推理模型回复中解析 action/tool_call。"""
    if not (text or "").strip():
        return text
    return _THINK_BLOCK_RE.sub("", text).strip()


def _strip_json_comments(text: str) -> str:
    """移除 JSON 字符串中的行内 // 注释，便于宽容解析模型返回。"""
    if not (text or "").strip():
        return text
    return _JSON_LINE_COMMENT_RE.sub("", text)


def _parse_tool_calls_from_content(content: str) -> list[dict]:
    """从助手回复正文中解析 Qwen 等模型的 <tool_call>{"name":"...","arguments":...}</tool_call>。"""
    if not (content or "").strip():
        return []
    out = []
    # 1) 标准 <tool_call>...</tool_call> 块
    for m in _TOOL_CALL_BLOCK_RE.finditer(content):
        try:
            obj = json.loads(_strip_json_comments(m.group(1).strip()))
            name = obj.get("name")
            args = obj.get("arguments")
            if isinstance(args, str):
                args = json.loads(args) if args.strip() else {}
            if not isinstance(args, dict):
                args = {}
            if name:
                out.append({"name": name, "arguments": args})
        except Exception:
            continue
    if out:
        return out
    # 2) 兜底：正文中任意位置的 {"name":"...","arguments":...}（如只有 </tool_call> 或裸 JSON）
    for m in _JSON_TOOL_RE.finditer(content):
        try:
            name = m.group(1).strip()
            args_str = _strip_json_comments((m.group(2) or "").strip())
            args = json.loads(args_str) if args_str else {}
            if not isinstance(args, dict):
                args = {}
            if name:
                out.append({"name": name, "arguments": args})
        except Exception:
            continue
    return out


# 模型有时输出 ```json {"command":"...", "purpose":..., "output_file":...} ``` 而非 <tool_call>，用此兜底
_JSON_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def _parse_command_only_json(content: str, default_asset_name: str) -> list[dict]:
    """
    从正文中解析「仅含 command 的 JSON」块（如 ```json {"command":"...", "purpose":...} ```），
    转为 execute_command 调用。当 default_asset_name 有值时由调用方传入（单资产场景）。
    """
    if not (content or "").strip() or not (default_asset_name or "").strip():
        return []
    out: list[dict] = []
    for m in _JSON_CODE_BLOCK_RE.finditer(content):
        try:
            raw = (m.group(1) or "").strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                continue
            cmd = obj.get("command")
            if isinstance(cmd, str) and cmd.strip():
                out.append({
                    "name": "execute_command",
                    "arguments": {"asset_name": default_asset_name.strip(), "command": cmd.strip()},
                })
        except Exception:
            continue
    # 若没有 ```json 代码块，再尝试将整段内容视为单个 JSON（模型可能直接输出 {"command":"..."}）
    if not out:
        try:
            raw_all = content.strip()
            obj_all = json.loads(raw_all)
            if isinstance(obj_all, dict):
                cmd_all = obj_all.get("command")
                if isinstance(cmd_all, str) and cmd_all.strip():
                    out.append({
                        "name": "execute_command",
                        "arguments": {"asset_name": default_asset_name.strip(), "command": cmd_all.strip()},
                    })
        except Exception:
            pass
    return out


def _normalize_base_url(base_url: str) -> str:
    """
    OpenAI SDK 的 base_url 通常需要包含 /v1。
    - DeepSeek 官方: https://api.deepseek.com
    - Ollama OpenAI 兼容: http://host:11434/v1
    """
    base_url = (base_url or "").strip()
    if not base_url:
        return "https://api.deepseek.com"

    parsed = urlparse(base_url)
    if not parsed.scheme:
        # 允许用户只填 host:port
        base_url = "http://" + base_url
        parsed = urlparse(base_url)

    path = (parsed.path or "").rstrip("/")
    if path in ("", "/"):
        return base_url.rstrip("/") + "/v1"
    if path == "/v1":
        return base_url.rstrip("/")
    # 若用户给了更深路径（例如 /api/v1），尊重用户输入
    return base_url.rstrip("/")


def create_client(config: DeepSeekConfig) -> OpenAI:
    base_url = _normalize_base_url(config.base_url)
    # 本地服务（如 Ollama）常不校验 key，但 SDK 需要一个字符串
    api_key = (config.api_key or "").strip() or "ollama"
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def chat_once(
    client: OpenAI,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """不使用 tools 的普通对话。max_tokens/temperature 可选，自研 Agent 建议限长+低 temperature 以加速且稳定 JSON。"""
    logger.info("chat_once 请求 model=%s messages条数=%s", model, len(messages))
    kwargs: dict = {"model": model, "messages": messages}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.chat.completions.create(**kwargs)
    msg = response.choices[0].message
    content = getattr(msg, "content", None) or (msg.model_dump().get("content") if hasattr(msg, "model_dump") else None) or ""
    content = (content or "").strip()
    # DeepSeek R1 等推理模型常把正文放在 reasoning，content 为空
    if not content:
        reasoning = getattr(msg, "reasoning", None) or (msg.model_dump().get("reasoning") if hasattr(msg, "model_dump") else None)
        if reasoning and (str(reasoning) or "").strip():
            content = (str(reasoning) or "").strip()
            logger.info("chat_once content 为空，使用 reasoning 作为回复 len=%s", len(content))
    if not content:
        try:
            raw = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
        except Exception:
            raw = {"content": getattr(msg, "content", None)}
        logger.warning(
            "chat_once 收到空 content，message 键=%s 前300字=%s",
            list(raw.keys()),
            repr(str(raw.get("content", raw.get("reasoning", "")))[:300]),
        )
    return content


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

1. 理解用户的意图（如：巡检某台服务器、部署应用、查看磁盘/内存/进程等）。
2. 通过 list_assets 获取当前的 Linux 资产信息。
3. 根据用户需求，使用 execute_command 执行相应命令，并根据结果继续执行后续操作或给出结论。
4. 所有回复必须使用**简体中文**，不得使用英文自然语言描述（命令名、日志原文除外）。

【重要】资产范围规则：
- **若用户消息开头带有【本次指定运维对象：仅限以下资产：xxx】**（由前端/接口传入），则视为用户已指定资产，必须**直接**对所列资产执行命令，不得仅通过文字描述「我将执行 xxx」，而要实际执行命令。
- **若用户未明确指定操作资产**，且消息中无【本次指定运维对象】，你需要先通过 list_assets 查询资产列表，然后**询问用户**操作资产。例如：「当前有资产：linux_222、linux_111，请指定操作资产（如 linux_222），或选择「全部」以操作所有资产」。
- **仅在用户明确指定单台资产**（例如「对 linux_222 做巡检」）或**明确要求所有资产**（如「全部」「所有资产」「所有服务器」等），或者消息中包含【本次指定运维对象】时，才可执行对应资产的命令。
- **一次只执行一条命令**，根据执行结果再决定是否继续执行下一步操作。 

【重要】多轮执行与失败跟进：
- 当命令执行失败或输出错误（如服务启动失败、配置错误等），你必须继续调用 execute_command 执行诊断命令（如 systemctl status、journalctl -xeu 等），并根据输出继续排查，不可仅用文字回复「请执行 xxx 查看日志」。
- 若你通过 `<tool_result>` 收到的结果表示未成功或需要进一步排查，你必须**继续执行下一条命令**，而非仅提供文字建议。只有在问题已解决或你收集到足够信息给出最终结论时，才返回文字回复。
- **一次只执行一条命令**，每次执行后根据命令的输出决定下一步，形成连贯的多轮交互。

【重要】根据命令输出判断结果：
- 在宣称操作成功前，必须根据命令的实际输出确认操作状态。例如「未安装 nginx 包」或者「安装成功」等，若安装或卸载失败，不能简单回复「成功」。
- 对于卸载或删除操作，如果输出提示「不再需要」等，需继续执行 `apt autoremove -y` 清理残留依赖后，再返回最终结论。

【重要】命令必须具体、可执行：
- **禁止在命令中使用占位符**，如 `<webapp_user>`、`<username>`、`<host>` 等。必须根据上下文写出真实值：例如路径为 /home/hulin/tmp 时，用户名为 hulin，应使用 `hulin`；若需当前 SSH 用户执行，直接写命令即可（如 `ls /home/hulin/tmp`），不要写 `su - <user>`。
- **若你打算执行某条命令，必须在同一轮中调用 execute_command**，不得只回复「我将执行…」「请执行…」等文字而不调用工具。若你只回复了文字说将要执行却未调用工具，则视为未完成；应在本轮就调用 execute_command，或继续多轮直到给出最终结论。
- **不得只给「建议」而不执行**：若你的结论里会出现「Recommended Actions: Check with `top`」「建议执行 xxx 命令」「Use `journalctl` to」等，你必须先调用 execute_command 执行这些命令，再根据执行结果给出最终结论；不得只回复文字建议而不实际调用工具。
- **不得只认错不执行**：若你发现应调用工具却未调用，不要只回复「错误更正」「正确处理流程应是」或请用户「告知具体目标」——应直接在本轮调用 execute_command 执行后续巡检/排查命令（如 docker logs、journalctl 等），再根据结果给出结论。

其他规则：
- 只执行用户明确要求或合理推断的操作，避免执行危险命令（如 rm -rf / 等破坏性命令）。
- 在巡检时，可以按以下顺序执行：系统信息、CPU/内存/磁盘、关键进程、日志错误等。
- 反馈要简洁、专业，用中文总结执行结果，确保用户能够快速理解。

通过这一流程，确保每一次操作都是安全、准确的。"""

# 使用「提示词 + 解析回复」实现函数调用时，追加到 system 的说明（不依赖 API tool_calls）
PROMPT_TOOLS_INSTRUCTION = """

【工具调用方式】你只能通过以下格式调用工具，不要编造占位符（如 <user>），参数必须为具体值。
在回复中输出：<tool_call>{"name":"工具名","arguments":{...}}</tool_call>
一次只输出一个 tool_call。执行结果会以 <tool_result> 形式在下一条消息中返回；收到结果后你可继续输出下一个 <tool_call> 或给出最终文字结论。
**重要**：当你说「接下来执行 xxx 命令」「让我执行 xxx」「执行的下一步命令：xxx」时，必须在同一条回复里直接输出对应的 <tool_call>，不要只写文字说明再等下一轮；否则程序会认为你未调用工具。
**重要**：不要写「请回复 <tool_result> 后续执行结果」——只有你先在本条回复中输出 <tool_call>，程序执行命令后才会自动在下一条消息里返回 <tool_result>；你不能要求用户或程序「回复 tool_result」，必须先由你输出 <tool_call>。

【可用工具】
1. list_assets
   描述：列出当前配置的所有 Linux 资产（服务器）名称与连接信息。
   参数：arguments 为 {}，无参数。

2. execute_command
   描述：在指定资产上执行 shell 命令。
   参数：asset_name（字符串，资产名称）、command（字符串，要执行的完整命令）。
   示例：<tool_call>{"name":"execute_command","arguments":{"asset_name":"web-server-01","command":"df -h"}}</tool_call>
"""

# 自研 Agent：模型只输出约定格式，程序解析并调度（无任何 API tool_calls）
SELF_CODED_SYSTEM = """你是一个 Linux 运维助手。用户会给出自然语言指令，你需要通过输出「仅一条」约定格式来驱动程序执行，不要输出任何其他文字、markdown 或解释。
必须直接输出一个 JSON 或 <tool_call>，不要先输出 <think> 或思考过程，不要返回空内容。

【输出格式】二选一，任一种均可：

方式 A（JSON）：
1) 执行命令：{"action": "execute_command", "asset": "实际资产名（如 linux_222）", "command": "实际命令（如 getent passwd）"}
2) 查询资产：{"action": "list_assets"}
3) 结束总结：{"action": "final", "message": "中文总结"}
**禁止**在 asset 中写「资产名称」、在 command 中写「shell 命令」等占位符，必须用用户指定的资产名和具体命令。

【final 的 message 要求】必须是针对用户指令和命令输出的 1～2 句专业总结（如：服务是否正常、有无异常、建议操作），禁止只回复「任务已完成」「已执行」等空话。例如用户要求「查看 docker 服务」且已执行 systemctl status docker，应总结 Docker 状态、是否有异常日志等。**当命令输出为列表/表格类数据**（如 getent passwd、用户与权限、docker ps、进程列表等）时，请用 **Markdown 表格** 汇总，**且表格中的每一行必须严格来自 <tool_result> 中的实际输出，禁止添加或臆造未在输出中出现的用户、容器、进程等条目**。示例：
| 用户 | UID | Shell | 说明 |
|------|-----|-------|------|
| root | 0 | /bin/bash | 超级用户 |
 这样前端会渲染成易读的表格。

方式 B（<tool_call>，很多本地模型更易输出）：
1) 执行命令：<tool_call>{"name":"execute_command","arguments":{"asset_name":"实际资产名","command":"实际命令"}}</tool_call>（asset_name/command 必须为具体值，禁止写「资产名称」「shell 命令」）
2) 查询资产：<tool_call>{"name":"list_assets","arguments":{}}</tool_call>
3) 结束总结：仅用 JSON {"action": "final", "message": "中文总结"}（无 tool_call）

【严禁】禁止输出 initial_request、details、simulated_server_interaction、simulated、scenario 等任何非上述三种格式的 JSON。禁止输出多段或嵌套的“模拟对话”类结构。若需要执行操作，必须直接输出 action 为 execute_command 的 JSON；若需要先查资产，必须只输出 {"action": "list_assets"}。

【规则】asset 和 command 必须为具体可执行值；面向用户的 message 必须用简体中文。一次只输出一个 action。action 为 final 时，message 必须根据用户指令和**仅根据 <tool_result> 中的实际内容**给出结论或表格，禁止添加未在命令输出中出现的条目（如臆造的用户名、容器名等）。禁止只回复「任务已完成」「已执行」「将遵守格式」等空话。

【上下文】当「用户」下一条消息是 <tool_result>...</tool_result> 时，表示你上一条命令/查询已执行完毕，你必须根据结果**立即**输出下一个 action（继续 execute_command 或 list_assets，或 final 总结），不要只分析文字不输出 action。

【JSON 与命令】输出时 "command" 内如需换行请用 \\n 表示，勿在 JSON 字符串中写真实换行。在服务器上写多行文件时，请用 heredoc（如 cat << 'EOF' ... EOF）或 base64 写入，不要用多条 echo 拼 \\n，易产生语法错误。回复尽量简短，只输出一个 JSON，不要输出大段解释。

【非交互式执行】命令通过 SSH 非交互执行，**禁止使用会等待 stdin 输入的交互式命令**，否则会超时。例如：修改用户密码时不要用 `passwd 用户名`（会等待输入新密码），应使用 `echo "用户名:新密码" | chpasswd`（需 root）；其他需输入的地方用管道、heredoc 或脚本替代。"""


def _fix_newlines_in_json_strings(s: str) -> str:
    """把 JSON 字符串值内的真实换行替换为 \\n，便于 json.loads 解析（模型常在 command 里写多行）。"""
    out: list[str] = []
    i = 0
    in_string = False
    escape_next = False
    while i < len(s):
        c = s[i]
        if escape_next:
            out.append(c)
            escape_next = False
            i += 1
            continue
        if in_string:
            if c == "\\":
                out.append(c)
                escape_next = True
                i += 1
                continue
            if c == '"':
                in_string = False
                out.append(c)
                i += 1
                continue
            if c in "\n\r":
                out.append("\\n")
                if c == "\r" and i + 1 < len(s) and s[i + 1] == "\n":
                    i += 1
                i += 1
                continue
            out.append(c)
            i += 1
            continue
        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _parse_self_coded_action(content: str) -> dict | None:
    """从模型回复中解析自研 Agent 的 action。支持 <tool_call>、整段 JSON、```json ... ```；会先剥离 <think> 块。"""
    if not (content or "").strip():
        return None
    raw = _strip_think_blocks(content).strip()
    if not raw:
        return None
    # 1) 先尝试 <tool_call> 格式（本地模型常出此格式，比纯 JSON 更稳）
    tool_calls = _parse_tool_calls_from_content(raw)
    if len(tool_calls) == 1:
        tc = tool_calls[0]
        name = (tc.get("name") or "").strip()
        args = tc.get("arguments") or {}
        if name == "list_assets":
            return {"action": "list_assets"}
        if name == "execute_command":
            asset = (args.get("asset_name") or args.get("asset") or "").strip()
            cmd = (args.get("command") or "").strip()
            if asset or cmd:
                return {"action": "execute_command", "asset": asset, "command": cmd}
    # 2) 再尝试整段 JSON（含对换行、尾部逗号的容错）
    _parse_error: list[str] = []  # 记录首次解析失败原因，便于诊断

    def _try_load_action(s: str) -> dict | None:
        for t in (s, _fix_newlines_in_json_strings(s), re.sub(r",\s*([}\]])", r"\1", s)):
            try:
                obj = json.loads(_strip_json_comments(t))
                if not isinstance(obj, dict):
                    continue
                if obj.get("action"):
                    return obj
                # 兼容模型返回 {"response": "总结内容"} 而非 {"action": "final", "message": "..."}
                if obj.get("response") is not None:
                    return {"action": "final", "message": str(obj.get("response", "")).strip()}
            except Exception as e:
                if not _parse_error:
                    _parse_error.append(f"{type(e).__name__}: {e}")
                continue
        return None

    for block in (raw,):
        o = _try_load_action(block)
        if o:
            return o
    # 先尝试「第一个完整 {...}」再按整段试，便于模型在 JSON 后追加说明时仍能解析
    start = raw.find("{")
    if start >= 0 and '"action"' in raw:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    seg = raw[start : i + 1]
                    o = _try_load_action(seg)
                    if o:
                        return o
                    break
    # 再尝试从 ```json ... ``` 中取；若失败则把字符串值内换行转为 \n 再解析
    for m in _JSON_CODE_BLOCK_RE.finditer(raw):
        inner = (m.group(1) or "").strip()
        o = _try_load_action(inner)
        if o:
            return o
    # 最后尝试：去掉首条非 JSON 前缀后再解析（如 "Here is the command: {...}"）
    if "{" in raw and '"action"' in raw:
        o = _try_load_action(raw[raw.find("{"):])
        if o:
            return o
    # 推理模型（如 DeepSeek R1）常把 JSON/<tool_call> 放在回复末尾，尝试从尾部截取再解析
    if len(raw) > 400:
        for tail in (raw[-2000:], raw[-1200:], raw[-800:], raw[-500:]):
            if "{" in tail and '"action"' in tail:
                o = _try_load_action(tail[tail.find("{"):])
                if o:
                    return o
            tool_calls = _parse_tool_calls_from_content(tail)
            if len(tool_calls) == 1:
                tc = tool_calls[0]
                name = (tc.get("name") or "").strip()
                args = tc.get("arguments") or {}
                if name == "list_assets":
                    return {"action": "list_assets"}
                if name == "execute_command":
                    asset = (args.get("asset_name") or args.get("asset") or "").strip()
                    cmd = (args.get("command") or "").strip()
                    if asset or cmd:
                        return {"action": "execute_command", "asset": asset, "command": cmd}
    # 推理模型有时把 final JSON 写在 reasoning 正文中间/末尾，整段搜索 "action":"final" 或 "action": "final" 后取完整 {...}
    for _m in re.finditer(r'\{\s*"action"\s*:\s*"final"', raw):
        _start = _m.start()
        _depth = 0
        for _i in range(_start, len(raw)):
            if raw[_i] == "{":
                _depth += 1
            elif raw[_i] == "}":
                _depth -= 1
                if _depth == 0:
                    _seg = raw[_start : _i + 1]
                    o = _try_load_action(_seg)
                    if o:
                        return o
                    break
    # 模型常把 message 写成多行或截断导致 Unterminated string，尝试从 "message": " 后提取到结尾或下一个未转义的 "
    _msg_prefix = re.search(r'"message"\s*:\s*"', raw)
    if _msg_prefix and ("final" in raw[:_msg_prefix.start()] or '"action"' in raw[:_msg_prefix.start()]):
        _start = _msg_prefix.end()
        _buf = []
        _i = _start
        while _i < len(raw):
            if raw[_i] == "\\" and _i + 1 < len(raw):
                _c = raw[_i + 1]
                if _c == "n":
                    _buf.append("\n")
                elif _c == "t":
                    _buf.append("\t")
                elif _c == '"':
                    _buf.append('"')
                elif _c == "\\":
                    _buf.append("\\")
                else:
                    _buf.append(raw[_i])
                    _buf.append(_c)
                _i += 2
                continue
            if raw[_i] == '"':
                break
            _buf.append(raw[_i])
            _i += 1
        _msg = "".join(_buf).strip()
        if _msg and len(_msg) > 5 and not _is_final_message_fluff(_msg):
            return {"action": "final", "message": _msg}
    # 诊断：解析失败时打印关键信息，便于排查「未解析到有效 action」
    logger.info(
        "_parse_self_coded_action 未匹配 raw_len=%s 首条 json 报错=%s 前400字 repr=%s",
        len(raw),
        (_parse_error[0] if _parse_error else "-"),
        repr((raw[:400] + ("..." if len(raw) > 400 else ""))),
    )
    return None


def _is_final_message_fluff(message: str) -> bool:
    """判断 final 的 message 是否为「遵守格式」类空话而非真实总结。"""
    if not (message or "").strip():
        return True
    s = (message or "").strip()
    # 含 Markdown 表格（多列 | 或分隔行 ---）的视为有效总结，不判为空话
    if "|" in s and ("\n" in s or "---" in s or s.count("|") >= 3) and len(s) > 30:
        return False
    fluff_phrases = (
        "根据用户指示",
        "仅输出三种",
        "严格遵守",
        "以符合规则",
        "符合规则",
        "指定格式",
        "都将严格遵守",
        "任务已完成",
        "任务已完成。",
        "已执行。",
        "已执行完毕",
    )
    if any(p in s for p in fluff_phrases):
        return True
    # 过短且无实质内容（无巡检/结果/状态/用户/权限等关键词）
    if len(s) < 25 and not any(
        k in s
        for k in ("巡检", "结果", "正常", "异常", "状态", "运行", "建议", "docker", "服务", "容器", "用户", "权限")
    ):
        return True
    return False


def _parse_getent_passwd_to_table(tool_result: str) -> str | None:
    """从 getent passwd 命令输出解析并生成 Markdown 表格。若无法解析则返回 None。
    规则严格，避免把 ip addr、link/ether 等输出误判为 passwd 行。"""
    if not tool_result:
        return None
    # 明显不是 passwd 输出：含 ip/网络接口特征则直接不解析
    if "link/ether" in tool_result or "scope global" in tool_result or "inet " in tool_result and "mtu" in tool_result:
        return None
    lines = tool_result.strip().split("\n")
    users = []
    invalid_prefixes = ("link", "inet", "valid", "brd", "scope", "altname", "state ", "group ")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("<"):
            continue
        # getent passwd 格式：username:password:uid:gid:gecos:home:shell（严格 7 段）
        parts = line.split(":")
        if len(parts) != 7:  # passwd 行恰好 7 段，避免匹配含大量冒号的其他输出
            continue
        username, uid, gid = parts[0].strip(), parts[2].strip(), parts[3].strip()
        shell = (parts[6] or "").strip()
        # 用户名：仅字母数字、下划线、减号，且不以 link/inet 等开头；shell 必须是路径（含 /）
        if not username or not uid.isdigit() or not gid.isdigit():
            continue
        if any(username.lower().startswith(p) for p in invalid_prefixes):
            continue
        if "/" not in shell or " " in username:
            continue
        users.append({"username": username, "uid": uid, "gid": gid, "shell": shell})
    if len(users) < 2:
        return None
    table_lines = ["| 用户名 | UID | GID | Shell |", "|--------|-----|-----|-------|"]
    for u in users:
        table_lines.append(f"| {u['username']} | {u['uid']} | {u['gid']} | {u['shell']} |")
    return "\n".join(table_lines)


def _parse_df_h_line(line: str) -> dict | None:
    """从 df -h 单行解析出 size/used/avail/use%/mount。仅当行为 /dev 开头且含 % 时解析。"""
    line = line.strip()
    if not line.startswith("/dev") or "%" not in line:
        return None
    parts = line.split()
    if len(parts) < 5:
        return None
    # 格式：... size used avail use% mount，取最后 5 列
    size, used, avail, use_pct, mount = parts[-5], parts[-4], parts[-3], parts[-2], parts[-1]
    if "%" not in use_pct:
        return None
    return {"size": size, "used": used, "avail": avail, "use_pct": use_pct, "mount": mount}


def _build_df_table_from_messages(messages: list[dict]) -> str | None:
    """从 messages 中提取带 asset 的 <tool_result>，解析 df -h 输出，按资产生成汇总表。"""
    rows: list[tuple[str, dict]] = []  # (asset, row_dict)
    # 带 asset 的块：<tool_result asset="xxx">...</tool_result>
    pattern = re.compile(r'<tool_result\s+asset="([^"]+)"[^>]*>([\s\S]*?)</tool_result>', re.IGNORECASE)
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content") or ""
        for m in pattern.finditer(content):
            asset, body = m.group(1).strip(), m.group(2)
            if "df" not in body.lower() and "Filesystem" not in body and "文件系统" not in body:
                continue
            for line in body.split("\n"):
                row = _parse_df_h_line(line)
                if row:
                    rows.append((asset, row))
                    break
    if len(rows) < 2:
        return None
    table_lines = ["| 资产 | 总空间 | 已用 | 可用 | 使用率 | 挂载点 |", "|------|--------|------|------|--------|--------|"]
    for asset, r in rows:
        table_lines.append(f"| {asset} | {r['size']} | {r['used']} | {r['avail']} | {r['use_pct']} | {r['mount']} |")
    return "\n".join(table_lines)


def _looks_like_final_summary(content: str) -> bool:
    """模型未按 JSON 输出但给了一段自然语言总结时，视为可用的最终回复（如 ## Summarize\\n...）。"""
    if not (content or "").strip() or len(content) < 50 or len(content) > 8000:
        return False
    c = content.strip()
    # 已是约定格式则交解析器处理，不在此处兜底
    if c.startswith("{") or c.startswith("<tool_call") or '"action"' in c[:100]:
        return False
    # 像「我将执行」或提问，不当作总结
    if any(p in c for p in ("我将", "我会执行", "请指定", "请告知", "? ", "？")):
        return False
    # 像总结：含 Summarize/summary/总结，或 服务/容器/状态 + running/active/Up/运行
    if "Summarize" in c or "summary" in c.lower() or "总结" in c:
        return True
    if ("Docker" in c or "docker" in c or "服务" in c or "容器" in c) and (
        "running" in c or "active" in c or "Up" in c or "状态" in c or "运行" in c
    ):
        return True
    if "command" in c.lower() and ("result" in c.lower() or "output" in c.lower() or "执行" in c):
        return True
    return False


def chat_with_self_coded_fc(
    client: OpenAI,
    model: str,
    messages: list[dict],
    dispatch: Callable[[dict], tuple[str, bool]],
    trace_id: str | None = None,
    max_rounds: int = 50,
    on_turn: Callable[[str, str], None] | None = None,
) -> str:
    """
    自研 Agent 循环：模型只输出纯 JSON（action/asset/command 或 action/final/message），
    dispatch(action) 返回 (下一条 user 消息, 是否结束)；若结束则返回 final 的 message。
    on_turn(role, content)：每轮追加 assistant/user 时回调，用于记录交互到文件。
    """
    prefix = f"[{trace_id}] " if trace_id else ""
    nudge_count = 0
    # 自研 Agent：本地模型不限制 token，避免总结/表格被截断；首轮略高 temperature
    for r in range(max_rounds):
        chat_kw = {"max_tokens": 8192, "temperature": 0.5 if r == 0 else 0.3}
        logger.info("%schat_with_self_coded_fc 第 %s 轮请求 model=%s", prefix, r + 1, model)
        content = chat_once(client, model, messages, **chat_kw)
        content = (content or "").strip()
        # 空回复时重试一次：首轮用略高 temperature；第 2 轮起用略高 max_tokens（R1 在 tool_result 后常返空）
        if not content:
            if r == 0:
                logger.info("%schat_with_self_coded_fc 首轮空回复，重试一次（temperature=0.7）", prefix)
                content = chat_once(client, model, messages, max_tokens=8192, temperature=0.7)
            else:
                logger.info("%schat_with_self_coded_fc 第 %s 轮空回复，重试一次（max_tokens=8192）", prefix, r + 1)
                content = chat_once(client, model, messages, max_tokens=8192, temperature=0.3)
            content = (content or "").strip()
        action = _parse_self_coded_action(content)
        if not action:
            # 再试一次：去掉首条非 JSON 前缀（从第一个 { 开始）后解析
            if (content or "").strip() and "{" in content and '"action"' in content:
                idx = content.find("{")
                action = _parse_self_coded_action(content[idx:])
            if not action:
                c = (content or "").strip()
                logger.warning(
                    "%schat_with_self_coded_fc 未解析到有效 action reply_len=%s 前600字=%s",
                    prefix,
                    len(c),
                    repr(c[:600] + ("..." if len(c) > 600 else "")),
                )
                # 模型有时直接给自然语言总结（如 ## Summarize\\n...），视为最终回复
                if r >= 1 and c and _looks_like_final_summary(c):
                    logger.info("%schat_with_self_coded_fc 将自然语言总结当作最终回复", prefix)
                    return c
                if nudge_count < 2:
                    nudge_count += 1
                    logger.info("%schat_with_self_coded_fc 未解析到有效 action，补发格式提醒（第%s次）", prefix, nudge_count)
                    messages.append({"role": "assistant", "content": content})
                    nudge_content = (
                        "你刚才的回复不是规定的 action 格式。请只输出以下三种之一（可用 JSON 或 <tool_call>）：\n"
                        '1) {"action": "list_assets"} 或 <tool_call>{"name":"list_assets","arguments":{}}</tool_call>\n'
                        '2) {"action": "execute_command", "asset": "资产名", "command": "具体命令"} 或 <tool_call>{"name":"execute_command","arguments":{"asset_name":"资产名","command":"具体命令"}}</tool_call>\n'
                        '3) {"action": "final", "message": "本次任务的一两句话中文总结"}（message 必须是真实结论，不要空话）'
                    )
                    messages.append({"role": "user", "content": nudge_content})
                    if on_turn:
                        on_turn("assistant", content)
                        on_turn("user", nudge_content)
                    continue
                logger.warning(
                    "%schat_with_self_coded_fc 未解析到有效 action，当作最终回复 reply_len=%s 前600字=%s",
                    prefix,
                    len(c),
                    repr(c[:600] + ("..." if len(c) > 600 else "")),
                )
                return content or "模型未返回有效 JSON 动作。"
        act = action.get("action", "")
        messages.append({"role": "assistant", "content": content})
        user_msg, is_final = dispatch(action)
        if on_turn:
            on_turn("assistant", content)
        if is_final:
            logger.info("%schat_with_self_coded_fc 收到 final，结束", prefix)
            final_msg = (user_msg or "").strip() or (action.get("message") or "").strip()
            # 仅当用户指令与「用户/权限」相关时，才用代码解析 getent passwd 生成表格（避免 IP 配置等场景被误替换）
            first_user_content = ""
            for m in messages:
                if m.get("role") == "user":
                    first_user_content = (m.get("content") or "")[:500]
                    break
            is_user_list_query = any(
                k in first_user_content for k in ("用户", "权限", "passwd", "getent", "账号", "账户")
            )
            is_disk_df_query = any(
                k in first_user_content for k in ("df", "磁盘", "空间", "挂载", "容量", "使用率")
            )
            if is_user_list_query:
                for msg in reversed(messages):
                    if msg.get("role") == "user" and "<tool_result>" in (msg.get("content") or ""):
                        tool_result = msg.get("content", "")
                        parsed_table = _parse_getent_passwd_to_table(tool_result)
                        if parsed_table:
                            intro = "已查询系统中的用户列表，以下是基于 getent passwd 命令输出的汇总：\n\n"
                            final_msg = intro + parsed_table
                            logger.info("%schat_with_self_coded_fc 代码层面解析 getent passwd 生成表格 len=%s", prefix, len(final_msg))
                            break
            elif is_disk_df_query:
                df_table = _build_df_table_from_messages(messages)
                if df_table:
                    intro = "已按资产汇总磁盘使用情况（基于 df 命令输出）：\n\n"
                    final_msg = intro + df_table
                    logger.info("%schat_with_self_coded_fc 代码层面解析 df 按资产生成表格 len=%s", prefix, len(final_msg))
            # 当本轮 content 实为长 reasoning（如 R1 只给 reasoning 无 content），且含表格/用户列表，而解析出的 message 很短时，优先采用长内容作为最终回复
            if (
                len(content) > 2000
                and "|" in content
                and content.count("|") >= 3
                and (("用户" in content or "UID" in content) or _looks_like_final_summary(content))
                and len(final_msg) < 500
            ):
                final_msg = content.strip()[:12000]
                if len(content) > 12000:
                    final_msg += "\n\n(以上为摘要，内容已截断。)"
                logger.info("%schat_with_self_coded_fc 采用本轮长 reasoning/总结作为最终回复 len=%s", prefix, len(final_msg))
            if _is_final_message_fluff(final_msg):
                # 补发一轮：要求根据命令输出和用户意图给出专业总结，避免只回复「任务已完成」
                summary_nudge = (
                    "请根据上述 <tool_result> 和用户指令，用 1～2 句话给出专业总结（如：服务/命令执行状态、是否异常、建议），不要只回复「任务已完成」或空话。若结果为用户列表、容器列表等可表格化的数据，请在 message 中用 Markdown 表格汇总，**表格中每一行必须来自 <tool_result> 中的实际输出，禁止添加未在输出中出现的用户/容器等**。示例：| 用户 | UID | Shell |\\n|------|-----|-------|\\n| root | 0 | /bin/bash |。只输出 {\"action\": \"final\", \"message\": \"你的总结\"}。"
                )
                messages.append({"role": "user", "content": summary_nudge})
                if on_turn:
                    on_turn("user", summary_nudge)
                # 给足 max_tokens，避免总结或表格在 message 中被截断（本地模型不限制）
                extra = chat_once(client, model, messages, max_tokens=8192, temperature=0.2)
                extra = (extra or "").strip()
                # 若像截断的 JSON（以 {"action 开头但解析失败且很短），再试一次
                if extra and extra.startswith('{"action') and len(extra) < 80:
                    extra_action = _parse_self_coded_action(extra)
                    if not extra_action:
                        logger.info("%schat_with_self_coded_fc 总结轮回复似截断 len=%s，重试一次", prefix, len(extra))
                        extra = chat_once(client, model, messages, max_tokens=8192, temperature=0.2)
                        extra = (extra or "").strip()
                if extra:
                    extra_action = _parse_self_coded_action(extra)
                    if extra_action and (extra_action.get("action") or "").strip() == "final":
                        extra_msg = (extra_action.get("message") or "").strip()
                        if extra_msg and not _is_final_message_fluff(extra_msg):
                            final_msg = extra_msg
                            logger.info("%schat_with_self_coded_fc 补发一轮后得到有效总结", prefix)
                    # 未解析出 JSON 但返回了长推理/总结（如 R1 只给 reasoning 无 content），若像总结或含表格则直接采用
                    if _is_final_message_fluff(final_msg) and len(extra) > 400 and not extra.strip().startswith("{"):
                        if _looks_like_final_summary(extra) or ("|" in extra and extra.count("|") >= 3 and len(extra) > 100):
                            final_msg = extra.strip()
                            logger.info("%schat_with_self_coded_fc 补发一轮后采用长文本总结（非 JSON）", prefix)
                # 若第一轮总结只得到 reasoning（长文本无 JSON），再补发一次极简提示只要一行 JSON
                if _is_final_message_fluff(final_msg) and extra and len(extra) > 400 and not extra.strip().startswith("{"):
                    short_nudge = '只回复一行 JSON，不要任何推理或解释：{"action": "final", "message": "根据上方 <tool_result> 写 1～2 句总结或表格，表格仅包含输出中实际出现的用户/容器，禁止臆造"}'
                    messages.append({"role": "user", "content": short_nudge})
                    if on_turn:
                        on_turn("user", short_nudge)
                    extra2 = chat_once(client, model, messages, max_tokens=8192, temperature=0.1)
                    extra2 = (extra2 or "").strip()
                    if extra2:
                        extra_action2 = _parse_self_coded_action(extra2)
                        if extra_action2 and (extra_action2.get("action") or "").strip() == "final":
                            extra_msg2 = (extra_action2.get("message") or "").strip()
                            if extra_msg2 and not _is_final_message_fluff(extra_msg2):
                                final_msg = extra_msg2
                                logger.info("%schat_with_self_coded_fc 补发第二轮（只回复 JSON）后得到有效总结", prefix)
                        # 第二轮仍未解析出 JSON 但返回了长总结/表格时，直接采用
                        if _is_final_message_fluff(final_msg) and len(extra2) > 100:
                            if _looks_like_final_summary(extra2) or ("|" in extra2 and extra2.count("|") >= 3):
                                final_msg = extra2.strip()
                                logger.info("%schat_with_self_coded_fc 补发第二轮后采用长文本总结（非 JSON）", prefix)
                if _is_final_message_fluff(final_msg):
                    logger.info("%schat_with_self_coded_fc final message 仍为空话，使用兜底总结", prefix)
                    final_msg = "任务已完成。"
            return final_msg
        messages.append({"role": "user", "content": user_msg or "(无输出)"})
        if on_turn:
            on_turn("user", user_msg or "(无输出)")
    return "已达到最大轮数，任务未完成。"


def _looks_intent_to_execute_without_tool(content: str) -> bool:
    """检测是否为「说要执行命令但未调用工具」或「只给建议未执行」的回复，用于触发一次提醒。"""
    if not (content or "").strip():
        return False
    c = content.strip()
    # 中文：我将/我们将/请执行/我会执行/接下来执行/将执行/让我执行/执行的下一步命令/下一步我们将/接下来 + 执行/命令/检查
    intent_phrases = ("我将", "我们将", "请执行", "我会执行", "接下来执行", "将执行", "让我执行", "执行的下一步命令", "现在，让我执行", "下一步，我们将", "接下来")
    if any(p in c for p in intent_phrases) and ("执行" in c or "命令" in c or "检查" in c):
        return True
    # 误以为由程序返回 <tool_result>：如「请回复 <tool_result> 后续执行结果」——必须先输出 <tool_call> 才会得到 tool_result
    if "请回复" in c and ("<tool_result>" in c or "tool_result" in c):
        return True
    # 英文：Recommended Actions / Check ... with `cmd` / Use `cmd` 等只给建议未调工具
    if ("Recommended" in c or "Recommended Actions" in c) and "`" in c:
        return True
    if ("Check " in c or "check " in c) and " with `" in c:
        return True
    if ("Use `" in c or " use `" in c) or ("Consider " in c and "`" in c):
        return True
    if "建议" in c and ("执行" in c or "命令" in c or "`" in c):
        return True
    # 只认错不执行：承认违规/未调用工具，或说「正确流程应是」但未在本轮调工具，或让用户「告知具体目标」
    if ("错误更正" in c or "违规操作" in c or "未严格遵守" in c) and ("execute_command" in c or "工具" in c):
        return True
    if "正确处理流程应是" in c or "请直接告知具体目标" in c or "请告知具体目标" in c:
        return True
    return False


def _looks_asset_selection_request(content: str) -> bool:
    """检测模型是否在让用户重新「指定操作资产」，用于自动回击并继续执行。"""
    if not (content or "").strip():
        return False
    c = content.strip()
    phrases = (
        "指定操作资产",
        "指定操作对象",
        "请指定操作资产",
        "请仔细检查并指定操作资产",
        "请指定要操作的资产",
        "请先指定资产",
        "没有任何指定",
        "无法继续执行操作",
        "我无法继续执行操作",
    )
    return any(p in c for p in phrases)


def chat_with_tools(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict],
    tool_handlers: dict,
    trace_id: str | None = None,
    on_turn: Callable[[str, str], None] | None = None,
) -> tuple[str, list[dict]]:
    """带 function calling 的对话：若 AI 返回 tool_calls 则执行并继续，否则返回最终回复。on_turn 用于记录交互。"""
    prefix = f"[{trace_id}] " if trace_id else ""
    nudge_count = 0  # 已因「只说执行未调工具」提醒的轮数，最多 MAX_NUDGES 次
    while True:
        logger.info("%schat_with_tools 请求 model=%s messages条数=%s", prefix, model, len(messages))
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        content = (msg.content or "").strip()

        # 1) API 原生返回了 tool_calls：按 OpenAI 规范执行并继续
        if msg.tool_calls:
            messages.append(msg)
            if on_turn:
                on_turn("assistant", content or ("[API tool_calls: " + ", ".join(t.function.name for t in msg.tool_calls) + "]"))
            logger.info("%schat_with_tools 工具调用数=%s（API 原生）", prefix, len(msg.tool_calls))
            results_parts = []
            for tc in msg.tool_calls:
                name = tc.function.name
                logger.info("%s工具调用 name=%s", prefix, name)
                if name not in tool_handlers:
                    result = f"未知工具: {name}"
                else:
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception as e:
                        result = f"参数解析失败: {e}"
                    else:
                        logger.info("%s工具调用参数 args=%s", prefix, args)
                        result = tool_handlers[name](**args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    }
                )
                results_parts.append(f"【{name}】\n{result}")
            if on_turn:
                on_turn("user", "<tool_result>\n" + "\n\n".join(results_parts) + "\n</tool_result>")
            continue

        # 2) 无 tool_calls：检查是否为 Qwen 等“正文内 <tool_call>”格式
        text_tool_calls = _parse_tool_calls_from_content(content)
        if text_tool_calls:
            logger.info("%schat_with_tools 从正文解析到工具调用数=%s（Qwen 等文本格式）", prefix, len(text_tool_calls))
            messages.append({"role": "assistant", "content": content})
            results_parts = []
            for tc in text_tool_calls:
                name = tc.get("name") or ""
                args = tc.get("arguments") or {}
                logger.info("%s工具调用 name=%s args=%s", prefix, name, args)
                if name not in tool_handlers:
                    result = f"未知工具: {name}"
                else:
                    try:
                        result = tool_handlers[name](**args)
                    except Exception as e:
                        result = f"执行失败: {e}"
                results_parts.append(f"【{name}】\n{result}")
            tool_result_content = "<tool_result>\n" + "\n\n".join(results_parts) + "\n</tool_result>"
            messages.append({"role": "user", "content": tool_result_content})
            if on_turn:
                on_turn("assistant", content)
                on_turn("user", tool_result_content)
            continue

        # 3) 既无 API tool_calls 也无正文 tool_call：视为最终回复
        # 兜底：若回复像「我将执行…」却未调工具，提醒一次并再请求一轮
        if nudge_count < MAX_NUDGES and _looks_intent_to_execute_without_tool(content):
            nudge_count += 1
            logger.info("%schat_with_tools 检测到「将执行」但未调工具，补发提醒（第%s次）再请求一轮", prefix, nudge_count)
            messages.append({"role": "assistant", "content": content})
            nudge_msg = "你刚才表示要执行命令、只给出了建议、或承认了未调用工具但未在本轮调用 execute_command。不要只认错或请用户「告知具体目标」——请立即在本轮调用 execute_command 执行后续巡检/排查命令，再根据结果给出结论。"
            messages.append({"role": "user", "content": nudge_msg})
            if on_turn:
                on_turn("assistant", content)
                on_turn("user", nudge_msg)
            continue
        if nudge_count < MAX_NUDGES and _looks_asset_selection_request(content):
            nudge_count += 1
            logger.info("%schat_with_tools 检测到「请求指定资产」，补发提醒（第%s次）再请求一轮", prefix, nudge_count)
            messages.append({"role": "assistant", "content": content})
            nudge_msg = "资产已经在本次会话中由系统选定并通过前缀提示给出，你无需也不能再次要求用户「指定操作资产」或声称「无法继续执行操作」。请直接使用 execute_command 针对已选资产继续执行巡检/排查命令，不要再回复类似「请指定操作资产」「无法继续执行操作」之类的内容。"
            messages.append({"role": "user", "content": nudge_msg})
            if on_turn:
                on_turn("assistant", content)
                on_turn("user", nudge_msg)
            continue
        logger.info("%schat_with_tools 最终回复长度=%s", prefix, len(content))
        return content, messages
    return "", messages


def chat_with_prompt_tools(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tool_handlers: dict,
    trace_id: str | None = None,
    default_asset_name: str | None = None,
    on_turn: Callable[[str, str], None] | None = None,
) -> tuple[str, list[dict]]:
    """
    通过「提示词 + 解析回复」实现函数调用：不传 tools 给 API，由模型在正文中输出
    <tool_call>{"name":"...","arguments":{...}}</tool_call>，解析后执行并注入 <tool_result>，循环直到模型返回纯文字。
    适用于不支持原生 tool_calls 的本地模型；调用方需在 system 中已加入 PROMPT_TOOLS_INSTRUCTION。
    default_asset_name：单资产时传入，用于解析「仅含 command 的 JSON」块（如 ```json {"command":"..."} ```）。
    on_turn 用于记录交互。
    """
    prefix = f"[{trace_id}] " if trace_id else ""
    nudge_count = 0
    while True:
        logger.info("%schat_with_prompt_tools 请求 model=%s messages条数=%s", prefix, model, len(messages))
        content = chat_once(client, model, messages)
        content = (content or "").strip()

        text_tool_calls = _parse_tool_calls_from_content(content)
        if not text_tool_calls and default_asset_name:
            text_tool_calls = _parse_command_only_json(content, default_asset_name)
            if text_tool_calls:
                logger.info("%schat_with_prompt_tools 从「仅含 command 的 JSON」解析到工具调用数=%s", prefix, len(text_tool_calls))
        if text_tool_calls:
            logger.info("%schat_with_prompt_tools 解析到工具调用数=%s", prefix, len(text_tool_calls))
            messages.append({"role": "assistant", "content": content})
            results_parts = []
            for tc in text_tool_calls:
                name = tc.get("name") or ""
                args = tc.get("arguments") or {}
                logger.info("%schat_with_prompt_tools 工具 name=%s args=%s", prefix, name, args)
                if name not in tool_handlers:
                    result = f"未知工具: {name}"
                else:
                    try:
                        result = tool_handlers[name](**args)
                    except Exception as e:
                        result = f"执行失败: {e}"
                results_parts.append(f"【{name}】\n{result}")
            tool_result_content = "<tool_result>\n" + "\n\n".join(results_parts) + "\n</tool_result>"
            messages.append({"role": "user", "content": tool_result_content})
            if on_turn:
                on_turn("assistant", content)
                on_turn("user", tool_result_content)
            continue

        if nudge_count < MAX_NUDGES and _looks_intent_to_execute_without_tool(content):
            nudge_count += 1
            logger.info("%schat_with_prompt_tools 检测到「将执行」但未调工具，补发提醒（第%s次）再请求一轮", prefix, nudge_count)
            messages.append({"role": "assistant", "content": content})
            nudge_msg = "你刚才表示要执行命令、给出了建议、或承认了未调用工具但未在本轮输出 <tool_call>。不要只认错或请用户「告知具体目标」——请立即在本轮输出 <tool_call>{\"name\":\"execute_command\",\"arguments\":{\"asset_name\":\"...\",\"command\":\"...\"}}</tool_call> 执行后续巡检/排查命令（如 journalctl -p err -b 等），再根据结果给出结论。"
            messages.append({"role": "user", "content": nudge_msg})
            if on_turn:
                on_turn("assistant", content)
                on_turn("user", nudge_msg)
            continue
        if nudge_count < MAX_NUDGES and _looks_asset_selection_request(content):
            nudge_count += 1
            logger.info("%schat_with_prompt_tools 检测到「请求指定资产」，补发提醒（第%s次）再请求一轮", prefix, nudge_count)
            messages.append({"role": "assistant", "content": content})
            nudge_msg = "资产已经在本次会话中由系统选定并通过前缀提示给出，你无需也不能再次要求用户「指定操作资产」或声称「无法继续执行操作」。请直接输出 <tool_call>{\"name\":\"execute_command\",\"arguments\":{\"asset_name\":\"...\",\"command\":\"...\"}}</tool_call> 针对已选资产继续执行巡检/排查命令，不要再回复类似「请指定操作资产」「无法继续执行操作」之类的内容。"
            messages.append({"role": "user", "content": nudge_msg})
            if on_turn:
                on_turn("assistant", content)
                on_turn("user", nudge_msg)
            continue
        logger.info("%schat_with_prompt_tools 最终回复长度=%s", prefix, len(content))
        return content, messages
