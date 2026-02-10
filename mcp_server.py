#!/usr/bin/env python3
"""
本地 MCP Server：暴露运维助手能力（list_assets / execute_command）给任意 MCP 客户端。

架构说明：
- 模型不直接 call tool，而是由 Orchestrator 解析模型输出后，通过 MCP Client 调用本 Server。
- 本 Server 仅做「工具注册 + 执行」，与 config.yaml / SSH 复用现有逻辑。

运行方式（二选一）：
  # 编排器调用工具（推荐，仅提供 POST /api/tool、/tool，无 MCP 挂载，避免校验冲突）
  pip install "mcp[cli]"
  AI_OPS_CONFIG=config.yaml python mcp_server.py

  # 或仅工具 HTTP：AI_OPS_CONFIG=config.yaml python mcp_server.py --tool-http

  # MCP Inspector / Claude Code 需另起进程（不同端口），例如：
  # AI_OPS_CONFIG=config.yaml python -c "from mcp_server import _mcp; _mcp.run(transport='streamable-http')" --port 8003
  # 然后连接 http://localhost:8003/mcp
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 项目根目录，保证可 import ai_ops_assistant
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ai_ops_assistant.config import load_config
from ai_ops_assistant.ssh_executor import execute_on_asset, get_asset_by_name, list_assets_display


def _get_config():
    path = Path(os.environ.get("AI_OPS_CONFIG", ROOT / "config.yaml"))
    return load_config(path)


# 兼容 MCP SDK v1 (FastMCP) 与 v2 (MCPServer)
try:
    from mcp.server.fastmcp import FastMCP
    _mcp = FastMCP("AI-Ops", json_response=True)
except ImportError:
    try:
        from mcp.server.mcpserver import MCPServer
        _mcp = MCPServer("AI-Ops", json_response=True)
    except ImportError:
        print("请安装 MCP SDK: pip install 'mcp[cli]'", file=sys.stderr)
        sys.exit(1)


@_mcp.tool()
def list_assets() -> str:
    """列出当前配置的所有 Linux 资产（名称与连接信息），用于确认要对哪台机器执行操作。"""
    config = _get_config()
    return list_assets_display(config)


@_mcp.tool()
def execute_command(asset_name: str, command: str) -> str:
    """在指定的 Linux 资产上执行 shell 命令。asset_name 必须在 list_assets 返回的列表中；command 为完整 shell 命令，如 df -h。"""
    config = _get_config()
    asset = get_asset_by_name(config, asset_name)
    if not asset:
        return f"未找到资产 '{asset_name}'。可用资产：\n{list_assets_display(config)}"
    return execute_on_asset(asset, command)


def _run_tool(name: str, arguments: dict) -> str:
    """供 REST /tool 与 MCP 共用的工具执行。"""
    if name == "list_assets":
        return list_assets()
    if name == "execute_command":
        cmd = (arguments.get("command") or "").replace("\\n", "\n").replace("\\t", "\t")
        return execute_command(
            arguments.get("asset_name") or arguments.get("asset") or "",
            cmd,
        )
    return f"未知工具: {name}"


# ---------- REST /tool：供 Orchestrator 通过 HTTP 调用（实现「通过 MCP 做 func call」） ----------
# 使用 /api/tool 避免与 MCP streamable HTTP 对 query.request 的校验冲突；仍保留 /tool 兼容旧配置
_TOOL_PATH = "/api/tool"
_TOOL_PATH_LEGACY = "/tool"


def _create_tool_app():
    import logging as _log
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    _logger = _log.getLogger("mcp_server.tool")

    async def _handle_tool(req: Request) -> JSONResponse:
        """纯 Starlette 处理，不经过 FastAPI，避免任何 query/body 校验。"""
        try:
            body = await req.body()
        except Exception as e:
            _logger.warning("POST (tool) 读取 body 失败: %s", e)
            return JSONResponse(status_code=200, content={"result": f"读取请求体失败: {e}"})
        _logger.info("POST (tool) 收到 body 长度=%s", len(body))
        try:
            body_decoded = json.loads(body.decode("utf-8")) if body else {}
        except Exception as e:
            _logger.warning("POST (tool) JSON 解析失败 raw_len=%s: %s", len(body), e)
            return JSONResponse(status_code=200, content={"result": f"请求体不是合法 JSON: {e}"})
        if not isinstance(body_decoded, dict):
            return JSONResponse(status_code=200, content={"result": "请求体必须为 JSON 对象"})
        name = (body_decoded.get("name") or "").strip() if isinstance(body_decoded.get("name"), str) else ""
        arguments = body_decoded.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        if not name:
            return JSONResponse(status_code=200, content={"result": "缺少或无效的 name 参数（应为 list_assets 或 execute_command）"})
        try:
            out = _run_tool(name, arguments)
            return JSONResponse(status_code=200, content={"result": out})
        except Exception as e:
            _logger.exception("执行工具 %s 失败", name)
            return JSONResponse(status_code=200, content={"result": f"执行失败: {type(e).__name__}: {e}"})

    routes = [
        Route(_TOOL_PATH, _handle_tool, methods=["POST"]),
        Route(_TOOL_PATH_LEGACY, _handle_tool, methods=["POST"]),
    ]
    app = Starlette(debug=False, routes=routes)
    return app


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="AI 运维助手 MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default=os.environ.get("MCP_TRANSPORT", "streamable-http"),
        help="stdio 供 Cursor/Claude 子进程连接，streamable-http 供 Inspector/浏览器",
    )
    parser.add_argument(
        "--tool-http",
        action="store_true",
        help="仅启动 REST /tool 服务，供 Orchestrator 调用（实现通过 MCP 的 func call）",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP 监听地址（仅 streamable-http / tool-http）")
    parser.add_argument("--port", type=int, default=8002, help="HTTP 端口（默认 8002 避免与主站 uvicorn 8000 冲突）")
    args = parser.parse_args()

    if args.tool_http:
        app = _create_tool_app()
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
        return

    if args.transport == "streamable-http":
        # 仅提供 /api/tool 与 /tool，不挂载 MCP app，避免 MCP SDK 挂载时对 query.request 的校验影响 POST /api/tool（见 python-sdk#1367）
        # 若需 MCP Inspector，请另起进程：AI_OPS_CONFIG=config.yaml python -c "from mcp_server import _mcp; _mcp.run(transport='streamable-http')" --port 8003
        import uvicorn
        tool_app = _create_tool_app()
        uvicorn.run(tool_app, host=args.host, port=args.port)
    else:
        _mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
