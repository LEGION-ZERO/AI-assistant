"""
FastAPI Web 入口：
- 静态页面：/ (index.html), /assets (assets.html)
- 指令执行：/api/run（一次性返回）, /api/run/stream（SSE 实时流）
- 资产管理：/api/assets CRUD（写入 config.yaml）
- 文件上传：/api/upload（上传到指定资产）
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent
# 允许在未安装包的情况下直接 `uvicorn main:app`
sys.path.insert(0, str(ROOT_DIR / "src"))

from ai_ops_assistant.config import AppConfig, AssetConfig, load_config, save_config  # noqa: E402
from ai_ops_assistant.orchestrator import run_instruction  # noqa: E402
from ai_ops_assistant.ssh_executor import get_asset_by_name, upload_file_to_asset  # noqa: E402


app = FastAPI(title="Linux 智能运维助手")

# 流式执行任务取消：trace_id -> threading.Event，收到停止请求时 set()
_run_cancel_events: dict[str, threading.Event] = {}

LOG_LEVEL = os.environ.get("AI_OPS_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ai_ops_assistant.web")

STATIC_DIR = ROOT_DIR / "static"
CONFIG_PATH = Path(os.environ.get("AI_OPS_CONFIG", "config.yaml"))
# 交互日志目录：
# - 默认：<项目根目录>/logs/interaction
# - 可通过环境变量 AI_OPS_INTERACTION_LOG_DIR 覆盖
INTERACTION_LOG_DIR = os.environ.get("AI_OPS_INTERACTION_LOG_DIR") or str(ROOT_DIR / "logs" / "interaction")


def _load() -> AppConfig:
    return load_config(CONFIG_PATH)


def _asset_public(a: AssetConfig) -> dict[str, Any]:
    return {
        "name": a.name,
        "host": a.host,
        "port": a.port,
        "username": a.username,
        "auth_type": "password" if (a.password or "").strip() else "key",
        "private_key_path": a.private_key_path,
    }


def _sse(event: str, data: Any) -> str:
    # 前端以 "\n\n" 作为事件块分隔，并期望 data 为 JSON
    return f"event: {event}\n" + "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


class RunRequest(BaseModel):
    instruction: str = Field(..., description="自然语言指令")
    asset_names: Optional[list[str]] = Field(default=None, description="可选：限制本次只可使用这些资产名")


class RunResponse(BaseModel):
    reply: str
    commands: list[dict[str, str]] = Field(default_factory=list, description="命令执行记录（asset_name/command/result）")


class RunStopRequest(BaseModel):
    trace_id: str = Field(..., description="要停止的流式执行任务 ID（由 /api/run/stream 首条 start 事件返回）")


class AssetCreateRequest(BaseModel):
    name: str
    host: str
    port: int = 22
    username: str
    password: Optional[str] = None
    private_key_path: Optional[str] = None


class AssetUpdateRequest(BaseModel):
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    private_key_path: Optional[str] = None


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/assets", include_in_schema=False)
def assets_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "assets.html")


@app.get("/api/assets")
def api_list_assets() -> list[dict[str, Any]]:
    config = _load()
    return [_asset_public(a) for a in config.assets]


@app.get("/api/assets/{name}")
def api_get_asset(name: str) -> dict[str, Any]:
    config = _load()
    a = get_asset_by_name(config, name)
    if not a:
        raise HTTPException(status_code=404, detail="未找到该资产")
    return _asset_public(a)


@app.post("/api/assets")
def api_create_asset(body: AssetCreateRequest) -> dict[str, Any]:
    config = _load()
    if get_asset_by_name(config, body.name):
        raise HTTPException(status_code=400, detail="资产名称已存在")

    if (body.password or "").strip() and (body.private_key_path or "").strip():
        raise HTTPException(status_code=400, detail="password 与 private_key_path 二选一即可")
    if not (body.password or "").strip() and not (body.private_key_path or "").strip():
        raise HTTPException(status_code=400, detail="请填写 password 或 private_key_path")

    config.assets.append(
        AssetConfig(
            name=body.name,
            host=body.host,
            port=body.port,
            username=body.username,
            password=(body.password or None),
            private_key_path=(body.private_key_path or None),
        )
    )
    save_config(config, CONFIG_PATH)
    return {"ok": True}


@app.put("/api/assets/{name}")
def api_update_asset(name: str, body: AssetUpdateRequest) -> dict[str, Any]:
    config = _load()
    a = get_asset_by_name(config, name)
    if not a:
        raise HTTPException(status_code=404, detail="未找到该资产")

    if body.host is not None:
        a.host = body.host
    if body.port is not None:
        a.port = body.port
    if body.username is not None:
        a.username = body.username

    # 认证更新策略（与前端约定对齐）：
    # - PUT 里 password 为空字符串：表示清空密码（切换为密钥模式时前端会传 ""）
    # - PUT 里 password 省略/None：表示不修改密码（前端编辑密码留空时不传）
    if body.password is not None:
        a.password = (body.password or None)

    # - private_key_path 传入：更新密钥路径；传入 "" 表示清空
    if body.private_key_path is not None:
        a.private_key_path = (body.private_key_path or None)

    save_config(config, CONFIG_PATH)
    return {"ok": True}


@app.delete("/api/assets/{name}")
def api_delete_asset(name: str) -> dict[str, Any]:
    config = _load()
    before = len(config.assets)
    config.assets = [a for a in config.assets if a.name != name]
    if len(config.assets) == before:
        raise HTTPException(status_code=404, detail="未找到该资产")
    save_config(config, CONFIG_PATH)
    return {"ok": True}


@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    asset_name: str = Form(...),
    remote_path: Optional[str] = Form(default=None),
) -> JSONResponse:
    config = _load()
    asset = get_asset_by_name(config, asset_name)
    if not asset:
        raise HTTPException(status_code=404, detail="未找到该资产")

    filename = (file.filename or "upload.bin").strip() or "upload.bin"
    if not remote_path:
        remote_path = f"/tmp/{filename}"

    # 保存到临时文件，再上传
    try:
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
        msg = upload_file_to_asset(asset, tmp_path, remote_path)
        ok = not msg.startswith("错误")
        if ok:
            return JSONResponse({"ok": True, "message": msg})
        return JSONResponse({"ok": False, "detail": msg}, status_code=400)
    finally:
        try:
            if "tmp_path" in locals() and tmp_path and Path(tmp_path).exists():
                Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/api/run", response_model=RunResponse)
def api_run(body: RunRequest) -> RunResponse:
    trace_id = uuid.uuid4().hex[:12]
    logger.info("[%s] /api/run 收到指令 instruction=%r asset_names=%s", trace_id, body.instruction, body.asset_names)
    commands: list[dict[str, str]] = []

    def on_command_start(asset_name: str, command: str) -> None:
        # 非流式接口仅用于记录开始（可选）
        logger.info("[%s] 命令开始 asset=%s cmd=%r", trace_id, asset_name, command)
        commands.append({"asset_name": asset_name, "command": command, "result": ""})

    def on_command(asset_name: str, command: str, result: str) -> None:
        # 更新最后一条匹配记录
        logger.info("[%s] 命令完成 asset=%s cmd=%r result_len=%s", trace_id, asset_name, command, len(result or ""))
        for i in range(len(commands) - 1, -1, -1):
            if commands[i]["asset_name"] == asset_name and commands[i]["command"] == command and not commands[i]["result"]:
                commands[i]["result"] = result
                break
        else:
            commands.append({"asset_name": asset_name, "command": command, "result": result})

    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        interaction_log_path = (Path(INTERACTION_LOG_DIR) / f"{trace_id}_{ts}.txt") if INTERACTION_LOG_DIR else None
        reply = run_instruction(
            body.instruction,
            config_path=CONFIG_PATH,
            on_command_start=on_command_start,
            on_command=on_command,
            asset_names=body.asset_names,
            trace_id=trace_id,
            interaction_log_path=interaction_log_path,
        )
        logger.info("[%s] /api/run 助手回复长度=%s", trace_id, len(reply or ""))
        return RunResponse(reply=reply, commands=commands)
    except FileNotFoundError as e:
        logger.warning("[%s] /api/run 配置文件未找到: %s", trace_id, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("[%s] /api/run 执行出错", trace_id)
        raise HTTPException(status_code=500, detail=f"执行出错: {e}")


@app.post("/api/run/stop")
def api_run_stop(body: RunStopRequest) -> dict[str, Any]:
    """请求停止正在进行的流式执行任务（在下一轮命令前生效，当前正在执行的单条命令会跑完）。"""
    ev = _run_cancel_events.get(body.trace_id)
    if not ev:
        raise HTTPException(status_code=404, detail="未找到该任务或任务已结束")
    ev.set()
    logger.info("[%s] 已收到停止请求", body.trace_id)
    return {"ok": True, "message": "已发送停止请求"}


@app.post("/api/run/stream")
async def api_run_stream(body: RunRequest) -> StreamingResponse:
    """
    SSE 流式执行：
    - start: {trace_id}（首条，用于停止时传入 /api/run/stop）
    - command_start: {asset_name, command}
    - command: {asset_name, command, result}
    - reply: {reply}
    - error: {detail}
    """

    trace_id = uuid.uuid4().hex[:12]
    logger.info("[%s] /api/run/stream 收到指令 instruction=%r asset_names=%s", trace_id, body.instruction, body.asset_names)
    cancel_event = threading.Event()
    _run_cancel_events[trace_id] = cancel_event
    q: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def on_command_start(asset_name: str, command: str) -> None:
        logger.info("[%s] SSE 推送 command_start asset=%s cmd=%r", trace_id, asset_name, command)
        q.put_nowait(("command_start", {"asset_name": asset_name, "command": command}))

    def on_command(asset_name: str, command: str, result: str) -> None:
        logger.info("[%s] SSE 推送 command asset=%s cmd=%r result_len=%s", trace_id, asset_name, command, len(result or ""))
        q.put_nowait(("command", {"asset_name": asset_name, "command": command, "result": result}))

    def on_model_reply(round_index: int, content: str) -> None:
        logger.info("[%s] SSE 推送 model_reply round=%s len=%s", trace_id, round_index, len(content or ""))
        q.put_nowait(("model_reply", {"round": round_index, "content": content or ""}))

    async def event_gen() -> AsyncGenerator[str, None]:
        # 先发一个注释行，帮助某些代理/反向代理尽早建立流
        yield ": stream start\n\n"
        yield _sse("start", {"trace_id": trace_id})

        done = asyncio.Event()

        def _runner() -> None:
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                interaction_log_path = (Path(INTERACTION_LOG_DIR) / f"{trace_id}_{ts}.txt") if INTERACTION_LOG_DIR else None
                reply = run_instruction(
                    body.instruction,
                    config_path=CONFIG_PATH,
                    on_command_start=on_command_start,
                    on_command=on_command,
                    on_model_reply=on_model_reply,
                    asset_names=body.asset_names,
                    trace_id=trace_id,
                    cancel_event=cancel_event,
                    interaction_log_path=interaction_log_path,
                )
                logger.info("[%s] SSE 助手回复长度=%s", trace_id, len(reply or ""))
                q.put_nowait(("reply", {"reply": reply}))
            except FileNotFoundError as e:
                logger.warning("[%s] SSE 配置文件未找到: %s", trace_id, e)
                q.put_nowait(("error", {"detail": str(e)}))
            except Exception as e:
                logger.exception("[%s] SSE 执行出错", trace_id)
                q.put_nowait(("error", {"detail": f"执行出错: {e}"}))
            finally:
                done.set()
                _run_cancel_events.pop(trace_id, None)

        asyncio.get_running_loop().run_in_executor(None, _runner)

        # 直到 runner 结束且队列清空为止
        while True:
            try:
                event, data = await asyncio.wait_for(q.get(), timeout=15)
                logger.debug("[%s] SSE 发送事件 event=%s", trace_id, event)
                yield _sse(event, data)
            except asyncio.TimeoutError:
                # 心跳，避免连接被中间层关闭
                yield ": keep-alive\n\n"

            if done.is_set() and q.empty():
                logger.info("[%s] /api/run/stream 流结束", trace_id)
                break

    return StreamingResponse(event_gen(), media_type="text/event-stream")

