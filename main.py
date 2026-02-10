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
import sqlite3
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

from ai_ops_assistant.config import AppConfig, AssetConfig, load_config  # noqa: E402
from ai_ops_assistant.orchestrator import run_instruction  # noqa: E402
from ai_ops_assistant import asset_db  # noqa: E402
from ai_ops_assistant.session_db import (  # noqa: E402
    init_db as init_session_db,
    session_delete,
    session_get,
    session_list,
    session_save,
)
from ai_ops_assistant.ssh_executor import upload_file_to_asset  # noqa: E402


app = FastAPI(title="Linux 智能运维助手")

# 流式执行任务取消：trace_id -> threading.Event，收到停止请求时 set()
_run_cancel_events: dict[str, threading.Event] = {}
# 流式执行任务状态：trace_id -> state（用于前端跨页面查看进度）
_run_states_lock = threading.Lock()
_run_states: dict[str, dict[str, Any]] = {}


def _run_state_init(trace_id: str, session_id: str, instruction: str, asset_names: Optional[list[str]]) -> None:
    with _run_states_lock:
        _run_states[trace_id] = {
            "trace_id": trace_id,
            "session_id": session_id,
            "instruction": instruction,
            "asset_names": asset_names,
            "running": True,
            "started_at": datetime.now().isoformat(),
            "commands": [],
            "model_replies": [],
            "reply": "",
            "error": "",
        }


def _run_state_append_command_start(
    trace_id: str, asset_name: str, command: str, asset_host: str | None = None
) -> None:
    with _run_states_lock:
        st = _run_states.get(trace_id)
        if not st:
            return
        st["commands"].append({
            "asset_name": asset_name,
            "command": command,
            "result": "",
            "asset_host": asset_host or "",
        })


def _run_state_update_command(
    trace_id: str, asset_name: str, command: str, result: str, asset_host: str | None = None
) -> None:
    with _run_states_lock:
        st = _run_states.get(trace_id)
        if not st:
            return
        cmds = st.get("commands") or []
        for i in range(len(cmds) - 1, -1, -1):
            if cmds[i].get("asset_name") == asset_name and cmds[i].get("command") == command and not cmds[i].get("result"):
                cmds[i]["result"] = result or ""
                if asset_host is not None:
                    cmds[i]["asset_host"] = asset_host or ""
                return
        cmds.append({
            "asset_name": asset_name,
            "command": command,
            "result": result or "",
            "asset_host": asset_host or "",
        })
        st["commands"] = cmds


def _run_state_append_model_reply(trace_id: str, round_index: int, content: str) -> None:
    with _run_states_lock:
        st = _run_states.get(trace_id)
        if not st:
            return
        st["model_replies"].append({"round": round_index, "content": content or ""})


def _run_state_finish(trace_id: str, reply: Optional[str] = None, error: Optional[str] = None) -> None:
    with _run_states_lock:
        st = _run_states.get(trace_id)
        if not st:
            return
        st["running"] = False
        st["finished_at"] = datetime.now().isoformat()
        if reply is not None:
            st["reply"] = reply or ""
        if error is not None:
            st["error"] = error or ""

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


@app.on_event("startup")
def _startup() -> None:
    init_session_db()
    asset_db.init_db()
    # 首次使用：若数据库中无资产且 config 中有，则从 config 迁移到数据库
    try:
        if not asset_db.asset_list() and CONFIG_PATH.exists():
            cfg = load_config(CONFIG_PATH)
            for a in cfg.assets:
                asset_db.asset_create(
                    name=a.name,
                    host=a.host,
                    port=a.port,
                    username=a.username,
                    password=a.password,
                    private_key_path=a.private_key_path,
                )
            if cfg.assets:
                logger.info("已从 config.yaml 迁移 %s 条资产到数据库", len(cfg.assets))
    except Exception as e:
        logger.warning("资产迁移检查失败: %s", e)


def _load() -> AppConfig:
    return load_config(CONFIG_PATH, get_assets=asset_db.asset_list)


def _asset_public(a: AssetConfig, group_id: Optional[int] = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": a.name,
        "host": a.host,
        "port": a.port,
        "username": a.username,
        "auth_type": "password" if (a.password or "").strip() else "key",
        "private_key_path": a.private_key_path,
    }
    if group_id is not None:
        out["group_id"] = group_id
    return out


def _sse(event: str, data: Any) -> str:
    # 前端以 "\n\n" 作为事件块分隔，并期望 data 为 JSON
    return f"event: {event}\n" + "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


def _build_effective_instruction(instruction: str, asset_names: Optional[list[str]]) -> str:
    """与 orchestrator 中一致的「资产前缀 + 指令」拼接，用于会话延续时新用户消息。"""
    instruction = (instruction or "").strip()
    if asset_names:
        prefix = (
            "【本次指定运维对象：仅限以下资产：" + "、".join(asset_names) + "。"
            "用户已在界面选定资产，请直接对上述资产调用 execute_command 执行用户指令，不要回复「请指定要操作的资产名称」，也不要仅用文字描述将要执行的操作而不调用 execute_command。】\n\n"
        )
        return prefix + instruction
    if any(
        k in (instruction or "")
        for k in ("所有服务器", "全部服务器", "检查所有", "所有主机", "每台", "全部主机", "所有资产")
    ):
        prefix = (
            "【用户要求检查/操作「所有」或「全部」服务器。请先 list_assets，然后对返回的**每一台**资产依次执行相应命令，不要遗漏任何一台，再输出 final 总结。】\n\n"
        )
        return prefix + instruction
    return instruction


def _session_title(instruction: str, max_len: int = 30) -> str:
    """用指令前若干字作为会话标题。"""
    s = (instruction or "").strip()
    if not s:
        return "新会话"
    s = s.replace("\n", " ").strip()
    return (s[:max_len] + "…") if len(s) > max_len else s


class RunRequest(BaseModel):
    instruction: str = Field(..., description="自然语言指令")
    asset_names: Optional[list[str]] = Field(default=None, description="可选：限制本次只可使用这些资产名")
    session_id: Optional[str] = Field(default=None, description="可选：延续该会话，不传则新建会话")
    asset_select: Optional[str] = Field(default=None, description="运维对象下拉框的原始值（如 资产名/__all__/__group__<id>）")


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
    group_id: Optional[int] = None


class AssetUpdateRequest(BaseModel):
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    private_key_path: Optional[str] = None
    group_id: Optional[int] = None


class GroupCreateRequest(BaseModel):
    name: str
    sort_order: int = 0
    remark: Optional[str] = None
    parent_id: Optional[int] = None


class GroupUpdateRequest(BaseModel):
    name: Optional[str] = None
    sort_order: Optional[int] = None
    remark: Optional[str] = None
    parent_id: Optional[int] = None


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/assets", include_in_schema=False)
def assets_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "assets.html")


@app.get("/about", include_in_schema=False)
def about_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "about.html")


@app.get("/api/assets")
def api_list_assets(group_id: Optional[int] = None) -> list[dict[str, Any]]:
    """若传 group_id，则只返回该组及其所有子组下的资产。"""
    all_assets = asset_db.asset_list()
    if group_id is not None:
        ids = set(asset_db.get_descendant_ids(group_id))
        all_assets = [a for a in all_assets if a.get("group_id") in ids]
    return [
        _asset_public(AssetConfig(**{k: v for k, v in a.items() if k != "group_id"}), a.get("group_id"))
        for a in all_assets
    ]


@app.get("/api/assets-tree")
def api_assets_tree() -> dict[str, Any]:
    """树形结构：groups（含各组及组内资产）、ungrouped（未分组资产）。避免与 /api/assets/{name} 冲突。"""
    return asset_db.asset_list_tree()


@app.get("/api/assets/{name}")
def api_get_asset(name: str) -> dict[str, Any]:
    a = asset_db.asset_get(name)
    if not a:
        raise HTTPException(status_code=404, detail="未找到该资产")
    group_id = a.get("group_id")
    clean = {k: v for k, v in a.items() if k != "group_id"}
    return _asset_public(AssetConfig(**clean), group_id)


@app.post("/api/assets")
def api_create_asset(body: AssetCreateRequest) -> dict[str, Any]:
    if asset_db.asset_get(body.name):
        raise HTTPException(status_code=400, detail="资产名称已存在")
    if (body.password or "").strip() and (body.private_key_path or "").strip():
        raise HTTPException(status_code=400, detail="password 与 private_key_path 二选一即可")
    if not (body.password or "").strip() and not (body.private_key_path or "").strip():
        raise HTTPException(status_code=400, detail="请填写 password 或 private_key_path")
    asset_db.asset_create(
        name=body.name,
        host=body.host,
        port=body.port,
        username=body.username,
        password=body.password,
        private_key_path=body.private_key_path,
        group_id=body.group_id,
    )
    return {"ok": True}


@app.put("/api/assets/{name}")
def api_update_asset(name: str, body: AssetUpdateRequest) -> dict[str, Any]:
    if not asset_db.asset_get(name):
        raise HTTPException(status_code=404, detail="未找到该资产")
    kwargs = {}
    if body.host is not None:
        kwargs["host"] = body.host
    if body.port is not None:
        kwargs["port"] = body.port
    if body.username is not None:
        kwargs["username"] = body.username
    if body.password is not None:
        kwargs["password"] = body.password or None
    if body.private_key_path is not None:
        kwargs["private_key_path"] = body.private_key_path or None
    if body.group_id is not None:
        kwargs["group_id"] = body.group_id
    if kwargs:
        asset_db.asset_update(name, **kwargs)
    return {"ok": True}


@app.get("/api/asset-groups")
def api_list_groups() -> list[dict[str, Any]]:
    return asset_db.group_list()


@app.get("/api/asset-groups/tree")
def api_groups_tree() -> list[dict[str, Any]]:
    """树形结构：每项含 id, name, sort_order, remark, parent_id, children（子组数组）。"""
    return asset_db.group_list_tree()


@app.post("/api/asset-groups")
def api_create_group(body: GroupCreateRequest) -> dict[str, Any]:
    try:
        gid = asset_db.group_create(
            name=body.name,
            sort_order=body.sort_order,
            remark=body.remark,
            parent_id=body.parent_id,
        )
        return {"ok": True, "id": gid}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="组名称已存在")


@app.put("/api/asset-groups/{gid:int}")
def api_update_group(gid: int, body: GroupUpdateRequest) -> dict[str, Any]:
    dump = body.model_dump(exclude_unset=True)
    kwargs = {}
    if "name" in dump:
        kwargs["name"] = dump["name"]
    if "sort_order" in dump:
        kwargs["sort_order"] = dump["sort_order"]
    if "remark" in dump:
        kwargs["remark"] = dump["remark"]
    if "parent_id" in dump:
        kwargs["parent_id"] = dump["parent_id"]
    if kwargs and not asset_db.group_update(gid, **kwargs):
        raise HTTPException(status_code=404, detail="未找到该组")
    return {"ok": True}


@app.delete("/api/asset-groups/{gid:int}")
def api_delete_group(gid: int) -> dict[str, Any]:
    if not asset_db.group_delete(gid):
        raise HTTPException(status_code=404, detail="未找到该组")
    return {"ok": True}


@app.delete("/api/assets/{name}")
def api_delete_asset(name: str) -> dict[str, Any]:
    if not asset_db.asset_delete(name):
        raise HTTPException(status_code=404, detail="未找到该资产")
    return {"ok": True}


@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    asset_name: str = Form(...),
    remote_path: Optional[str] = Form(default=None),
) -> JSONResponse:
    a = asset_db.asset_get(asset_name)
    if not a:
        raise HTTPException(status_code=404, detail="未找到该资产")
    asset = AssetConfig(**a)

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

    def on_command_start(asset_name: str, command: str, asset_host: str | None = None) -> None:
        logger.info("[%s] 命令开始 asset=%s cmd=%r", trace_id, asset_name, command)
        commands.append({
            "asset_name": asset_name,
            "command": command,
            "result": "",
            "asset_host": asset_host or "",
        })

    def on_command(asset_name: str, command: str, result: str, asset_host: str | None = None) -> None:
        logger.info("[%s] 命令完成 asset=%s cmd=%r result_len=%s", trace_id, asset_name, command, len(result or ""))
        for i in range(len(commands) - 1, -1, -1):
            if commands[i]["asset_name"] == asset_name and commands[i]["command"] == command and not commands[i]["result"]:
                commands[i]["result"] = result
                if asset_host is not None:
                    commands[i]["asset_host"] = asset_host or ""
                break
        else:
            commands.append({
                "asset_name": asset_name,
                "command": command,
                "result": result,
                "asset_host": asset_host or "",
            })

    try:
        config = _load()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        interaction_log_path = (Path(INTERACTION_LOG_DIR) / f"{trace_id}_{ts}.txt") if INTERACTION_LOG_DIR else None
        reply, _ = run_instruction(
            body.instruction,
            config_path=CONFIG_PATH,
            config=config,
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


@app.get("/api/run/status/{trace_id}")
def api_run_status(trace_id: str) -> dict[str, Any]:
    """查询某次流式执行的当前状态（用于前端跨页面恢复进度）。"""
    with _run_states_lock:
        st = _run_states.get(trace_id)
        if not st:
            raise HTTPException(status_code=404, detail="未找到该任务或任务已结束")
        # 返回浅拷贝，避免外部修改
        return dict(st)


@app.post("/api/run/stream")
async def api_run_stream(body: RunRequest) -> StreamingResponse:
    """
    SSE 流式执行：
    - start: {trace_id, session_id}（首条，用于停止时传入 /api/run/stop；session_id 用于前端延续会话）
    - command_start: {asset_name, command}
    - command: {asset_name, command, result}
    - reply: {reply}
    - error: {detail}
    """

    trace_id = uuid.uuid4().hex[:12]
    logger.info(
        "[%s] /api/run/stream 收到指令 instruction=%r asset_names=%s session_id=%s",
        trace_id, body.instruction, body.asset_names, body.session_id,
    )
    cancel_event = threading.Event()
    _run_cancel_events[trace_id] = cancel_event
    q: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    run_commands: list[dict[str, str]] = []  # 本轮执行的命令列表，用于写入会话 turns

    # 会话：延续已有或新建
    effective_content = _build_effective_instruction(body.instruction, body.asset_names)
    existing = session_get(body.session_id) if body.session_id else None
    if existing:
        session_id = body.session_id
        initial_messages = list(existing["messages"]) + [
            {"role": "user", "content": effective_content},
        ]
    else:
        session_id = uuid.uuid4().hex[:12]
        initial_messages = None

    def on_command_start(asset_name: str, command: str, asset_host: str | None = None) -> None:
        logger.info("[%s] SSE 推送 command_start asset=%s cmd=%r", trace_id, asset_name, command)
        _run_state_append_command_start(trace_id, asset_name, command, asset_host)
        q.put_nowait(("command_start", {"asset_name": asset_name, "command": command, "asset_host": asset_host or ""}))

    def on_command(asset_name: str, command: str, result: str, asset_host: str | None = None) -> None:
        run_commands.append({
            "asset_name": asset_name,
            "command": command,
            "result": result or "",
            "asset_host": asset_host or "",
        })
        logger.info("[%s] SSE 推送 command asset=%s cmd=%r result_len=%s", trace_id, asset_name, command, len(result or ""))
        _run_state_update_command(trace_id, asset_name, command, result or "", asset_host)
        q.put_nowait(("command", {"asset_name": asset_name, "command": command, "result": result, "asset_host": asset_host or ""}))

    def on_model_reply(round_index: int, content: str) -> None:
        logger.info("[%s] SSE 推送 model_reply round=%s len=%s", trace_id, round_index, len(content or ""))
        _run_state_append_model_reply(trace_id, round_index, content or "")
        q.put_nowait(("model_reply", {"round": round_index, "content": content or ""}))

    async def event_gen() -> AsyncGenerator[str, None]:
        yield ": stream start\n\n"
        yield _sse("start", {"trace_id": trace_id, "session_id": session_id})
        _run_state_init(trace_id, session_id, body.instruction.strip(), body.asset_names)

        done = asyncio.Event()

        config = _load()

        def _runner() -> None:
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                interaction_log_path = (Path(INTERACTION_LOG_DIR) / f"{trace_id}_{ts}.txt") if INTERACTION_LOG_DIR else None
                reply, updated_messages = run_instruction(
                    body.instruction,
                    config_path=CONFIG_PATH,
                    config=config,
                    on_command_start=on_command_start,
                    on_command=on_command,
                    on_model_reply=on_model_reply,
                    asset_names=body.asset_names,
                    trace_id=trace_id,
                    cancel_event=cancel_event,
                    interaction_log_path=interaction_log_path,
                    initial_messages=initial_messages,
                )
                logger.info("[%s] SSE 助手回复长度=%s", trace_id, len(reply or ""))
                _run_state_finish(trace_id, reply=reply or "", error=None)
                q.put_nowait(("reply", {"reply": reply}))
                if updated_messages is not None:
                    now = datetime.now().isoformat()
                    new_turn = {
                        "user": body.instruction.strip(),
                        "commands": list(run_commands),
                        "reply": reply or "",
                        "asset_select": body.asset_select,
                    }
                    existing_s = session_get(session_id)
                    if existing_s:
                        session_save(
                            session_id,
                            title=existing_s["title"],
                            created_at=existing_s["created_at"],
                            updated_at=now,
                            messages=updated_messages,
                            new_turn=new_turn,
                        )
                    else:
                        session_save(
                            session_id,
                            title=_session_title(body.instruction),
                            created_at=now,
                            updated_at=now,
                            messages=updated_messages,
                            new_turn=new_turn,
                        )
            except FileNotFoundError as e:
                logger.warning("[%s] SSE 配置文件未找到: %s", trace_id, e)
                _run_state_finish(trace_id, reply=None, error=str(e))
                q.put_nowait(("error", {"detail": str(e)}))
            except Exception as e:
                logger.exception("[%s] SSE 执行出错", trace_id)
                _run_state_finish(trace_id, reply=None, error=f"执行出错: {e}")
                q.put_nowait(("error", {"detail": f"执行出错: {e}"}))
            finally:
                done.set()
                _run_cancel_events.pop(trace_id, None)

        asyncio.get_running_loop().run_in_executor(None, _runner)

        while True:
            try:
                event, data = await asyncio.wait_for(q.get(), timeout=15)
                logger.debug("[%s] SSE 发送事件 event=%s", trace_id, event)
                yield _sse(event, data)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"

            if done.is_set() and q.empty():
                logger.info("[%s] /api/run/stream 流结束", trace_id)
                break

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/api/sessions")
def api_list_sessions() -> list[dict[str, Any]]:
    """列出所有会话（按更新时间倒序）。"""
    return session_list()


@app.get("/api/sessions/{session_id}")
def api_get_session(session_id: str) -> dict[str, Any]:
    """获取单个会话详情。turns 为聊天式轮次：每项含 user（用户原始指令）、commands（执行过程）、reply（助手回复）。"""
    s = session_get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="会话不存在")
    turns = s.get("turns") or []
    return {
        "id": s["id"],
        "title": s["title"],
        "created_at": s["created_at"],
        "updated_at": s["updated_at"],
        "turns": turns,
    }


@app.delete("/api/sessions/{session_id}")
def api_delete_session(session_id: str) -> dict[str, Any]:
    """删除会话。"""
    if not session_delete(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True}

