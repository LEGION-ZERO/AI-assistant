"""FastAPI Web 入口：提供 API 与前端页面。"""
import asyncio
import json
import tempfile
import threading
from pathlib import Path
from queue import Queue

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 确保从项目根运行时可导入
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ai_ops_assistant.orchestrator import run_instruction
from ai_ops_assistant.config import load_config, save_config
from ai_ops_assistant.config import AssetConfig as AssetConfigModel
from ai_ops_assistant.ssh_executor import get_asset_by_name, upload_file_to_asset

app = FastAPI(
    title="Linux 智能运维助手",
    description="接入 DeepSeek，通过自然语言指令完成 Linux 部署、巡检等运维任务",
    version="0.1.0",
)

# 静态资源目录（前端页面）
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class RunRequest(BaseModel):
    instruction: str
    asset_names: list[str] | None = None  # 可选，指定本次仅对这些资产执行


class CommandLog(BaseModel):
    asset_name: str
    command: str
    result: str


class RunResponse(BaseModel):
    reply: str
    commands: list[CommandLog]


class AssetItem(BaseModel):
    name: str
    host: str
    port: int
    username: str
    auth_type: str = "key"  # "password" | "key"
    private_key_path: str | None = None


class AssetCreate(BaseModel):
    name: str
    host: str
    port: int = 22
    username: str
    password: str | None = None
    private_key_path: str | None = None


class AssetUpdate(BaseModel):
    host: str | None = None
    port: int | None = None
    username: str | None = None
    password: str | None = None
    private_key_path: str | None = None


def _check_instruction_and_config(instruction: str):
    instruction = (instruction or "").strip()
    if not instruction:
        raise HTTPException(status_code=400, detail="instruction 不能为空")
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise HTTPException(
            status_code=503,
            detail="未找到 config.yaml，请复制 config.example.yaml 并配置。",
        )
    return instruction, config_path


@app.post("/api/run", response_model=RunResponse)
def api_run(req: RunRequest):
    """提交一条自然语言指令，返回 AI 回复与执行过的命令列表（非流式）。"""
    instruction, config_path = _check_instruction_and_config(req.instruction)
    commands_log: list[dict] = []

    def on_command(asset_name: str, command: str, result: str) -> None:
        commands_log.append({
            "asset_name": asset_name,
            "command": command,
            "result": result,
        })

    try:
        reply = run_instruction(
            instruction,
            config_path=config_path,
            on_command=on_command,
            asset_names=req.asset_names,
        )
        return RunResponse(reply=reply, commands=[CommandLog(**c) for c in commands_log])
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/run/stream")
def api_run_stream(req: RunRequest):
    """提交指令，以 SSE 流式返回：命令一开始就推送 command_start，执行完推送 command，最后 reply。"""
    instruction, config_path = _check_instruction_and_config(req.instruction)
    queue: Queue = Queue()

    def on_command_start(asset_name: str, command: str) -> None:
        queue.put(("command_start", {"asset_name": asset_name, "command": command}))

    def on_command(asset_name: str, command: str, result: str) -> None:
        queue.put(("command", {"asset_name": asset_name, "command": command, "result": result}))

    def run_in_thread() -> None:
        try:
            reply = run_instruction(
                instruction,
                config_path=config_path,
                on_command=on_command,
                on_command_start=on_command_start,
                asset_names=req.asset_names,
            )
            queue.put(("reply", {"reply": reply}))
        except Exception as e:
            queue.put(("error", {"detail": str(e)}))
        finally:
            queue.put(None)

    async def event_stream():
        loop = asyncio.get_event_loop()
        t = threading.Thread(target=run_in_thread)
        t.start()
        while True:
            item = await loop.run_in_executor(None, queue.get)
            if item is None:
                break
            event_type, data = item
            payload = json.dumps(data, ensure_ascii=False)
            yield f"event: {event_type}\ndata: {payload}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _get_config():
    try:
        return load_config(Path("config.yaml"))
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="未找到 config.yaml，请先复制 config.example.yaml 为 config.yaml")



@app.get("/api/assets")
def api_assets():
    """返回已配置的资产列表（不含密码，含认证方式）。"""
    try:
        config = load_config(Path("config.yaml"))
        return [
            AssetItem(
                name=a.name,
                host=a.host,
                port=a.port,
                username=a.username,
                auth_type="password" if a.password else "key",
                private_key_path=a.private_key_path,
            )
            for a in config.assets
        ]
    except FileNotFoundError:
        return []
    except Exception:
        return []


@app.get("/api/assets/{name}")
def api_asset_get(name: str):
    """获取单个资产（不含密码）。"""
    config = _get_config()
    for a in config.assets:
        if a.name == name:
            return AssetItem(
                name=a.name,
                host=a.host,
                port=a.port,
                username=a.username,
                auth_type="password" if a.password else "key",
                private_key_path=a.private_key_path,
            )
    raise HTTPException(status_code=404, detail=f"资产不存在: {name}")


@app.post("/api/assets")
def api_asset_create(body: AssetCreate):
    """新增资产。"""
    config = _get_config()
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="资产名称不能为空")
    for a in config.assets:
        if a.name == name:
            raise HTTPException(status_code=409, detail=f"资产名称已存在: {name}")
    if not body.password and not body.private_key_path:
        raise HTTPException(status_code=400, detail="请填写密码或私钥路径之一")
    config.assets.append(
        AssetConfigModel(
            name=name,
            host=(body.host or "").strip(),
            port=body.port or 22,
            username=(body.username or "").strip(),
            password=body.password or None,
            private_key_path=(body.private_key_path or "").strip() or None,
        )
    )
    save_config(config, Path("config.yaml"))
    return {"ok": True, "name": name}


@app.put("/api/assets/{name}")
def api_asset_update(name: str, body: AssetUpdate):
    """更新资产。不传的字段保持不变；password 传空字符串表示改为使用密钥。"""
    config = _get_config()
    for i, a in enumerate(config.assets):
        if a.name == name:
            new_host = body.host.strip() if body.host is not None else a.host
            new_port = body.port if body.port is not None else a.port
            new_user = body.username.strip() if body.username is not None else a.username
            new_pass = a.password
            if body.password is not None:
                new_pass = body.password if body.password else None
            new_key = a.private_key_path
            if body.private_key_path is not None:
                new_key = (body.private_key_path or "").strip() or None
            if not new_pass and not new_key:
                raise HTTPException(status_code=400, detail="至少保留密码或私钥路径之一")
            config.assets[i] = AssetConfigModel(
                name=a.name,
                host=new_host,
                port=new_port,
                username=new_user,
                password=new_pass,
                private_key_path=new_key,
            )
            save_config(config, Path("config.yaml"))
            return {"ok": True, "name": name}
    raise HTTPException(status_code=404, detail=f"资产不存在: {name}")


@app.delete("/api/assets/{name}")
def api_asset_delete(name: str):
    """删除资产。"""
    config = _get_config()
    for i, a in enumerate(config.assets):
        if a.name == name:
            config.assets.pop(i)
            save_config(config, Path("config.yaml"))
            return {"ok": True, "name": name}
    raise HTTPException(status_code=404, detail=f"资产不存在: {name}")


@app.post("/api/upload")
def api_upload(
    file: UploadFile = File(...),
    asset_name: str = Form(...),
    remote_path: str | None = Form(None),
):
    """上传文件到指定资产。remote_path 可选，不填则传到远程 /tmp/ 目录、使用原文件名。"""
    config = _get_config()
    asset = get_asset_by_name(config, asset_name)
    if not asset:
        raise HTTPException(status_code=404, detail=f"资产不存在: {asset_name}")
    filename = file.filename or "upload"
    rpath = (remote_path or "").strip() or f"/tmp/{filename}"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "").suffix) as tmp:
            content = file.file.read()
            tmp.write(content)
            tmp_path = tmp.name
        try:
            msg = upload_file_to_asset(asset, tmp_path, rpath)
            if msg.startswith("已上传"):
                return {"ok": True, "message": msg, "remote_path": rpath}
            raise HTTPException(status_code=502, detail=msg)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/assets")
def assets_page():
    """资产管理页面。"""
    index_file = STATIC_DIR / "assets.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="assets.html not found")


@app.get("/")
def index():
    """返回前端页面。"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Linux 智能运维助手 API", "docs": "/docs", "api_run": "POST /api/run"}
