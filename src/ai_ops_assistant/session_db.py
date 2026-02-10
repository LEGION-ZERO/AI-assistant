"""会话持久化：SQLite 存储会话列表与 messages（JSON）。"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("ai_ops_assistant.session_db")

# 默认数据库路径：项目根目录 / data / sessions.db，可通过环境变量 AI_OPS_SESSION_DB 覆盖
_DB_PATH: Optional[str] = None
_lock = threading.Lock()


def _get_db_path() -> str:
    global _DB_PATH
    if _DB_PATH is not None:
        return _DB_PATH
    default = Path(__file__).resolve().parent.parent.parent / "data" / "sessions.db"
    _DB_PATH = os.environ.get("AI_OPS_SESSION_DB") or str(default)
    return _DB_PATH


def _ensure_dir() -> None:
    Path(_get_db_path()).parent.mkdir(parents=True, exist_ok=True)


def _message_to_dict(m: Any) -> dict:
    """将单条 message（可能是 API 返回的对象或 dict）转为可 JSON 序列化的 dict。"""
    if isinstance(m, dict):
        out = dict(m)
    elif hasattr(m, "model_dump"):
        out = m.model_dump()
    else:
        out = dict(m) if hasattr(m, "keys") else {"role": "unknown", "content": str(m)}
    # 确保嵌套也可序列化（如 tool_calls 里的对象）
    return json.loads(json.dumps(out, default=str, ensure_ascii=False))


def _messages_to_json(messages: list[Any]) -> str:
    """将 messages 列表转为 JSON 字符串存储。"""
    return json.dumps([_message_to_dict(m) for m in messages], ensure_ascii=False)


def init_db() -> None:
    """创建会话表（若不存在），并确保有 turns 列（聊天式轮次）。"""
    _ensure_dir()
    with _lock:
        conn = sqlite3.connect(_get_db_path())
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    messages TEXT NOT NULL,
                    turns TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            # 兼容旧表：若无 turns 列则添加
            cur = conn.execute("PRAGMA table_info(sessions)")
            cols = [row[1] for row in cur.fetchall()]
            if "turns" not in cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN turns TEXT NOT NULL DEFAULT '[]'")
            conn.commit()
        finally:
            conn.close()
    logger.info("会话数据库已初始化 path=%s", _get_db_path())


def session_get(session_id: str) -> Optional[dict[str, Any]]:
    """按 id 获取会话，不存在返回 None。"""
    with _lock:
        conn = sqlite3.connect(_get_db_path())
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at, messages, turns FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            messages = json.loads(row["messages"]) if row["messages"] else []
            try:
                turns = json.loads(row["turns"] or "[]")
            except Exception:
                turns = []
            return {
                "id": row["id"],
                "title": row["title"] or "",
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "messages": messages,
                "turns": turns,
            }
        finally:
            conn.close()


def session_list() -> list[dict[str, Any]]:
    """列出所有会话（不含 messages），按 updated_at 倒序。"""
    with _lock:
        conn = sqlite3.connect(_get_db_path())
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "title": r["title"] or "",
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
                for r in rows
            ]
        finally:
            conn.close()


def session_save(
    session_id: str,
    title: str,
    created_at: str,
    updated_at: str,
    messages: list[Any],
    new_turn: Optional[dict[str, Any]] = None,
) -> None:
    """
    插入或替换一条会话。
    new_turn: 可选，本轮聊天内容 { "user": "用户原始指令", "commands": [ { "asset_name", "command", "result" } ], "reply": "助手回复" }，会追加到 turns。
    """
    _ensure_dir()
    messages_json = _messages_to_json(messages)
    existing = session_get(session_id)
    if new_turn is not None:
        turns = list(existing["turns"]) if existing else []
        turns.append(new_turn)
        turns_json = json.dumps(turns, ensure_ascii=False)
    else:
        turns_json = json.dumps(existing.get("turns", []), ensure_ascii=False) if existing else "[]"
    with _lock:
        conn = sqlite3.connect(_get_db_path())
        try:
            conn.execute(
                """
                INSERT INTO sessions (id, title, created_at, updated_at, messages, turns)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    updated_at = excluded.updated_at,
                    messages = excluded.messages,
                    turns = excluded.turns
                """,
                (session_id, title, created_at, updated_at, messages_json, turns_json),
            )
            conn.commit()
        finally:
            conn.close()


def session_delete(session_id: str) -> bool:
    """删除会话，返回是否删除了记录。"""
    with _lock:
        conn = sqlite3.connect(_get_db_path())
        try:
            cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
