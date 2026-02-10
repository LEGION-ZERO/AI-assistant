"""资产管理持久化：SQLite 存储资产列表与资产组（与会话共用同一 DB 文件）。"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("ai_ops_assistant.asset_db")

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


def _row_to_asset(r: sqlite3.Row) -> dict[str, Any]:
    return {
        "name": r["name"],
        "host": r["host"],
        "port": int(r["port"]),
        "username": r["username"],
        "password": r["password"] if r["password"] else None,
        "private_key_path": r["private_key_path"] if r["private_key_path"] else None,
        "group_id": r["group_id"] if r["group_id"] is not None else None,
    }


def init_db() -> None:
    """创建 asset_groups、assets 表（若不存在），并为 assets 添加 group_id 列（兼容旧库）。"""
    _ensure_dir()
    with _lock:
        conn = sqlite3.connect(_get_db_path())
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS asset_groups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    sort_order INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS assets (
                    name TEXT PRIMARY KEY,
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL DEFAULT 22,
                    username TEXT NOT NULL,
                    password TEXT,
                    private_key_path TEXT
                )
                """
            )
            # 兼容旧库：若无 group_id 列则添加
            cur = conn.execute("PRAGMA table_info(assets)")
            cols = [row[1] for row in cur.fetchall()]
            if "group_id" not in cols:
                conn.execute("ALTER TABLE assets ADD COLUMN group_id INTEGER REFERENCES asset_groups(id) ON DELETE SET NULL")
            # 兼容旧库：asset_groups 若无 remark 列则添加
            cur = conn.execute("PRAGMA table_info(asset_groups)")
            gcols = [row[1] for row in cur.fetchall()]
            if "remark" not in gcols:
                conn.execute("ALTER TABLE asset_groups ADD COLUMN remark TEXT")
            if "parent_id" not in gcols:
                conn.execute("ALTER TABLE asset_groups ADD COLUMN parent_id INTEGER REFERENCES asset_groups(id) ON DELETE SET NULL")
            conn.commit()
        finally:
            conn.close()
    logger.info("资产数据库已初始化 path=%s", _get_db_path())


def asset_list() -> list[dict[str, Any]]:
    """返回所有资产（按 name 排序）。"""
    with _lock:
        conn = sqlite3.connect(_get_db_path())
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT name, host, port, username, password, private_key_path, group_id FROM assets ORDER BY name"
            ).fetchall()
            return [_row_to_asset(r) for r in rows]
        finally:
            conn.close()


def asset_get(name: str) -> Optional[dict[str, Any]]:
    """按 name 获取一条资产，不存在返回 None。"""
    with _lock:
        conn = sqlite3.connect(_get_db_path())
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT name, host, port, username, password, private_key_path, group_id FROM assets WHERE name = ?",
                (name,),
            ).fetchone()
            if row is None:
                return None
            return _row_to_asset(row)
        finally:
            conn.close()


def asset_create(
    name: str,
    host: str,
    port: int = 22,
    username: str = "",
    password: Optional[str] = None,
    private_key_path: Optional[str] = None,
    group_id: Optional[int] = None,
) -> None:
    """插入一条资产。"""
    _ensure_dir()
    with _lock:
        conn = sqlite3.connect(_get_db_path())
        try:
            conn.execute(
                "INSERT INTO assets (name, host, port, username, password, private_key_path, group_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (name, host, port, username, password or None, private_key_path or None, group_id),
            )
            conn.commit()
        finally:
            conn.close()


def asset_update(
    name: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    private_key_path: Optional[str] = None,
    group_id: Optional[int] = None,
) -> bool:
    """更新一条资产，返回是否更新了记录。"""
    with _lock:
        conn = sqlite3.connect(_get_db_path())
        try:
            cur = conn.execute("SELECT name FROM assets WHERE name = ?", (name,))
            if cur.fetchone() is None:
                return False
            updates = []
            params = []
            if host is not None:
                updates.append("host = ?")
                params.append(host)
            if port is not None:
                updates.append("port = ?")
                params.append(port)
            if username is not None:
                updates.append("username = ?")
                params.append(username)
            if password is not None:
                updates.append("password = ?")
                params.append(password if (password or "").strip() else None)
            if private_key_path is not None:
                updates.append("private_key_path = ?")
                params.append(private_key_path if (private_key_path or "").strip() else None)
            if group_id is not None:
                updates.append("group_id = ?")
                params.append(group_id)
            if not updates:
                return True
            params.append(name)
            conn.execute("UPDATE assets SET " + ", ".join(updates) + " WHERE name = ?", params)
            conn.commit()
            return True
        finally:
            conn.close()


def asset_delete(name: str) -> bool:
    """删除一条资产，返回是否删除了记录。"""
    with _lock:
        conn = sqlite3.connect(_get_db_path())
        try:
            cur = conn.execute("DELETE FROM assets WHERE name = ?", (name,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


# ---------- 资产组 ----------

def group_list() -> list[dict[str, Any]]:
    """返回所有组（扁平，按 sort_order, name 排序），含 parent_id。"""
    with _lock:
        conn = sqlite3.connect(_get_db_path())
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, name, sort_order, remark, parent_id FROM asset_groups ORDER BY sort_order, name"
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "sort_order": int(r["sort_order"]),
                    "remark": r["remark"] if r["remark"] else "",
                    "parent_id": r["parent_id"] if r["parent_id"] is not None else None,
                }
                for r in rows
            ]
        finally:
            conn.close()


def group_list_tree() -> list[dict[str, Any]]:
    """返回树形结构：每项含 id, name, sort_order, remark, parent_id, children（子组列表）。"""
    flat = group_list()
    by_parent: dict[Optional[int], list[dict[str, Any]]] = {}
    for g in flat:
        pid = g.get("parent_id")
        by_parent.setdefault(pid, []).append({
            **g,
            "children": [],
        })
    def build(pid: Optional[int]) -> list[dict[str, Any]]:
        nodes = by_parent.get(pid, [])
        for n in nodes:
            n["children"] = build(n["id"])
        return sorted(nodes, key=lambda x: (x["sort_order"], x["name"]))
    return build(None)


def get_descendant_ids(gid: int) -> list[int]:
    """返回某组及其所有后代组的 id 列表（含自身）。"""
    flat = group_list()
    children_map: dict[Optional[int], list[int]] = {}
    for g in flat:
        pid = g.get("parent_id")
        children_map.setdefault(pid, []).append(g["id"])
    result: list[int] = []
    def collect(uid: int) -> None:
        result.append(uid)
        for cid in children_map.get(uid, []):
            collect(cid)
    collect(gid)
    return result


def group_get(gid: int) -> Optional[dict[str, Any]]:
    """按 id 获取一个组。"""
    with _lock:
        conn = sqlite3.connect(_get_db_path())
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT id, name, sort_order, remark, parent_id FROM asset_groups WHERE id = ?", (gid,)
            ).fetchone()
            if row is None:
                return None
            return {
                "id": row["id"],
                "name": row["name"],
                "sort_order": int(row["sort_order"]),
                "remark": row["remark"] if row["remark"] else "",
                "parent_id": row["parent_id"] if row["parent_id"] is not None else None,
            }
        finally:
            conn.close()


def group_create(
    name: str,
    sort_order: int = 0,
    remark: Optional[str] = None,
    parent_id: Optional[int] = None,
) -> int:
    """创建组，返回 id。"""
    _ensure_dir()
    with _lock:
        conn = sqlite3.connect(_get_db_path())
        try:
            cur = conn.execute(
                "INSERT INTO asset_groups (name, sort_order, remark, parent_id) VALUES (?, ?, ?, ?)",
                (name, sort_order, remark or "", parent_id),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def group_update(
    gid: int,
    name: Optional[str] = None,
    sort_order: Optional[int] = None,
    remark: Optional[str] = None,
    parent_id: Optional[int] = None,
) -> bool:
    """更新组，返回是否更新了记录。parent_id 可改为其他组或 None（根组），不可设为自身或后代以防循环。"""
    if parent_id is not None and parent_id == gid:
        return False
    if parent_id is not None and gid in get_descendant_ids(parent_id):
        return False  # 不能以后代为父（会成环）
    with _lock:
        conn = sqlite3.connect(_get_db_path())
        try:
            cur = conn.execute("SELECT id FROM asset_groups WHERE id = ?", (gid,))
            if cur.fetchone() is None:
                return False
            updates = []
            params = []
            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if sort_order is not None:
                updates.append("sort_order = ?")
                params.append(sort_order)
            if remark is not None:
                updates.append("remark = ?")
                params.append(remark)
            if parent_id is not None:
                updates.append("parent_id = ?")
                params.append(parent_id)
            elif parent_id is None:
                updates.append("parent_id = ?")
                params.append(None)
            if not updates:
                return True
            params.append(gid)
            conn.execute("UPDATE asset_groups SET " + ", ".join(updates) + " WHERE id = ?", params)
            conn.commit()
            return True
        finally:
            conn.close()


def group_delete(gid: int) -> bool:
    """删除组：子组的 parent_id 改为本组的 parent_id（上移一层），本组资产 group_id 置为 NULL，再删除本组。"""
    with _lock:
        conn = sqlite3.connect(_get_db_path())
        try:
            cur = conn.execute("SELECT parent_id FROM asset_groups WHERE id = ?", (gid,))
            row = cur.fetchone()
            if row is None:
                return False
            parent_id = row[0]
            conn.execute("UPDATE asset_groups SET parent_id = ? WHERE parent_id = ?", (parent_id, gid))
            conn.execute("UPDATE assets SET group_id = NULL WHERE group_id = ?", (gid,))
            cur = conn.execute("DELETE FROM asset_groups WHERE id = ?", (gid,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


def _asset_public_row(r: sqlite3.Row) -> dict[str, Any]:
    """单行转对外暴露的资产信息（不含密码）。"""
    return {
        "name": r["name"],
        "host": r["host"],
        "port": int(r["port"]),
        "username": r["username"],
        "auth_type": "password" if (r["password"] or "").strip() else "key",
        "private_key_path": r["private_key_path"] if r["private_key_path"] else None,
        "group_id": r["group_id"] if r["group_id"] is not None else None,
    }


def asset_list_tree() -> dict[str, Any]:
    """返回树形结构：groups（含各组及其 assets）、ungrouped（未分组资产）。"""
    groups = group_list()
    all_assets = asset_list()
    by_group: dict[Optional[int], list[dict[str, Any]]] = {g["id"]: [] for g in groups}
    by_group[None] = []
    for a in all_assets:
        gid = a.get("group_id")
        by_group.setdefault(gid, []).append({
            "name": a["name"],
            "host": a["host"],
            "port": a["port"],
            "username": a["username"],
            "auth_type": "password" if (a.get("password") or "").strip() else "key",
            "private_key_path": a.get("private_key_path"),
        })
    return {
        "groups": [
            {"id": g["id"], "name": g["name"], "sort_order": g["sort_order"], "assets": by_group.get(g["id"], [])}
            for g in groups
        ],
        "ungrouped": by_group.get(None, []),
    }
