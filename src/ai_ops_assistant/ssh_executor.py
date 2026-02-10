"""在 Linux 资产上通过 SSH 执行命令。优先 paramiko，不可用时回退到系统 ssh 命令（如 Windows OpenSSH）。"""
import logging
import subprocess
from pathlib import Path
from typing import Optional

from .config import AssetConfig, AppConfig

logger = logging.getLogger("ai_ops_assistant.ssh")

_paramiko_available = False
_paramiko = None

try:
    import paramiko as _paramiko
    _paramiko_available = True
except Exception:
    # ImportError（缺库）或加载 cryptography 失败等，均回退到系统 ssh
    pass


def _resolve_key_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _sshpass_available() -> bool:
    """检查系统是否有 sshpass（用于密码认证）。"""
    try:
        subprocess.run(
            ["sshpass", "-V"],
            capture_output=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _execute_paramiko(asset: AssetConfig, command: str, timeout: int) -> str:
    """使用 paramiko 执行。"""
    client = _paramiko.SSHClient()
    client.set_missing_host_key_policy(_paramiko.AutoAddPolicy())
    try:
        connect_kw: dict = {
            "hostname": asset.host,
            "port": asset.port,
            "username": asset.username,
            "timeout": timeout,
        }
        if asset.password:
            connect_kw["password"] = asset.password
        elif asset.private_key_path:
            connect_kw["key_filename"] = str(_resolve_key_path(asset.private_key_path))
        else:
            return "错误：该资产未配置 password 或 private_key_path，无法连接。"
        client.connect(**connect_kw)
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        if err.strip():
            return out + "\n[stderr]\n" + err
        return out
    except _paramiko.AuthenticationException as e:
        return f"SSH 认证失败: {e}"
    except _paramiko.SSHException as e:
        return f"SSH 错误: {e}"
    except Exception as e:
        return f"执行失败: {type(e).__name__}: {e}"
    finally:
        client.close()


def _execute_subprocess(asset: AssetConfig, command: str, timeout: int) -> str:
    """使用系统 ssh 执行。密钥认证直接 ssh；密码认证在 Linux 下可用 sshpass（需安装）。"""
    key_path = None
    if asset.private_key_path:
        p = _resolve_key_path(asset.private_key_path)
        if p.exists():
            key_path = str(p)

    use_sshpass = bool(asset.password and not key_path)
    if asset.password and not key_path and not _sshpass_available():
        return (
            "错误：当前使用系统 ssh 命令，密码认证需要二选一：\n"
            "  1) 安装 paramiko：pip install paramiko\n"
            "  2) 或安装 sshpass 后使用密码：sudo apt install sshpass（Ubuntu/Debian）"
        )

    # ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p port [-i key] user@host "cmd"
    args = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-p", str(asset.port),
    ]
    if key_path:
        args.extend(["-i", key_path])
    args.append(f"{asset.username}@{asset.host}")
    args.append(command)

    env = None
    if use_sshpass and asset.password:
        env = {**__import__("os").environ, "SSHPASS": asset.password}
        # 使用 sshpass -e，从环境变量 SSHPASS 读密码，避免出现在进程列表
        args = ["sshpass", "-e"] + args

    try:
        r = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        out = r.stdout or ""
        if r.stderr:
            out = out + "\n[stderr]\n" + r.stderr
        if r.returncode != 0 and not out.strip():
            out = f"[exit code {r.returncode}]" + (("\n" + r.stderr) if r.stderr else "")
        return out
    except FileNotFoundError as e:
        if use_sshpass and "sshpass" in str(args[:2]):
            return "错误：未找到 sshpass。请安装：sudo apt install sshpass（Ubuntu/Debian）"
        return "错误：未找到 ssh 命令。请安装 OpenSSH 客户端。"
    except subprocess.TimeoutExpired:
        return f"执行超时（{timeout} 秒）。"
    except Exception as e:
        return f"执行失败: {type(e).__name__}: {e}"


def _command_timeout(command: str, default: int = 30) -> int:
    """包管理类命令可能较慢，给更长超时。"""
    cmd_lower = (command or "").strip().lower()
    if any(x in cmd_lower for x in ("apt-get", "apt ", "apt-", "dnf ", "yum ", "zypper ")):
        return max(default, 120)
    return default


def execute_on_asset(asset: AssetConfig, command: str, timeout: int | None = None) -> str:
    """在单台资产上执行命令，返回 stdout+stderr 合并文本。"""
    if timeout is None:
        timeout = _command_timeout(command, default=60)
    logger.info("execute_on_asset 开始 asset=%s host=%s port=%s cmd=%r timeout=%s", asset.name, asset.host, asset.port, command, timeout)
    if _paramiko_available:
        result = _execute_paramiko(asset, command, timeout)
    else:
        result = _execute_subprocess(asset, command, timeout)
    logger.info("execute_on_asset 完成 asset=%s cmd=%r result_len=%s", asset.name, command, len(result or ""))
    return result


def _upload_paramiko(asset: AssetConfig, local_path: str, remote_path: str, timeout: int = 60) -> str:
    """使用 paramiko SFTP 上传文件。"""
    client = _paramiko.SSHClient()
    client.set_missing_host_key_policy(_paramiko.AutoAddPolicy())
    try:
        connect_kw: dict = {
            "hostname": asset.host,
            "port": asset.port,
            "username": asset.username,
            "timeout": timeout,
        }
        if asset.password:
            connect_kw["password"] = asset.password
        elif asset.private_key_path:
            connect_kw["key_filename"] = str(_resolve_key_path(asset.private_key_path))
        else:
            return "错误：该资产未配置 password 或 private_key_path，无法连接。"
        client.connect(**connect_kw)
        sftp = client.open_sftp()
        try:
            sftp.put(local_path, remote_path)
            return f"已上传至 {remote_path}"
        finally:
            sftp.close()
    except _paramiko.AuthenticationException as e:
        return f"SSH 认证失败: {e}"
    except _paramiko.SSHException as e:
        return f"SSH 错误: {e}"
    except Exception as e:
        return f"上传失败: {type(e).__name__}: {e}"
    finally:
        client.close()


def _upload_subprocess(asset: AssetConfig, local_path: str, remote_path: str, timeout: int = 60) -> str:
    """使用 scp 上传文件。仅支持密钥认证或 sshpass。"""
    if asset.password and not asset.private_key_path and not _sshpass_available():
        return "错误：上传文件需安装 paramiko 或 sshpass 以支持密码认证。"
    key_path = None
    if asset.private_key_path:
        p = _resolve_key_path(asset.private_key_path)
        if p.exists():
            key_path = str(p)
    use_sshpass = bool(asset.password and not key_path)
    args = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-P", str(asset.port),
    ]
    if key_path:
        args.extend(["-i", key_path])
    args.append(local_path)
    args.append(f"{asset.username}@{asset.host}:{remote_path}")
    env = None
    if use_sshpass and asset.password:
        env = {**__import__("os").environ, "SSHPASS": asset.password}
        args = ["sshpass", "-e"] + args
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
        if r.returncode == 0:
            return f"已上传至 {remote_path}"
        return (r.stderr or r.stdout or f"exit {r.returncode}").strip()
    except FileNotFoundError:
        return "错误：未找到 scp 命令。请安装 OpenSSH 客户端。"
    except subprocess.TimeoutExpired:
        return f"上传超时（{timeout} 秒）。"
    except Exception as e:
        return f"上传失败: {type(e).__name__}: {e}"


def upload_file_to_asset(
    asset: AssetConfig, local_path: str, remote_path: str, timeout: int = 60
) -> str:
    """将本地文件上传到资产上的指定路径。"""
    if _paramiko_available:
        return _upload_paramiko(asset, local_path, remote_path, timeout)
    return _upload_subprocess(asset, local_path, remote_path, timeout)


def get_asset_by_name(config: AppConfig, name: str) -> Optional[AssetConfig]:
    for a in config.assets:
        if a.name == name:
            return a
    return None


def list_assets_display(config: AppConfig) -> str:
    """返回供 AI 阅读的资产列表文本。"""
    if not config.assets:
        return "当前没有配置任何 Linux 资产，请在 config.yaml 的 assets 中添加。"
    lines = []
    for a in config.assets:
        lines.append(f"- {a.name}: {a.username}@{a.host}:{a.port}")
    return "\n".join(lines)
