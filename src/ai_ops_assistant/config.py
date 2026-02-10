"""配置加载：DeepSeek、资产列表。"""
import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class DeepSeekConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    # 为 true 时使用「提示词 + 解析回复」实现函数调用，不依赖 API 的 tool_calls（适合不支持原生 tools 的本地模型）
    use_prompt_tools: bool = False
    # 为 true 时使用「自研 Agent」：模型只输出纯 JSON（action/asset/command 或 action/final/message），由程序解析并调度执行，最适合本地 Ollama
    use_self_coded_fc: bool = False
    # 为 true 时，自研 Agent 的工具执行通过 MCP Tool HTTP 完成（需先启动 mcp_server.py --tool-http），实现「通过 MCP 做 func call」
    use_mcp_for_tools: bool = False
    mcp_tool_url: str = "http://127.0.0.1:8002/api/tool"  # use_mcp_for_tools 时 POST 地址；用 /api/tool 避免与 MCP streamable HTTP 校验冲突


class AssetConfig(BaseModel):
    name: str
    host: str
    port: int = 22
    username: str
    password: Optional[str] = None
    private_key_path: Optional[str] = None

    def get_display(self) -> str:
        return f"{self.name} ({self.username}@{self.host}:{self.port})"


class DingTalkConfig(BaseModel):
    """钉钉应用机器人：app_secret 用于校验签名；encoding_aes_key 用于解密消息（开启消息加密时必填）。
    事件订阅/URL 校验需 token（签名 Token）；加密 success 时用 app_key（Client ID/AppKey）或 corp_id（企业 ID），
    消息接收地址校验一般用 app_key。"""
    app_secret: str = ""
    encoding_aes_key: str = ""
    token: str = ""
    corp_id: str = ""
    app_key: str = ""  # Client ID / AppKey，事件订阅加密响应时使用（与 corp_id 二选一或填 app_key）


class AppConfig(BaseModel):
    deepseek: DeepSeekConfig = Field(default_factory=DeepSeekConfig)
    assets: list[AssetConfig] = Field(default_factory=list)
    dingtalk: DingTalkConfig = Field(default_factory=DingTalkConfig)


def _config_path(path: Optional[Path] = None) -> Path:
    if path is None:
        return Path(os.environ.get("AI_OPS_CONFIG", "config.yaml"))
    return path


def load_config(path: Optional[Path] = None) -> AppConfig:
    p = _config_path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"未找到配置文件 {p}，请复制 config.example.yaml 为 config.yaml 并填写。"
        )
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    config = AppConfig(**raw)
    if not config.deepseek.api_key and os.environ.get("DEEPSEEK_API_KEY"):
        config.deepseek.api_key = os.environ["DEEPSEEK_API_KEY"]
    return config


def save_config(config: AppConfig, path: Optional[Path] = None) -> None:
    """将配置写回 YAML 文件（用于资产管理增删改）。"""
    p = _config_path(path)
    data = config.model_dump()
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
