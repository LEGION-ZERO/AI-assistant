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


class AssetConfig(BaseModel):
    name: str
    host: str
    port: int = 22
    username: str
    password: Optional[str] = None
    private_key_path: Optional[str] = None

    def get_display(self) -> str:
        return f"{self.name} ({self.username}@{self.host}:{self.port})"


class AppConfig(BaseModel):
    deepseek: DeepSeekConfig = Field(default_factory=DeepSeekConfig)
    assets: list[AssetConfig] = Field(default_factory=list)


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
