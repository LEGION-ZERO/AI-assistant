#!/usr/bin/env python3
"""根目录启动脚本：无需 pip install 也可直接运行。用法: python run.py "你的指令" 或 python run.py -i"""
import sys
from pathlib import Path

# 将 src 加入路径，便于未安装包时直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ai_ops_assistant.cli import main

if __name__ == "__main__":
    main()
