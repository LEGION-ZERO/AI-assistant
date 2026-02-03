"""CLI 入口：输入指令，交给 AI 交互完成。"""
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

from .orchestrator import run_instruction


def main() -> None:
    console = Console()
    interactive = "--interactive" in sys.argv or "-i" in sys.argv
    args = [a for a in sys.argv[1:] if a not in ("--interactive", "-i")]

    if len(args) < 1 and not interactive:
        console.print(
            Panel(
                "[bold]Linux 智能运维助手[/bold]\n\n"
                "用法:\n"
                "  [cyan]python -m ai_ops_assistant.cli \"你的指令\"[/cyan]\n"
                "  [cyan]python -m ai_ops_assistant.cli -i[/cyan]  交互模式，连续输入指令\n\n"
                "示例:\n"
                "  • 对 web-server-01 做一次巡检\n"
                "  • 列出所有服务器并检查磁盘使用率\n"
                "  • 在 db-server-01 上查看 MySQL 进程和内存\n\n"
                "请先复制 config.example.yaml 为 config.yaml 并填写 DeepSeek API Key 与资产。",
                title="使用说明",
                border_style="blue",
            )
        )
        sys.exit(0)

    def on_command(asset_name: str, command: str, result: str) -> None:
        console.print(Panel(f"[dim]$ {command}[/dim]\n\n{result}", title=f"🖥 {asset_name}", border_style="dim"))

    config_path = Path("config.yaml")

    def do_instruction(user_instruction: str) -> bool:
        """执行一条指令，成功返回 True，失败返回 False。"""
        user_instruction = user_instruction.strip()
        if not user_instruction:
            return True
        try:
            console.print("[dim]正在执行，AI 将自动规划并调用 SSH...[/dim]\n")
            reply = run_instruction(
                user_instruction,
                config_path=config_path,
                on_command=on_command,
            )
            console.print(Panel(Markdown(reply), title="✅ 助手回复", border_style="green"))
            return True
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            return False
        except Exception as e:
            console.print(f"[red]执行出错: {e}[/red]")
            return False

    if interactive:
        console.print("[bold]Linux 智能运维助手[/bold] 交互模式，输入指令后回车执行，输入 [cyan]exit[/cyan] 或 [cyan]quit[/cyan] 退出。\n")
        while True:
            user_instruction = Prompt.ask("[bold cyan]指令[/bold cyan]")
            if user_instruction.strip().lower() in ("exit", "quit", "q"):
                break
            do_instruction(user_instruction)
            console.print()
        return

    user_instruction = " ".join(args).strip()
    if not user_instruction:
        console.print("[red]请输入有效指令。[/red]")
        sys.exit(1)

    ok = do_instruction(user_instruction)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
