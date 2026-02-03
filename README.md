# Linux 智能运维助手

接入 **DeepSeek**，通过自然语言指令自动完成 Linux 资产运维：服务器巡检、自动部署、查看磁盘/内存/进程等。你只需输入指令，其余由 AI 规划并在目标机器上执行。

## 功能概览

- **自然语言指令**：如「对 web-server-01 做一次巡检」「检查所有服务器磁盘使用率」
- **DeepSeek 接入**：使用 OpenAI 兼容 API，规划步骤并决定执行哪些命令
- **Linux 执行**：通过 SSH 在配置的资产上执行命令，结果回传 AI 继续分析
- **资产管理**：在 `config.yaml` 中配置多台 Linux 服务器（密码或密钥认证）

<img width="1886" height="842" alt="image" src="https://github.com/user-attachments/assets/6ce09a77-ecbb-4da9-8a11-3498b57d89f6" />
<img width="1920" height="875" alt="image" src="https://github.com/user-attachments/assets/b49d075f-1f5a-4410-a7b4-7dde8a63dcee" />




## 快速开始

### 1. 安装依赖

```bash
cd D:\project\AI-assistant
pip install -r requirements.txt
```

### 2. 配置

复制示例配置并填写：

```bash
copy config.example.yaml config.yaml
```

编辑 `config.yaml`：

- **deepseek.api_key**：必填，从 [DeepSeek 控制台](https://platform.deepseek.com/api_keys) 获取；也可设置环境变量 `DEEPSEEK_API_KEY` 而不写入文件
- **assets**：添加你的 Linux 服务器（host、username、password 或 private_key_path）

### 3. 运行

**方式一：Web 界面（FastAPI）**

```bash
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0
```

浏览器访问 `http://localhost:8000`，在页面输入指令并点击「执行」即可。  
**资产管理**：访问 `http://localhost:8000/assets` 可增删改 Linux 资产（配置写入 config.yaml），并可上传文件到指定资产。

- **API 文档**：`http://localhost:8000/docs`
- **提交指令**：`POST /api/run`，请求体 `{"instruction": "对 linux_222 做一次巡检"}`，返回 `{"reply": "...", "commands": [...]}`
- **资产列表**：`GET /api/assets` 返回已配置资产（不含密码）
- **资产管理**：`POST /api/assets` 新增、`PUT /api/assets/{name}` 更新、`DELETE /api/assets/{name}` 删除
- **上传文件**：`POST /api/upload`（multipart：file、asset_name、remote_path 可选）将文件上传到指定资产

**方式二：命令行直接运行（无需安装包）**

```bash
python run.py "对 web-server-01 做一次巡检"
python run.py -i   # 交互模式，输入 exit 退出
```

**方式三：安装后以模块运行**

```bash
pip install -e .
python -m ai_ops_assistant.cli "对 web-server-01 做一次巡检"
ai-ops-assistant "对 web-server-01 做一次巡检"
```

**更多示例**

```bash
python run.py "对 linux_222 做一次巡检"
python run.py "列出所有服务器并检查磁盘使用率"
python run.py "在 db-server-01 上查看 top 前 10 进程和内存"
```

执行过程中，CLI 会打印每次在对应资产上执行的命令与结果，最后输出助手的总结。

## 项目结构

```
AI-assistant/
├── main.py                # FastAPI Web 入口
├── run.py                 # 命令行入口（无需安装包）
├── static/
│   └── index.html         # Web 前端页面
├── config.example.yaml    # 配置示例（勿提交密钥）
├── config.yaml            # 实际配置（本地填写，已 gitignore）
├── requirements.txt
├── pyproject.toml
├── README.md
└── src/
    └── ai_ops_assistant/
        ├── __init__.py
        ├── config.py         # 配置加载
        ├── llm.py            # DeepSeek API + 工具定义
        ├── ssh_executor.py   # SSH 执行与资产管理
        ├── orchestrator.py   # 指令 → AI → 执行 → 反馈 编排
        └── cli.py            # 命令行入口
```

## 安全说明

- 不要将 `config.yaml` 或含 API Key / 密码的文件提交到版本库
- 建议使用 SSH 密钥而非密码；生产环境可将 API Key 放在环境变量，由程序从环境变量读取后写入配置或直接传参

## 扩展

- 可在 `llm.py` 中增加更多工具（如「上传文件」「执行脚本」），并在 `orchestrator.py` 中注册
- 可在 `config.yaml` 中增加更多资产，AI 会通过 `list_assets` 看到并选择目标机器
