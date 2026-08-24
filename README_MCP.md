# agent_roundtable MCP Server

> 本 MCP 服务是基于 [Random-Walk2026/agent_roundtable](https://github.com/Random-Walk2026/agent_roundtable)（作者 [Lamjinlab](https://github.com/Random-Walk2026)）的封装与扩展，感谢原作者。

把 `agent_roundtable` 封装成一个 **MCP 服务**，让任意支持 MCP 的 Agent（DSH、Claude Desktop、Cursor 等）可以直接调用：

- **调研时**：用 `search_knowledge` 检索本地专家语料库，用 `list_agents` / `get_agent` 了解可用的专家视角。
- **调研后**：用 `run_roundtable` 召集多位专家 Agent 圆桌讨论，把你整理好的调研发现通过 `context` 注入给专家组，输出结构化最终总结和 Markdown 报告。

## 快速开始

### 1. 安装依赖

要求 Python 3.10+（建议 3.13）。在仓库根目录执行：

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # macOS / Linux
python -m pip install -r requirements-mcp.txt
cp .env.example .env           # 可选：填入真实的 LLM API key
```

> `llm_gateway` 是可选的。没有它时服务也能跑：`mock` 模式直接可用，真实 LLM 走服务内置的 OpenAI 兼容客户端（见下文「LLM 配置」）。

### 2. 启动服务

默认以 `stdio` 传输方式启动（适合 DSH / Claude Desktop / Cursor 等本地客户端）：

```bash
.venv\Scripts\python mcp_entry.py
# 等价写法：python -m mcp_server
```

也可以跑 HTTP 传输：

```bash
.venv\Scripts\python mcp_entry.py --transport sse
.venv\Scripts\python mcp_entry.py --transport streamable-http
```

### 3. 注册到你的 Agent

#### DSH（DeepSeek Harness）

DSH 的 MCP 服务器通过 `cordis.patch.yml` 配置。在 `$DSH_HOME/profiles/<profile>/cordis.patch.yml`（例如 `web` 或 `headless` profile）的顶层数组里加一条 `insert`：

```yaml
- insert:
    - id: mcp-agent-roundtable
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: agent-roundtable
        transport: stdio
        command: E:/code/讨论agentMCP/agent_roundtable/.venv/Scripts/python.exe
        args: ['mcp_entry.py']
        cwd: E:/code/讨论agentMCP/agent_roundtable
        toolCallTimeoutMs: 600000   # 真实 LLM 圆桌可能超过默认 60s
```

保存后用 `dsh --profile <name> --dump-config` 确认配置被组合识别，然后**重启 DSH** 生效。工具会以 `mcp__agent-roundtable__<tool>` 的形式出现在会话里。

> **DSH 集成注意（踩坑记录）**：
>
> 1. DSH 的 `dsh-tools` 只接受极小的 JSON Schema 子集（单一 `type`、`properties`/`required`/`additionalProperties`、`items`、`enum`/`const`、`oneOf`），**不接受 `anyOf`**。因此 MCP 工具参数**不能用 `Optional`/`X | None`**（FastMCP 会生成 `anyOf`，导致 DSH 的 `ListToolsRequest` 校验失败、DSH 加载报错）——所有可选参数请用默认值（`""`/`0`）表示"未提供"。
>
> 2. `run_roundtable` 用真实 LLM 时单次可能超过 MCP 默认 60s 超时，DSH 配置里要加 `toolCallTimeoutMs: 600000`。

#### Claude Code

用 `claude mcp add` 注册（scope 可选 `user` 全局 / `local` 当前项目）：

```bash
claude mcp add agent-roundtable --scope user -- \
  E:/code/讨论agentMCP/agent_roundtable/.venv/Scripts/python.exe \
  E:/code/讨论agentMCP/agent_roundtable/mcp_entry.py
claude mcp list    # 应显示 √ Connected
```

> 注意：Claude Code 的 stdio MCP 子进程以**启动目录**为工作目录（无 cwd 配置项）。本服务的 `mcp_entry.py` 已基于自身文件位置注入仓库根到 `sys.path`，因此从任何目录启动都能工作；日志也会统一写到仓库的 `logs/`。首次调用工具需在交互会话里批准权限，或用 `--allowedTools "mcp__agent-roundtable__*"` 预授权。

#### Claude Desktop / Cursor 等通用客户端

在 `claude_desktop_config.json`（或客户端对应的 MCP 配置）里写 stdio 条目：

```json
{
  "mcpServers": {
    "agent-roundtable": {
      "command": "E:/code/讨论agentMCP/agent_roundtable/.venv/Scripts/python.exe",
      "args": ["mcp_entry.py"],
      "cwd": "E:/code/讨论agentMCP/agent_roundtable"
    }
  }
}
```

## 工具列表

| 工具 | 用途 | 关键参数 |
| --- | --- | --- |
| `list_councils` | 列出可用的圆桌配置（专家名单） | - |
| `list_agents` | 列出所有已配置的专家 Agent | - |
| `get_agent` | 查看单个专家的完整人设卡 | `agent_id` |
| `list_knowledge_corpora` | 列出本地 RAG 语料库 | - |
| `search_knowledge` | 检索本地语料（调研时用） | `corpus_id`, `query`, `top_k`, `knowledge_scope` |
| `plan_council` | 开会前分析阵容、给出调整/添加专家建议 | `topic`, `council` |
| `run_roundtable` | 召集圆桌讨论（调研后用，支持定制阵容） | `topic`, `council`, `rounds`, `mock`, `provider`, `context`, `add_personas`, `adjust_personas` 等 |
| `list_reports` | 列出已生成的报告 | `output_dir`, `limit` |
| `read_report` | 读取报告内容（仅限 `output_dir` 内） | `path`, `max_chars` |
| `provider_info` | 查看哪些 LLM provider 已配置 | - |

### `run_roundtable` 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `topic` | 必填 | 讨论主题 |
| `council` | `experts` | 圆桌名称（见 `list_councils`） |
| `rounds` | `2` | 讨论轮数（1-10） |
| `mock` | `false` | 离线模拟模式，不需要 key |
| `provider` | `auto` | `auto`/`mock`/`deepseek`/`openai`/`openrouter`/`gemini`/自定义 |
| `model` | 空 | 覆盖 provider 默认模型 |
| `api_key` | 空 | 显式 key（默认从 `.env` 读） |
| `base_url` | 空 | 自定义 OpenAI 兼容端点 |
| `context` | 空 | 调研发现/背景资料，注入为第 0 轮消息供专家组参考 |
| `add_personas` | 空 | JSON 数组字符串，新增专家（每项至少含 `name` 和 `role`） |
| `adjust_personas` | 空 | JSON 数组字符串，调整现有专家（每项含 `id` 与要覆盖的字段） |
| `output_dir` | `logs` | 报告输出目录 |
| `temperature` | `0.7` | 生成温度 |
| `max_output_tokens` | `4096` | 单次生成上限 |

## 典型使用流程

一个 Agent 的完整「调研 → 讨论」流程：

1. **了解可选专家**：`list_councils` / `list_agents`，选定圆桌（如 `experts` 或 `china_debt`）。
2. **调研时查语料**：`search_knowledge(corpus_id="macroeconomics", query="总需求与货币政策")`，把命中的段落作为调研依据。
3. **开会前确认阵容**：`plan_council(topic, council)` 返回当前成员、角色和调整建议，主 Agent 据此判断是否需要**添加新专家**或**调整现有专家**以适应当前需求。
4. **调研后召集讨论**（可按需定制阵容）：

```json
{
  "topic": "AI 对投资和就业的长期影响",
  "council": "experts",
  "rounds": 2,
  "mock": false,
  "provider": "auto",
  "context": "调研发现：1) 本地语料显示……；2) 在线资料显示……；3) 需要重点讨论……",
  "add_personas": "[{\"name\": \"Energy Expert\", \"role\": \"能源与产业政策专家\", \"worldview\": \"从能源供需、地缘政治与技术替代分析问题\"}]",
  "adjust_personas": "[{\"id\": \"history\", \"role\": \"技术史与信息史专家\"}]"
}
```

`add_personas` 每条支持字段：`name`（必填）、`role`（必填）、`worldview`、`speaking_style`、`strengths`、`weaknesses`、`catchphrases`、`rag_expert_name`、`profile`（`id` 可选，缺省由 name 生成）。`adjust_personas` 每条：`id`（必填，须是现有成员）+ 任意可覆盖字段。

5. **消费结果**：工具返回 `final_summary` + 完整 `messages`（每条带 provider/model 与认识论标签），同时报告写入 `logs/`，可用 `list_reports` / `read_report` 读取。

## LLM 配置

`run_roundtable` 的 `provider` 参数可选：

| provider | 说明 | 读取的 key |
| --- | --- | --- |
| `mock` | 确定性本地模拟，不需要任何 key（跑通流程用） | - |
| `auto` | 按 `deepseek → openai → openrouter → gemini` 顺序取第一个有 key 的；都没有则退回 mock | 见下 |
| `deepseek` | DeepSeek OpenAI 兼容接口 | `DEEPSEEK_API_KEY`（`DEEPSEEK_BASE_URL` 可改） |
| `openai` | OpenAI | `OPENAI_API_KEY`（`OPENAI_BASE_URL` 可改） |
| `openrouter` | OpenRouter | `OPENROUTER_API_KEY_1` / `OPENROUTER_API_KEY_2` |
| `gemini` | Gemini OpenAI 兼容接口 | `GEMINI_API_KEY_1` / `GEMINI_API_KEY_2` |
| 任意字符串 | 自定义 OpenAI 兼容端点 | 需要同时传 `base_url` 与 `api_key` |

另外可用 `model` / `temperature` / `max_output_tokens` / `api_key` / `base_url` 覆盖默认值。Key 只从仓库根目录的 `.env` 读取，不会写进日志或报告。

## 安全性

- **API key 不进仓库**：key 只从 `.env`（已被 `.gitignore` 忽略）或环境变量读取，代码零硬编码；`provider_info` 只报告"是否已配置"，不回显 key。
- **报告读取受限**：`read_report` 只允许读取 `output_dir`（默认 `logs/`）内的文件，绝对路径或 `..` 穿越到目录外会被拒绝，避免 MCP 客户端读取宿主机任意文件。
- **LLM 请求容错**：内置 OpenAI 兼容客户端带 2 次重试，网络抖动不会直接中断圆桌。

## 与 llm_gateway 的关系

原项目的 `llm/facade.py`、`llm/router.py`、`rag/config.py` 顶层依赖 `llm_gateway`（同级私有包）。本服务已把它们改成**可选依赖**：

- 没有 `llm_gateway`：`mock` 直接可用；真实 LLM 由本服务内置的 OpenAI 兼容客户端提供（走 `.env` key 直连）。
- 有 `llm_gateway`（`pip install -e ../llm_gateway`）：原项目的 provider-chain、CLI 订阅（Claude/Codex）、CLIProxyAPI 等能力全部保留，`create_llm` / `create_llm_from_config` 自动走网关。

## 目录结构

```text
mcp_server/
├── __init__.py      # 包定义
├── __main__.py      # python -m mcp_server 入口
├── llm_client.py    # .env 加载 + OpenAI 兼容客户端（含重试）+ provider 解析
├── core.py          # 全部工具实现（可单测，不依赖 MCP 会话）
└── server.py        # FastMCP 层：工具注册 + 进度上报
mcp_entry.py         # 仓库根入口：python mcp_entry.py
requirements-mcp.txt # MCP 服务依赖
tests/test_mcp_tools.py
```

## 测试

```bash
.venv\Scripts\python -m pytest tests/test_mcp_tools.py -q
```

## 常见问题

**Q：没有 API key 能试吗？**

能。`run_roundtable` 传 `"mock": true`（或 `provider: "mock"`）即可离线跑通完整流程。

**Q：为什么返回的 JSON 里中文在终端显示乱码？**

是 Windows 控制台编码（GBK）显示问题，数据本身是 UTF-8；报告文件也是 UTF-8。

**Q：`search_knowledge` 返回 0 条？**

仓库默认不带语料正文（只有 `.gitkeep` 占位）。先按项目 README 把 Markdown 资料放进 `knowledge/experts/<领域>/`，再运行 `python -m rag.ingest --expert-name <领域> --embedding-provider keyword` 重建索引。

**Q：能不能让讨论接着上一轮继续？**

可以把上一轮的报告内容作为新一轮 `run_roundtable` 的 `context` 传入，专家组会先读背景再发言；也可以直接提高 `rounds`。

**Q：注册到 DSH 后工具没出现？**

先 `dsh --profile <name> --dump-config` 确认配置被组合识别；再确认重启了 DSH（web profile 的 HMR 默认禁用，改配置必须重启进程）；最后在**新会话**里查看工具（工具列表在会话开始时快照）。