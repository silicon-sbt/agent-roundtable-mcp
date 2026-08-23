# agent_roundtable

一个本地运行的多 Agent 圆桌项目。

你给一个话题，系统会请几位“专家 Agent”轮流发言，主持人负责追问、总结，最后把整场讨论保存成一份 Markdown 报告。

这个项目适合做三类事：

- 想让多个不同视角的 Agent 一起讨论一个问题。
- 想给不同专家接入不同书籍、论文、长文资料，再做 RAG 检索。
- 想给具体人物接入书籍、推文、新闻等多来源语料，做风格 + 语料双驱动的圆桌。
- 想用一个简单 UI 配置每个 Agent 用什么模型，而不是每次改代码。

你不需要懂 LangGraph 才能用它。把它理解成一个“可配置的专家圆桌工具”就可以。

## 现在能做什么

- 用命令行或本地网页 UI 启动圆桌。
- 每个 Agent 可以单独设置 `provider:model@effort` 降级链。
- API key 只放在 `.env`，不会写进 JSON 或日志。
- 主线专家从 `knowledge/experts/` 读取书籍级长文；人物 Agent 可从 `knowledge/people/` 读取多来源语料。
- 报告里标明每条发言使用的 provider / model，以及认识论标签（是否命中本地语料、是否需联网核实等）。
- 多轮讨论时发言顺序会轮换，避免固定席位带来的偏见。
- 运行结果自动写入 `logs/`。
- 没有 API key 时可以用 `--mock` 先跑通流程。

当前项目有三条内置圆桌：

- `experts`：宏观、投资、计算、哲学、历史五位主题专家。
- `persona_inspired`：巴菲特、芒格、达利欧、哈耶克风格启发，以及沈栋多来源语料 Agent。
- `china_debt`：混合圆桌，宏观经济学 + 沈栋 + 历史，讨论债务与资产负债表问题。

## 3 分钟快速开始

使用 Python 3.10+，建议在独立虚拟环境中安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

`requirements.txt` 通过 `-e ../llm_gateway` 复用同级的共享 LLM 网关；本地目录需保持
`agent_roundtable/` 与 `llm_gateway/` 为兄弟目录。身份、传输、模型发现与降级链执行只在后者维护。

没有 API key，也可以先跑一个模拟版本：

```bash
python main.py --topic "AI 对投资和就业的长期影响" --council experts --rounds 1 --mock
```

混合圆桌示例：

```bash
python main.py --topic "中国债务问题会走向怎样的资产负债表衰退？" --council china_debt --rounds 1 --mock
```

如果看到终端输出总结，并出现类似下面的提示，就说明项目已经跑通：

```text
Log saved to: logs/AI对投资和就业的长期影响_20260701_123456.md
```

## 用本地 UI 启动

更推荐普通用户从 UI 开始：

```bash
python -m streamlit run ui/app.py
```

浏览器会打开一个本地页面，通常是：

```text
http://localhost:8501
```

在 UI 里你可以做这些事：

- 选择圆桌：`experts`、`persona_inspired` 或 `china_debt`。
- 设置每个 Agent 使用哪条 provider-chain。
- 从 `config/model_example.json` 复制常用模型链。
- 让新一轮讨论接着上一次 transcript 继续。
- 点击按钮保存到 `config/agent_llms.json`。
- 输入一个话题，直接调用真实 LLM 运行。
- 运行时查看进度条、当前阶段、最近事件；失败时直接看到错误详情。
- 在页面里预览最终总结和对话记录。

UI 里的 `Mock mode` 默认关闭。
如果你要测试真实 LLM，请保持它关闭。

## 配置真实 LLM

真实 API key 放在 `.env` 文件里。不要把真实 key 写到 JSON、YAML、README、日志或截图里。

OpenRouter 推荐这样命名：

```env
OPENROUTER_API_KEY_1=你的第一个_key
OPENROUTER_API_KEY_2=你的第二个_key
OPENROUTER_TIMEOUT=60
```

Gemini 可以这样配置：

```env
GEMINI_API_KEY_1=你的第一个_key
GEMINI_API_KEY_2=你的第二个_key
```

OpenAI 和 DeepSeek 也预留了配置：

```env
OPENAI_API_KEY=

DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

DeepSeek 使用 OpenAI 兼容接口，`base_url` 是 `https://api.deepseek.com`。UI 的实时目录和离线回退都由 `llm_gateway` 提供；项目内不再维护模型名或 key 名清单。

`claude` / `codex` 固定通过官方本地 CLI 复用用户自己的订阅登录态；`codex_proxy` / `grok` / `antigravity` 固定走本机 CLIProxyAPI HTTP。两类身份不能通过 `.env` 互相覆盖。Copilot 与 Ollama Cloud 已移除。模型链语法见 `config/model_example.json`：

```text
provider:model@effort
```

例如：

```text
claude:sonnet@high, codex:gpt-5.6-terra@medium, codex_proxy:gpt-5.6-terra@medium
```

`@effort` 会传给支持推理档位的订阅 / CLIProxy provider；Gemini / OpenAI / OpenRouter / DeepSeek 直连会忽略它。共享 `llm_gateway` 会把 Claude / Codex CLI 调用限制为纯文本、无工具、只读和非交互执行，并拒绝把它们改走 HTTP 或 SDK。

### 可选依赖：CLIProxyAPI

`codex_proxy` / `antigravity` / `grok` 这几类 provider-chain 条目，需要在本机另外运行 **CLIProxyAPI** 才能调用。它是一个独立的开源代理服务，把免费 Codex 账号及其他代理账号统一暴露成 OpenAI-compatible HTTP 接口；`codex_proxy` 特意与用户自己的 `codex` 订阅 CLI 分开命名。

- 项目地址：<https://github.com/router-for-me/CLIProxyAPI>
- 按它的说明启动；`llm_gateway` 默认从 `~/.cli-proxy-api/config.yaml` 只读发现地址和 key。
- 项目不再复制这两项，只需按需设置超时：

```env
CLI_PROXY_TIMEOUT=600
```

如果你不用 `codex_proxy` / `antigravity` / `grok` 这些链，就**不需要** CLIProxyAPI —— 直连 API（Gemini/OpenAI/OpenRouter/DeepSeek）和 Claude/Codex 官方本地 CLI 都不依赖它。

## 每个 Agent 用什么模型

集中配置文件是：

```text
config/agent_llms.json
```

它只负责一件事：说明每个 Agent 背后调用什么 LLM 降级链。

例子：

```json
{
  "agents": {
    "macroeconomics": {
      "provider_chain": "antigravity:gemini-3.1-pro-low, gemini:gemini-2.5-flash, claude:sonnet@high",
      "temperature": 0.3,
      "max_output_tokens": 4096
    },
    "computing": {
      "provider_chain": "codex:gpt-5.6-terra@high, antigravity:gemini-3.5-flash-low, gemini:gemini-3.5-flash",
      "temperature": 0.25,
      "max_output_tokens": 4096
    }
  }
}
```

真实 key 仍然只从 `.env` 读取。`config/model_example.json` 是 `llm-gateway refresh-example` 生成的完整速查清单，UI 关闭实时模型时只读使用它；项目不手工维护其中的模型列表。

如果你用 UI 修改模型，UI 会自动写回这个 JSON。

## 配置优先级

当你运行圆桌时，模型配置按下面的顺序生效：

1. 命令行 `--mock`、`--provider` 或 `--provider-chain`：临时覆盖所有 Agent。
2. `config/agent_llms.json`：每个 Agent 的集中模型配置。
3. Agent YAML 里的 `llm` 字段：后备配置。
4. 路由没有写 model 时，使用 `llm_gateway` 的 provider 默认值。
5. 如果没有可用 key，`auto` 会退回本地 mock。

普通使用建议：

- 想快速体验：用 `--mock`。
- 想认真使用：改 `.env` 和 `config/agent_llms.json`。
- 想临时测试某条模型链：用 `--provider-chain "claude:sonnet@high, gemini:gemini-2.5-flash"`。

## 项目结构

```text
agent_roundtable/
├── main.py                  # 命令行入口
├── ui/
│   └── app.py               # Streamlit 本地 UI
├── roundtable/              # 业务层：怎么开圆桌会
│   ├── graph.py             # 圆桌流程：主持人、Agent、总结如何串起来
│   ├── agents.py            # 每个 Agent 发言时做什么
│   ├── agent_llm_config.py  # 读写 config/agent_llms.json
│   ├── loader.py            # 读取 domain_experts / persona_inspired / councils 配置
│   ├── discovery.py         # 自动发现已配置的 Agent 与知识库目录
│   ├── epistemics.py        # 认识论标签：本地语料 / 风格推演 / 需联网核实
│   ├── order.py             # 多轮发言顺序轮换
│   ├── logger.py            # 保存 Markdown 报告
│   ├── prompts.py           # 提示词模板
│   └── state.py             # 圆桌运行时的数据结构
├── llm/                     # 圆桌专用的极薄业务适配
│   ├── facade.py            # 把 gateway chain 适配为 LLMClient.generate
│   ├── router.py            # gateway dispatch + 本项目 MockLLM 接缝
│   └── providers_api.py     # LLMClient 协议、MockLLM 与展示信息
├── config/                  # 全部声明式配置
│   ├── domain_experts/      # 主线专家：接 experts 语料
│   ├── persona_inspired/    # 人物风格启发：纯风格或 people 语料
│   ├── councils/            # 圆桌名单：哪些 Agent 一起上桌
│   ├── agent_llms.json      # 每个 Agent 使用什么模型链
│   └── model_example.json   # gateway 自动生成的完整模型速查清单
├── knowledge/               # 本地语料（不进 Git）
│   ├── experts/             # 书籍级长文，按领域分目录
│   └── people/              # 人物多来源语料：book / x / news / report
├── scripts/                 # 语料导入辅助脚本
│   ├── parse_twitter_jsonl.py
│   └── import_person_source.py
├── rag/                     # 切分文档、建立索引、检索资料
├── vector_db/chroma/        # 本地生成的 RAG 索引
├── logs/                    # 每次运行生成的 Markdown 报告
├── tests/                   # 自动测试
├── requirements.txt         # Python 依赖
└── .env.example             # 环境变量模板
```

## 几个关键文件夹是什么意思

### `config/domain_experts/` 和 `config/persona_inspired/`

这里放 Agent 的“角色卡”。

一个 Agent YAML 会描述：

- 名称、角色、世界观、说话风格
- 优点和盲点
- RAG 对应的知识目录（如有）
- `agent_type`：`domain_expert` 或 `persona_inspired`

例如 `config/domain_experts/macroeconomics.yaml` 代表“宏观经济与制度专家”。

人物 Agent 分两类：

- **纯风格**：如 `buffett`、`munger`，只靠人格设定推演，不接本地语料。
- **多来源语料**：如 `desmond_shum`，从 `knowledge/people/desmond_shum/` 检索书籍、推文、新闻、报告。

### `config/councils/`

这里放“哪几个人一起开会”。

例如 `config/councils/experts.yaml`：

```yaml
members:
  - macroeconomics
  - investing
  - computing
  - philosophy
  - history
```

混合圆桌 `config/councils/china_debt.yaml` 可以把专家与人物 Agent 放在同一张桌上：

```yaml
members:
  - macroeconomics
  - desmond_shum
  - history
```

### `knowledge/`

本地语料分两层：

**`knowledge/experts/`** — 书籍级长文，对应 `domain_expert`：

```text
knowledge/experts/macroeconomics/general_theory.md
```

**`knowledge/people/`** — 人物多来源语料，对应 `persona_inspired`：

```text
knowledge/people/desmond_shum/book/red_roulette.md
knowledge/people/desmond_shum/x/corpus.md
knowledge/people/desmond_shum/news/
knowledge/people/desmond_shum/report/
```

注意：真正的书籍、论文、推文、长文文件默认不会提交到 GitHub。
这是为了避免误传版权内容、私人资料或太大的文件。仓库里只保留目录占位（`.gitkeep`）和 `knowledge/README.md`。

### `logs/`

这里放每次圆桌生成的报告。日志文件也不会提交到 GitHub。

## 如何加入自己的 Markdown 资料

### 给主题专家加书籍

1. 找到对应目录：

```text
knowledge/experts/macroeconomics/
```

2. 把你的 `.md` 长文放进去。

3. 重建本地 RAG 索引：

```bash
python -m rag.ingest --expert-name macroeconomics --embedding-provider keyword
```

### 给人物 Agent 加多来源语料

1. 按来源类型放入子目录：

```text
knowledge/people/desmond_shum/book/
knowledge/people/desmond_shum/x/
knowledge/people/desmond_shum/news/
knowledge/people/desmond_shum/report/
```

2. 使用导入脚本（可选）：

```bash
# 从 tweets.jsonl 导出长帖 Markdown
python scripts/parse_twitter_jsonl.py ~/path/to/tweets.jsonl \
  --command export-md --originals-only --min-length 300 --lang zh

# 导入单本书籍
python scripts/import_person_source.py desmond_shum book "/path/to/book.md"
```

3. 重建索引：

```bash
# 全量索引
python -m rag.ingest --person-name desmond_shum --embedding-provider keyword

# 只增量更新某一类来源（合并进现有索引）
python -m rag.ingest --person-name desmond_shum --source-kind x --embedding-provider keyword
```

4. 再运行圆桌：

```bash
python main.py --topic "财政政策是否能稳定就业？" --council experts --rounds 1
```

运行时，配置了 RAG 的 Agent 会优先从自己的目录检索相关资料。
如果检索到资料，报告里会显示引用来源和来源类型（`book` / `x` / `news` / `report`）。

## RAG 是什么

你可以把 RAG 理解成“开卷考试”。

普通 LLM 只靠模型自己记得的东西回答。
RAG 会先从你放进 `knowledge/` 的资料里找相关段落，再把这些段落交给 Agent 参考。

这个项目当前推荐先用 `keyword`：

```bash
python -m rag.ingest --embedding-provider keyword
```

`keyword` 不需要额外 API key，适合入门和本地测试。

每个索引目录会同时生成 `chunks.jsonl` 和 `index_manifest.json`，后者记录语料范围、源文件、内容指纹与索引版本。重命名或删除语料目录后，可在全量重建时顺便清理已失效的旧索引：

```bash
python -m rag.ingest --embedding-provider keyword --prune-obsolete
```

人物语料检索时，不同来源类型有不同权重：`book` > `report` > `x` > `news`。

如果你以后要做更强的语义检索，可以再接 embedding 模型。项目里已经预留了 `mock`、`openai`、`openrouter` 等 embedding provider 的入口，但入门不需要先碰这些。

## 认识论标签

每条 Agent 发言在报告里会附带认识论标签，帮助你判断这条回答的依据：

| 标签 | 含义 |
| --- | --- |
| `本地语料` | 本次发言命中了本地 RAG 检索结果 |
| `无本地命中` | 配置了 RAG 但未检索到相关内容 |
| `纯风格推演` | 未配置 RAG，仅靠人格设定推演 |
| `需联网核实` | 涉及现实数据或时事，应自行核实 |

## 当前内置专家

主线 `experts`：

| Agent ID | 角色 | RAG |
| --- | --- | --- |
| `macroeconomics` | 宏观经济与制度专家 | 是 |
| `investing` | 长期投资与商业分析专家 | 是 |
| `computing` | 人工智能与计算思想专家 | 是 |
| `philosophy` | 哲学与思想史专家 | 是 |
| `history` | 历史与战略专家 | 是 |

副线 `persona_inspired`：

| Agent ID | 风格 | RAG |
| --- | --- | --- |
| `buffett` | 巴菲特风格启发 | 否（待接入人物语料） |
| `munger` | 芒格风格启发 | 否（待接入人物语料） |
| `dalio` | 达利欧风格启发 | 否（待接入人物语料） |
| `hayek` | 哈耶克风格启发 | 否（待接入人物语料） |
| `desmond_shum` | 沈栋风格启发 | 是（多来源语料） |

混合 `china_debt`：`macroeconomics` + `desmond_shum` + `history`。

这里的“风格启发”不是冒充本人，也不代表本人观点。它只是用类似的思考风格组织回答。

## 如何新增一个专家

假设你要新增一个“能源专家”：

1. 新建 Agent 文件：

```text
config/domain_experts/energy.yaml
```

2. 写入角色卡：

```yaml
name: "Energy Expert"
role: "能源与产业政策专家"
worldview: "从能源供需、基础设施、地缘政治和技术替代分析问题"
speaking_style: "清晰、谨慎，先讲约束再讲判断"
strengths:
  - "能源供需分析"
  - "产业链拆解"
weaknesses:
  - "可能低估金融市场短期波动"
catchphrases:
  - "先看能源约束"
rag_expert_name: "energy"
agent_type: "domain_expert"
profile:
  focus:
    - "能源安全"
    - "电力系统"
    - "油气与新能源"
```

3. 新建知识目录：

```text
knowledge/experts/energy/
```

4. 在 `config/councils/experts.yaml` 或新 council 里加入：

```yaml
members:
  - energy
```

5. 在 `config/agent_llms.json` 里给它配置模型。

6. 放入资料并重建索引：

```bash
python -m rag.ingest --expert-name energy --embedding-provider keyword
```

提示：当前测试固定约束了内置专家列表。如果你 fork 后要长期加入新专家，需要同步更新 `tests/test_structure.py`。

## 命令行常用方式

模拟运行，不需要 API key：

```bash
python main.py --topic "AI 对投资和就业的长期影响" --council experts --rounds 1 --mock
```

混合圆桌：

```bash
python main.py --topic "中国债务问题会走向怎样的资产负债表衰退？" --council china_debt --rounds 2 --mock
```

真实 LLM 运行，按 `config/agent_llms.json` 为每个 Agent 分配模型：

```bash
python main.py --topic "AI 对投资和就业的长期影响" --council experts --rounds 1
```

临时强制所有 Agent 使用 OpenRouter：

```bash
python main.py --topic "黄金、美元、比特币哪个更适合避险？" --council experts --rounds 2 --provider openrouter
```

临时强制所有 Agent 使用一条 workflow 风格降级链：

```bash
python main.py --topic "AI 会不会取代程序员？" --council experts --rounds 1 --provider-chain "claude:sonnet@high, gemini:gemini-2.5-flash"
```

指定输出目录：

```bash
python main.py --topic "AI 对就业的影响" --council experts --rounds 1 --output-dir logs
```

## 生成的报告在哪里

报告保存在 `logs/`。

文件名会根据话题和时间自动生成，例如：

```text
logs/AI对投资和就业的长期影响_20260701_123456.md
```

报告内容包括：

- 每轮主持人的问题
- 每个 Agent 的发言
- 每条发言使用的 provider / model
- 认识论标签
- RAG 引用来源及来源类型
- 每轮小结
- 最终总结

## 开源时要注意什么

默认 `.gitignore` 已经帮你避开几类不适合上传的内容：

- `.env`：真实 API key
- `knowledge/**/*.md`、`knowledge/**/*.txt`、`knowledge/**/*.pdf` 等：本地书籍、论文、推文、长文
- `logs/**`：运行报告
- `vector_db/chroma/**`、`indexes/**`：本地索引
- `__pycache__/`、`.pytest_cache/` 等临时文件

可以提交的通常是：

- 代码
- README
- `config/domain_experts/*.yaml`
- `config/persona_inspired/*.yaml`
- `config/councils/*.yaml`
- `config/agent_llms.json`
- `knowledge/README.md`
- `knowledge/**/.gitkeep` 占位文件

如果你的 `knowledge/` 里有版权书籍或私人资料，不要提交。

## 常见问题

### 我不会配置 API key，能不能先试？

可以。用 `--mock`：

```bash
python main.py --topic "测试一下" --council experts --rounds 1 --mock
```

### UI 里的模型从哪里来？

UI 直接编辑 provider-chain。关闭“实时模型”时读取 gateway 自动生成的 `config/model_example.json`；开启后，由 `llm_gateway` 统一发现本地 Codex/Claude、CLIProxyAPI 与直连 API 的模型。真实调用仍按 `config/agent_llms.json` 和 `.env` 执行，项目内没有第二份 catalog、provider registry 或 transport。

### 为什么真实 LLM 运行失败？

常见原因：

- API key 没填。
- provider-chain 写错，或对应 `.env` key / 本机官方 CLI 登录态 / CLIProxyAPI 没准备好。
- 模型名不可用。
- provider 限流或网络失败。
- 免费模型暂时不可用。

先用 `--mock` 确认项目流程没问题，再检查真实 LLM 配置。

### 我放了 Markdown，为什么没有引用？

可能是：

- 没有运行 `python -m rag.ingest` 重建索引。
- Markdown 放错目录（`experts/` vs `people/`）。
- Agent YAML 里的 `rag_expert_name` 和 `knowledge/` 目录名不一致。
- 话题和资料内容相关性不高。

### 为什么测试要求知识库是长文？

这个项目的定位是“书籍级长文驱动的专家圆桌”。
短笔记、临时摘要、抓取片段不适合放进 `knowledge/experts/` 作为正式语料。

人物语料 `knowledge/people/` 则支持多来源混合，包括推文长帖。

当前测试会保护这个约定。你自己 fork 后如果要支持短资料，可以调整测试和知识库规则。

## 开发者验证

本地提交前建议运行与 GitHub Actions 相同的核心检查（CI 覆盖 Python 3.10 与 3.13）：

```bash
python -m pip check
python -m compileall -q main.py roundtable llm rag person_crawl_agent ui tests
python -m pytest -q
python -m json.tool config/agent_llms.json >/dev/null
```

启动 UI：

```bash
python -m streamlit run ui/app.py
```

## 项目设计原则

- Agent 的长期人设放在 YAML。
- 每次实际调用什么模型放在 JSON。
- 真实 API key 只放 `.env`。
- 本地语料和运行日志不进 Git。
- 知识库分 `experts/`（领域长文）和 `people/`（人物多来源）两层。
- 先用 `keyword` RAG 跑通，再考虑更复杂的 embedding。
- CLI 和 UI 共用同一条运行链路，避免两套逻辑不一致。
