# 拼多多客服 Agent

面向拼多多商家的 AI 客服桌面应用，重点服务售前、售中、售后三类客服场景。项目基于 PyQt6 构建桌面工作台，通过拼多多 WebSocket 接收实时会话消息，结合商品、订单、物流、结构化知识库、RAG 检索和人工兜底策略生成客服回复。

当前版本围绕“可控客服工作流”设计：大模型不直接凭经验猜答案，而是在结构化上下文、订单状态、物流状态、知识检索、工具调用和安全规则约束下回复。知识库没有可靠命中、商品没有锁定、复杂售后、图片/视频等高风险场景会追问或转人工，尽量避免 AI 编造承诺。

本项目基于原作者 [@JC0v0](https://github.com/JC0v0) 的 `Customer-Agent` 二次开发。

- 原项目地址：`https://github.com/JC0v0/Customer-Agent`
- 原项目许可证：MIT License
- 当前仓库保留原作者来源说明，并在原项目基础上增加拼多多客服业务、三场景结构化知识库、订单/物流上下文、商品族管理、混合 RAG 检索、受控工具调用、桌面端知识库管理和更完整的测试覆盖。

## 当前版本

`v2026.06.16`

主要变化见 [CHANGELOG.md](CHANGELOG.md)。本版本重点优化 RAG 检索、场景判定、知识库编辑和人工兜底链路。

## 6.16 主要改动

### RAG 和知识库检索

- 预检索不再靠客户关键词粗略猜售前、售中、售后。
- 大场景按订单状态、最新物流状态、客户是否确认收到来判定。
- 没有订单时固定按售前处理。
- 有订单但未签收、未确认收到时固定按售中处理。
- 订单或物流显示签收，或客户明确说已收到后，才进入售后。
- 预检索 query 会结合最近上下文，短句追问可以带上前文主题，但当前问题优先，避免旧话题污染。
- 场景内检索改为混合召回：向量召回、关键词/BM25 风格召回、别名/短语精确召回、meta 重排、质量闸门。
- 检索命中质量不足时，不把明显偏题的知识直接注入给模型。
- `search_knowledge` 不再回退到旧商品知识库或旧客服知识库；新场景知识没有搜到时直接走代码层转人工。

### 知识库和商品族管理

- 新增售前、售中、售后三场景结构化知识编辑。
- 新增商品族列表，可按商品族编辑售前、售中、售后知识。
- 商品族支持“绑定链接”列表，方便把多个商品链接归到同一套结构化知识。
- 绑定链接页面新增“添加链接”，可以把新同步的商品加入已有商品族，并复制对应三场景知识结构。
- 知识库总页面新增“一键重构知识库索引”，用于重建场景知识向量索引。
- 商品族知识支持以一个基准链接同步到同族商品，减少重复维护。

### Agent 回复控制

- 商品卡、订单卡、图片、视频和客户真实文本通过 `TurnContext` 分开解析。
- 纯商品卡但没有客户问题时，不让模型展开商品参数，优先追问客户想了解什么。
- 缺少商品身份时，商品参数类问题不让模型猜，优先追问具体商品。
- 未签收订单中出现“退款、坏了、退货”等售后诉求时，先询问客户是否收到，不直接按售后承诺处理。
- 售后图片/视频、高风险售后问题场景可直接转人工。
- 工具调用触发转人工后，Agent 直接返回固定客户话术，不再让模型二次改写，避免把内部工具格式输出给客户。

### 稳定性和安全

- LLM 支持主模型超时后切换兜底模型。
- 图片 URL 会做更严格的格式清理，降低供应商接口因非法图片链接报错的概率。
- 日志敏感信息清洗覆盖 token、cookie、authorization、api_key 等字段。
- 夜间模式回复状态增加清理策略，降低长期运行的内存增长风险。
- 会话 ID 构造加强，降低缺少 `recipient_uid` 时串话的风险。
- 工具执行、消息队列、PDD 连接、登录、请求封装和资源释放增加更多边界测试。

## 业务能力

- 拼多多 WebSocket 实时消息接入和自动重连。
- 多店铺账号接入，按账号启动独立自动回复线程。
- 售前、售中、售后三场景规则加载和场景内知识检索。
- 商品卡、订单卡、客户文本、图片、视频分离解析。
- 商品知识、场景知识、订单状态和物流状态联合约束回复。
- 自动发送文本回复、商品卡片，必要时转人工。
- 商品未锁定时避免猜商品，优先询问或返回候选商品。
- 支持订单状态、物流状态、签收状态等上下文约束。
- 支持夜间模式、禁词过滤、回复清洗和兜底回复。

## 核心流程

```text
拼多多 WebSocket
    -> PDDChannel
    -> Message Queue
    -> Handler Chain
       -> MessagePreprocessor
       -> KeywordHandler
       -> AIReplyHandler
    -> CustomerAgent
       -> TurnContext 解析
       -> 订单/物流上下文刷新
       -> 售前/售中/售后场景判定
       -> 场景知识混合预检索
       -> LLM 工具调用循环
       -> 回复清洗 / 转人工
    -> SendMessage / transfer_conversation
```

关键原则：

- 客户真实文本和商品/订单元数据分开处理。
- 商品卡和订单卡只作为上下文，不直接污染客户问题。
- 场景判定优先看订单和物流，不靠关键词猜大场景。
- 知识库或工具没有明确答案时，不编造，转人工或追问。
- 图片/视频不是商品功能依据，高风险问题交给人工。

## 架构分层

```text
Agent/                  自研 CustomerAgent 和工具注册
  CustomerAgent/custom/ LLM 客户端、消息构建、工具执行、会话管理、TurnContext
  CustomerAgent/tools/  search_knowledge、send_product_card、transfer_conversation

Channel/                渠道接入
  pinduoduo/            拼多多 WebSocket、登录、消息解析和 API 封装

Message/                消息队列和处理器链
  core/                 queue、consumer、handler 管理
  handlers/             预处理、关键词、AI 回复处理

bridge/                 Context / Reply 抽象
core/                   DI 容器、服务注册、连接状态、缓存
database/               SQLAlchemy 模型、知识库服务、商品同步、向量检索
ui/                     PyQt6 桌面界面
utils/                  日志、路径、运行时资源、夜间模式等工具
scripts/                Windows 构建脚本和安全的维护脚本
app.py                  应用入口
```

## Agent 工具

| 工具 | 作用 |
| --- | --- |
| `search_knowledge` | 按店铺、商品、场景检索结构化场景知识；没有可靠命中时转人工。 |
| `send_product_card` | 发送当前商品卡；没有锁定商品时返回候选商品。 |
| `transfer_conversation` | 将会话转接给人工客服。 |

工具调用由 `ToolExecutor` 统一执行。CustomerAgent 会把工具结果回填给模型继续推理；如果工具结果已经明确转人工，则直接返回固定话术。

## TurnContext

`TurnContext` 是上下文解析底座，负责把一轮原始消息拆成稳定结构：

- `customer_text`：客户真实文本。
- `product_card`：商品 ID、商品名、规格、价格等。
- `order_card`：订单号、订单状态、物流状态、支付状态、快递单号等。
- `media`：图片、视频标记和图片链接。
- `turn_type`：当前 turn 是否包含文本、商品卡、订单卡、媒体。
- `parse_warnings`：解析异常提示。

它不做 embedding、不做意图路由、不改知识库，只负责把输入清理成可控结构。

## 模型推荐

本项目使用 OpenAI 兼容接口，建议采用“主模型 + 兜底模型 + embedding 服务”的组合。

推荐配置：

- 主模型：选择支持工具调用、上下文较长、中文客服对话稳定的模型，例如 Qwen 系列或同等级 OpenAI 兼容模型。
- 兜底模型：选择响应更稳定、延迟更低的模型，用于主模型超时或报错时接管。
- Embedding：建议使用稳定的中文向量模型，并保持本地/远程 embedding 服务可用，否则新知识索引无法完整重建。

注意：

- 售后客服不建议只依赖一个通用聊天模型裸跑，必须配合结构化知识库、订单状态、物流状态和转人工兜底。
- 不建议把 API Key 写进代码。请写入本地 `config.json`，并确保该文件不提交。

## 配置

仓库只提供 `config.example.json`，本地运行时需要复制为 `config.json`：

```powershell
Copy-Item config.example.json config.json
```

常用配置项：

| 配置 | 说明 |
| --- | --- |
| `llm.model_name` | 主模型名称。 |
| `llm.api_key` | OpenAI 兼容接口密钥，本地填写，不提交。 |
| `llm.api_base` | OpenAI 兼容接口地址。 |
| `llm.max_tokens` | 普通回复最大输出长度。 |
| `llm.tool_call_max_tokens` | 工具调用场景最大输出长度。 |
| `llm.fallback` | 兜底模型配置。 |
| `agent.token_window` | 会话上下文窗口。 |
| `agent.compress_ratio` | 触发上下文压缩的比例。 |
| `agent.max_loops` | 工具调用最大循环次数。 |
| `agent.scene_prompt_files` | 售前、售中、售后场景 Prompt 路径。 |
| `business_hours` | 人工客服工作时间。 |
| `night_mode` | 夜间模式时间。 |
| `db_path` | 本地 SQLite 数据库路径。 |

`config.json`、数据库、日志、runtime 数据和浏览器用户数据都属于本地运行数据，不应提交到仓库。

## 本地运行

环境要求：

- Windows
- Python 3.11+
- uv

安装依赖：

```powershell
uv sync
```

安装 Playwright 浏览器：

```powershell
uv run playwright install chromium
```

启动桌面应用：

```powershell
uv run python app.py
```

也可以使用项目提供的启动脚本：

```powershell
.\start_local.ps1
```

## 构建 Windows 可执行文件

构建脚本位于 `scripts/`。常用命令：

```powershell
uv run python scripts/build_win_exe.py --clean
```

更多构建说明见 [scripts/README.md](scripts/README.md)。

## License

MIT License。原项目版权归原作者所有，当前仓库在保留原许可证和来源说明的基础上进行二次开发。
