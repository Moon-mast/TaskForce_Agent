# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 回复约定
回复简洁,只回复本次要写的代码,和要进行的测试,多余内容不回复

## 项目概况

**TaskForce**:基于 LangGraph 的单用户本地 Agent 工作台(企业级投资调研助手:知识库问答 + 联网调研 + 沙箱代码执行;复杂调研问题先拆解为计划、经确认后多智能体接力执行),CLI / HTTP API / Web 工作台三入口,另有 LangGraph Studio 调试点,面向实习求职的项目经历。Windows 开发机,uv 管理,Python 3.12。

**当前状态**:后端模块 R + 00~11、前端 12 个模块已全部完成;知识库内核已换代为 `src/rag_v01/`(旧 `tools/rag` 检索栈下线,方案与剩余任务见 `docs/new_module/rag_0.1/`);**当前正在收尾 plan_0.1(Plan-and-Execute,ADR-0012):里程碑 M1~M4 完成,M5 剩真实冒烟**。进度以 `docs/dev/TODO.md` 与各 `docs/new_module/*/TODO.md` 为口径,一切方案以 `docs/` 文档为准。

## 代码布局(src 七包,平级,依赖单向无环)

```
src/
├── agent/      # 主图(supervisor/answer/ask/memory/planner)+ subagents/(三子图 + 共享 ReAct 骨架)+ contracts/(Route/TaskContract/ResultSummary/Plan 系)
├── rag_v01/    # 知识库 RAG 内核:docling 解析、父子分块、Milvus 混合检索、ragas 评估(带独立 CLI)
├── tools/      # tool/(内置:文件读写、clock)、skills/、mcp/、sandbox/、websearch/、rag/(仅剩 kb_search 工具封装)
├── prompts/    # 全部提示词(md 数据包,零代码,占位符用 $name 形式)
├── settings/   # 基础设施:config、loader(load_prompt 读 md)、usage、session、model_overrides(设置页覆盖表)、db/
├── api/        # HTTP 入口:main.py + routers/(七 router:chat/knowledge/memory/skills/mcp/model_settings/health)
└── cli/        # 终端入口:repl.py + context/streaming/commands/(斜杠命令按域一文件)
dev/front/      # Web 工作台(Vue 3 + Vite 独立工程:不进 uv / hatchling / pytest;生产构建产物由 FastAPI 同源托管)
langgraph.json  # `langgraph dev` 入口,指向 agent/studio.py 的 make_studio_graph
```

依赖方向:`cli/api -> agent -> {settings, tools, rag_v01}`;`tools -> settings`;`prompts` 纯数据零依赖。仓库根的 `skills/`、`agents.md`、`mcp_config.json`、`.env` 是运行时数据(均 gitignore),不进 src。

## 文档体系(动手前必读)

| 文档 | 作用 |
|---|---|
| `docs/ARCHITECTURE.md` | **现有架构总览**:包结构、主/子图拓扑、契约与依赖规则的成文介绍 |
| `docs/dev/TODO.md` | **每次开发前必读**:任务清单与当前进度,完成一项勾一项;从第一个未勾选项继续 |
| `docs/dev/ROADMAP.md` | 开发进度唯一事实源:模块总表、依赖图、契约归属表、契约变更登记 |
| `docs/dev/NN-xxx/DEV.md` 及专项计划(如 `docs/dev/react-refactor-plan.md`) | 12 个模块的开发文档,按编号 00->11 严格顺序执行;专项计划挂 docs/dev/ 根 |
| `docs/DESIGN.md` | 架构方案定稿(what/why) |
| `docs/PROMPT-DESIGN.md` | 主/子智能体的上下文与提示词设计 |
| `docs/adr/0001-0012` | 架构决策记录(0012 = Plan-and-Execute;含被否决项,勿"修复"有意为之的设计) |
| `docs/troubleshooting/` | **开发问题与解决记录**:实际踩坑沉淀(现象/根因/解决),按模块组织,解决后及时登记,防重复踩坑 |
| `docs/new_module/` | 现行模块方案与进度(**唯一口径**,如 rag_0.1、plan_0.1;新方案文档一律建在此处) |
| `docs/improved/` | 历史方案(**已下线**,头部带失效横幅;仅存档,勿按其实施) |
| `docs/guide/` | 面向人的讲解与面试材料(非工程约束文档) |
| `docs/personal/` | 简历/演示等个人素材(非工程文档) |
| `docs/ARCH-REVIEW.md` | 架构审查快照存档(2026-09-04) |
| `docs/ARCH-REVIEW-20260923.html` | 架构审查报告(2026-09-23,8 条候选 c1-c8 已全部落地;浏览器打开) |
| `docs/demo-scripts.md` | 五条演示剧本(README 配套) |
| `dev/front/docs/` | 前端开发文档(现行,API-CONTRACT 契约三件套) |
| `CONTEXT.md` | 术语表;代码命名与沟通必须用其术语(如"任务契约"而非"prompt","会话线程"而非"会话") |

> **索引纪律**:新目录/新文档**先入表再建文件**;旧方案被取代时在头部就地加『状态+日期+继任链接』横幅,不删除历史文档。

## 开发工作流(硬性纪律)

1. **教学模式(最高优先级)**:本项目是用户的**练手项目**,目的是学习 LangGraph。**用户未明确要求时,禁止代写业务代码**--应当讲解实现思路、给出代码骨架/示例与关键 API、指出坑位,由用户亲自编写;用户写完后可请求 review 与答疑。文档、脚手架配置(pyproject / docker-compose / .env.example 等)不是学习重点,可正常代写。**测试/业务分工(硬性)**:`tests/` 下的测试代码一律由 Claude 负责编写与维护(含随模块落地的单测);业务代码由 Claude 给出**完整代码 + 改动点清单**,用户亲手誊写,Claude 不落盘写业务代码。
2. **严格按模块编号推进**,依赖未完成的模块不开工;开工前先读对应 `DEV.md`,按其分步任务清单执行,每步跑通验收命令才勾选。
3. **TODO.md 进度纪律**:每次开发前先读 `docs/dev/TODO.md` 确认进度;任务通过验收后在该文件打勾,模块完成时同步更新 ROADMAP 总表;`docs/new_module/*/TODO.md` 内的里程碑同理。
4. **契约唯一出处**:共享 schema(`Route`/`TaskContract`/`ResultSummary`/`AgentState`/Plan 系等)只在归属模块定义(见 ROADMAP §6 契约归属表,契约代码在 `agent/contracts/`),其他包一律 import;改契约必须先在 ROADMAP §7 登记。
5. **全同步**:禁止引入 async/await。FastAPI 端点一律 `def`(自动走线程池);SSE 用同步 generator + `StreamingResponse`;LangGraph 用 `graph.stream` 同步 API。
6. **双入口共用业务层**:CLI REPL 与 FastAPI 只调同一套 `build_graph()` / `run_turn()`(均在 `agent/` 包),Studio 装配点同样只调 `build_graph()`;不允许出现第二套实现。
7. **提示词一律放 `prompts/` 下的 md 文件**,经 `settings/loader.py` 的 `load_prompt(name, **slots)` 加载渲染;禁止把提示词字符串内联在节点代码里。占位符用 `$name` 形式(`string.Template`),避免与 JSON 花括号冲突。
8. **并行纪律**:同步 Send 备胎通道的并行分支各自独立取 psycopg 连接(连接非线程安全);子智能体绝不直接 `interrupt()`,缺信息时在结果摘要的 `needs_clarification` 标注。
9. 沟通与文档一律使用中文。
10. **问题沉淀**:开发中遇到 bug/坑位并解决验证后,先向用户询问是否值得沉淀,经确认再记录到 `docs/troubleshooting/<模块>.md`(格式见该目录 README:现象/根因/解决/关联);具普遍性的坑回填对应 DEV.md 坑表;不擅自记录。

## 常用命令

```bash
uv sync                        # 安装依赖(hatchling 打包 src 七包,aliyun 镜像已配)
docker compose up -d           # 起本地 PostgreSQL + pgvector(主机端口 5433;知识库 Milvus 跑在 WSL,不在此 compose 内)
uv run pytest -q               # 全量测试
uv run pytest tests/test_graph.py -q                    # 单个文件
uv run pytest tests/test_graph.py -k test_dispatch -q   # 单个测试
uv run ruff check .            # lint
uv run python -m cli.repl      # CLI REPL 入口
uv run uvicorn api.main:app --port 8010 --reload    # FastAPI 入口(端口固定 8010,勿改;8000 是沙箱隧道)
uv run langgraph dev           # LangGraph Studio 调试入口(langgraph.json -> agent/studio.py)
cd dev/front && npm run build  # 前端产物(FastAPI 同源托管);开发双进程用 npm run dev(5173,/api 由 vite proxy 转发)
cd dev/front && npm run test && npx vue-tsc --noEmit  # 前端测试与类型检查
```

各命令按对应模块 DEV.md 的验收标准为准;外部依赖(远程沙箱、MCP、Milvus)未配置时相关测试自动 skip,不得让主线阻塞在外部依赖上。import 一律 `from agent.xxx import ...`、`from tools.xxx import ...`、`from settings.xxx import ...`、`from rag_v01.xxx import ...`。

## 架构 big picture(读多份文档才能拼出的全貌)

- **Plan-and-Execute(ADR-0012,plan_0.1)**:复杂调研先计划后执行。planner 仅在生成计划时调一次 LLM(`with_structured_output(PlanDraft)`);`plan_draft`(调 LLM)与 `plan_confirm`(纯确定性、interrupt 挂起点)必须双节点拆分——resume 后整节点重跑,重跑的是零副作用的 plan_confirm,计划不漂移。用户批准后按批派发,每批完成由 supervisor 内 `_advance_plan` 纯 Python 短路推进(以 task_id 对账 ResultSummary、解锁后续步骤、分批、收口),**计划推进绝不走路由 LLM,LLM 不得擅自改计划**;plan 以 dict 存 `AgentState.plan`(无 reducer、整体覆盖、仅 plan_draft/plan_confirm/supervisor 单点写,子图不碰)。同步 Send 备胎通道不支持 plan(tasks=None 时 plan_confirm 直接 goto answer 附说明)。
- **Supervisor 拓扑与异步派发**:主图九节点 = `supervisor` 结构化路由 + `answer`/`ask`/`memory` + `plan_draft`/`plan_confirm` + 三个子图包装节点;派发主通路是 `agent/tasks.py` 的 TaskManager(线程池后台执行子图,submit 即返回可继续交互),同步 Send fan-out 仅作 tasks=None(测试/无管理器)时的备胎;结果由 supervisor 每轮 `drain_done()` 原子回收注入,answer 消费即清;`AUTO_NOTICE` 自动唤醒汇总(REPL watcher 线程 / Web 前端轮询 `GET /chat/tasks` + `POST /chat/summary`),plan 推进复用同一唤醒通路,唤醒轮以 PREFIX_SYSTEM_NOTICE 识别并短路跳过路由 LLM。
- **HITL 三挂起点**:ask(问询)、memory(`/confirm yes|no`)、plan_confirm(计划批准),全部在主图,任意时刻至多一个挂起;子图绝不 `interrupt()`(ADR-0008,挂起点清单由 ADR-0012 修订)。
- **子智能体独立上下文**:子图只收到任务契约四件套(task/user_utterance/input_data/output_schema),不读主图 messages、不知彼此存在;只回传结构化结果摘要;supervisor 是唯一叙事者。
- **执行智能体的工具三层来源**:`tools/tool/`(内置,如文件读写、clock,操作目标为沙箱工作区)+ `tools/skills/`(用户扩展)+ `tools/mcp/`(外部协议),装配在 executor 子图;知识库检索经 `tools/rag/kb_search.py` 封装暴露给 retriever。
- **记忆双层**:短期 = checkpointer(按 thread_id,PostgresSaver `from_conn_string` 连接池);长期 = Store + pgvector(仅用户事实与偏好,每消息向量检索 top-5 只注入主智能体)。
- **存储分工**:PostgreSQL + pgvector 只管会话 checkpoint 与长期记忆 Store;知识库的向量与 BM25 在 **Milvus**(HNSW + 内置 BM25,跑在 WSL,不在 docker compose 内),整栈为 rag_v01(docling 解析 + 父子分块 + 双路召回 + 手写 RRF),详见 `docs/new_module/rag_0.1/`。
- **控制面在本地、执行面在远程**:执行沙箱是用户自建的远程 Docker 服务,本仓库只做 `execute_python` 工具接入(SANDBOX_URL/API_KEY)。
- **第三装配点 LangGraph Studio**:`uv run langgraph dev` 经 langgraph.json 挂 `agent/studio.py` 的 `make_studio_graph()`;与 cli/api 的差别仅为调试便利(不挂本地 checkpointer——平台托管持久化;不接 watcher 自动唤醒——plan 靠交互轮推进),图结构与业务层完全共用 `build_graph()`。
- 模型:火山方舟豆包 / DeepSeek(LLM)+ 智谱(embedding),均 OpenAI 兼容协议;`.env` 是默认值,设置页「模型」节可覆盖 base_url/api_key/模型名(落 `.taskforce/model_config.json`,保存后重启生效)。
