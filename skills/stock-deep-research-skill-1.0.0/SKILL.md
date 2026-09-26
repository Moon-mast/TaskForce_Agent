---
name: stock-deep-research
description: 股票投资深度调研一体化引擎：从需求澄清、结构化研究指令生成，到多智能体并行执行 8 阶段投资尽调、GoT 图控制、多源结果综合与引用验证，端到端产出带完整引用的尽调报告。当用户提到股票分析、投资研究、投资尽调、基本面分析、估值、护城河、财报分析，或直接给出股票代码/公司名（如 600519、AAPL、00700.HK）要求研究，或需要对任意主题做深度调研、多来源交叉验证、验证引用质量、综合多份研究发现时，使用此技能——即使用户只说"帮我研究一下这家公司"。
---

# Stock Deep Research（股票深度调研一体化引擎）

将 7 个原子技能整合为一个端到端工作流：

**需求澄清 → 结构化研究指令 → 执行计划 → 多智能体并行调研（GoT 增强）→ 综合汇整 → 引用验证 → 交付报告**

## 能力地图与原技能对照

| 环节 | 整合的原技能 | 详细文件（按需阅读） |
|------|-------------|---------------------|
| 阶段 0：需求澄清与指令生成 | stock-question-refiner、question-refiner | [references/question-refinement.md](references/question-refinement.md) |
| 股票尽调 8 阶段执行 | stock-research-executor | [references/stock-research-phases.md](references/stock-research-phases.md) |
| 通用主题 7 阶段研究 | research-executor | [references/general-research.md](references/general-research.md) |
| GoT 图控制（复杂主题增强） | got-controller | [references/got-controller.md](references/got-controller.md) |
| 多智能体结果综合 | synthesizer | [references/synthesizer.md](references/synthesizer.md) |
| 引用与质量验证 | citation-validator | [references/citation-validator.md](references/citation-validator.md) |
| 并行部署、搜索工具、失败恢复 | （原散落各处） | [references/execution-playbook.md](references/execution-playbook.md) |

**渐进式加载**：不要一次读完所有 references。先完成意图路由，只读当前阶段需要的文件。

## 第一步：意图路由

| 用户输入特征 | 工作流 | 下一步 |
|-------------|--------|--------|
| 提到股票/代码/公司名 + 研究、分析、尽调意图 | **股票全流程**（默认主流程） | 读 question-refinement.md，先澄清再生成指令 |
| 已有完整结构化调研指令（含 ticker、投资风格、持有周期、研究范围） | 股票执行（跳过阶段 0） | 读 stock-research-phases.md + execution-playbook.md |
| 非投资类研究问题（行业、技术、政策等开放主题） | 通用 7 阶段流程 | 读 general-research.md |
| 主题复杂/高风险/多空观点严重冲突 | 在上述流程上**叠加** GoT | 读 got-controller.md |
| 已有多份研究发现/多个 agent 产出需要合并成报告 | 仅综合 | 读 synthesizer.md |
| 已有报告需要验证引用、来源质量 | 仅验证 | 读 citation-validator.md |

路由规则：
- 只要研究对象是上市公司股票（A 股/港股/美股），一律走股票全流程；通用流程仅供非投资主题使用。
- 用户问"值不值得买 / 能不能涨" → 不给建议，直接转入股票全流程的阶段 0（见下方红线）。
- 拿不准是股票还是通用主题 → 问用户一句，不要猜。

## 端到端工作流总览

### 股票全流程（主流程）

```
阶段 0  需求澄清   问 5 组问题：基本信息 / 投资参数（风格·周期·风险）/ 研究重点 /
                   深度 / 特殊关注 → 生成结构化调研指令
                   （问题清单与指令模板见 references/question-refinement.md）
阶段 1  执行计划   验证指令完整性 → 阶段优先级排序（深挖 2-3 个重点阶段）→
                   智能体部署矩阵 → 呈现计划请用户确认
阶段 2  并行调研   8 阶段 × 每阶段 4 个 agent，单条消息并行启动，逐阶段产出报告
                   （智能体分工、输出模板、投资风格调整见 references/stock-research-phases.md；
                    agent 提示词模板与部署规范见 references/execution-playbook.md）
阶段 3  综合汇整   共识分级、矛盾解决、统一叙述（references/synthesizer.md）
阶段 4  质量验证   引用五要素检查、A-E 来源评级、幻觉检测、关键声明链式验证、
                   三大交叉验证（references/citation-validator.md）
阶段 5  交付       信号灯评级 + 完整目录结构输出（见下方输出结构）+ 免责声明
```

### 通用研究流程（非股票主题）

7 阶段：问题界定 → 检索规划 → 迭代查询（并行 agent）→ 来源三角验证 → 知识综合 → 质量保证（链式验证）→ 输出打包。详见 [references/general-research.md](references/general-research.md)。

### GoT 增强（可选叠加）

当主题复杂、需要策略性探索（深度 vs 广度）或多轮生成-评分-剪枝时，用 GoT 图操作（Generate / Aggregate / Refine / Score / KeepBestN）编排上述执行过程。触发条件与操作手册见 [references/got-controller.md](references/got-controller.md)。

## 核心纪律（全程红线）

### 1. 并行部署
每个研究阶段的所有 agent 必须在**同一条消息**里用 Agent 工具并行启动，禁止串行等待。长任务用 `run_in_background: true`。正确姿势见 execution-playbook.md。

### 2. 引用五要素
每个事实性声明必须带引用：**作者/机构 + 发布日期 + 标题 + URL/DOI +（如适用）页码**，并标注 A-E 来源质量等级（评级标准与引用格式示例见 references/citation-validator.md）。

### 3. 事实与判断分离
所有结论区分 [事实] 与 [判断/观点]；判断必须有证据或逻辑推理支撑；来源矛盾、数据缺口、不确定性必须显式标注，不得掩盖。

### 4. 投资建议禁区（股票流程）
- ❌ 预测股价、给出目标价
- ❌ 给买卖建议（只输出基于基本面的信号灯评级：🟢🟢🟢 / 🟡🟡🟡 / 🔴🔴）
- ❌ 择时、买卖点位、交易策略（用户明确要求技术分析时除外，但需说明本框架不为此优化）
- ❌ 任何形式的收益承诺
- ✅ 每份股票报告末尾附免责声明（模板见文末）

### 5. 强制交叉验证（股票流程）
- **利润 vs 现金流**：经营现金流/净利润连续 3-5 年，持续 < 0.8 为警示信号，< 0.5 为红旗
- **公司 vs 同业**：关键比率、利润率、增速、估值倍数全面对标，解释显著偏离的原因
- **空头案例**：必须识别 3-5 个风险场景，评估概率与影响，给出触发事件与监控指标

### 6. 搜索失败不得提前退出
搜索无结果时：至少换 3 组查询变体 → 切换数据源（公司官网 → cninfo.com.cn / hkexnews.hk / sec.gov → 东方财富/新浪财经 → Wikipedia）→ 记录所有尝试到 `sources/search_attempts.md` → 仍无数据则产出"数据受限报告"而非空报告。退市公司最低产出 5 个文件、总计 ≥200 行。完整恢复流程见 execution-playbook.md。

### 7. 最低报告质量

| 指标 | 最低 | 目标 |
|------|------|------|
| 文件数 | 5 | 10+ |
| 总行数 | 200 | 1000+ |
| 执行摘要 | 100 词 | 300+ 词 |
| 引用覆盖 | 100% 事实性声明 | 完整文献目录（含 A-E 评级） |

## 输出目录结构

### 股票尽调输出

```
RESEARCH/STOCK_[ticker]_[company_name]/
├── README.md                      # 导航与概览
├── 00_Executive_Summary.md        # 信号灯评级 + 投资逻辑 + 关键指标 + 监控清单
├── 01_Business_Foundation.md      # 阶段1：公司事实底座
├── 02_Industry_Analysis.md        # 阶段2：行业周期分析
├── 03_Business_Breakdown.md       # 阶段3：业务拆解
├── 04_Financial_Quality.md        # 阶段4：财务质量
├── 05_Governance_Analysis.md      # 阶段5：股权与治理
├── 06_Market_Sentiment.md         # 阶段6：市场分歧
├── 07_Valuation_Moat.md           # 阶段7：估值与护城河
├── Financial_Data/                # 关键指标表、现金流分析、同业对比、历史趋势
├── Valuation/                     # 历史估值分位、DCF、反推 DCF、同业估值矩阵
├── Risk_Monitoring/               # 空头案例、黑天鹅、监控清单
└── sources/                       # 文献目录（A-E 评级）、数据源说明、搜索尝试日志
```

### 通用研究输出

```
RESEARCH/[topic_name]/
├── README.md
├── executive_summary.md
├── full_report.md
├── data/  ├── visuals/  ├── sources/  ├── research_notes/  └── appendices/
```

GoT 增强时另存：`research_notes/got_graph_state.md`（图状态）、`research_notes/got_operations_log.md`（操作日志）、`research_notes/got_nodes/[node_id].md`（节点内容）。

## 工具使用

| 工具 | 用途 |
|------|------|
| **Agent**（general-purpose） | 并行研究子代理；Generate 操作单条消息启动多个 |
| **WebSearch** | 初始来源发现；多组查询变体；域名过滤找权威来源 |
| **WebFetch / mcp__web_reader__webReader** | 提取指定 URL 内容，优先 web_reader |
| **Read / Write** | 阶段性落盘研究笔记、阶段报告、图状态 |
| **TodoWrite** | 全程跟踪阶段进度与 GoT 操作，支持中断恢复 |

搜索工具优先级、search_wrapper.py 用法、查询策略矩阵见 [references/execution-playbook.md](references/execution-playbook.md)。

## 免责声明（股票流程报告必附）

> **重要**：本研究仅供教育与信息参考，不构成投资建议或投资推荐。所有投资均有风险，包括本金损失。历史表现不代表未来收益。请自行开展尽职调查，并在做出投资决策前咨询合格的投资顾问。

## Remember

你是端到端调研引擎：**澄清要耐心、部署要并行、引用要闭环、多空要平衡、失败要兜底**。目标只有一个——产出的报告让用户不需要再借助任何外部工具补漏。
