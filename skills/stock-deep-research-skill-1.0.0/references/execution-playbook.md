# 执行手册：并行部署、搜索工具与失败恢复

整合原 `stock-research-executor/instructions.md` 中的执行规范。适用于股票尽调与通用研究两条流程。

## 1. 并行部署铁律

**❌ 错误（串行）**：
```
启动 Agent 1A... → 等完成 → 启动 Agent 1B... → 等完成...
```

**✅ 正确（并行）**：同一条消息中调用多个 Agent 工具，全部立即并发执行：
```
并行启动阶段 1 的 4 个 agent：
[Agent 1A] 研究核心业务与产品
[Agent 1B] 研究营收利润构成
[Agent 1C] 研究客户与价值链位置
[Agent 1D] 研究 3-5 年战略变化
```

长任务用 `run_in_background: true`；全部完成后统一收集结果。

## 2. Agent 提示词模板（五要素）

每个研究 agent 的提示词必须包含：

**① 明确目标**
```
你是专注于 [公司名]([代码]) [具体侧面] 的研究 agent。
任务：[具体、可度量的目标]
时间范围：[研究哪个时间段]
输出格式：[带引用的结构化摘要]
```

**② 搜索工具指引**
- 若工作目录存在 `search_wrapper.py`（智谱搜索封装）：`python search_wrapper.py "<关键词>" [数量1-50] [域名过滤] [时间范围noLimit/one_day/one_week/one_month/one_year] [内容medium/high]`，返回 JSON（title/link/content/media/publish_date）
  - A 股用中文关键词优先财经媒体；港美股中英结合；同一数据点至少 2-3 来源交叉验证
- 否则用 **WebSearch**（多组查询变体、`site:` 域名过滤如 site:sec.gov / site:cninfo.com.cn）+ **WebFetch / mcp__web_reader__webReader**（提取内容，优先 web_reader）

**③ 研究重点**
```
关键问题: 1. [问题1] 2. [问题2] 3. [问题3]
数据点: [要找的具体数据]
来源优先级: 最权威[...] / 补充[...] / 需验证[...]
```

**④ 输出要求**
```
1. Executive Summary（2-3 段）
2. Detailed Findings（分小节，逐问题回答）
3. Key Data Table（指标、趋势）
4. Source Citations（每条声明: 作者/机构, 日期, 标题, URL + A-E 评级）
5. Confidence（High/Medium/Low 及理由）
6. Contradictions and Gaps（来源分歧、未能确定项）
7. Red Flags（如有）
```

**⑤ 质量标准**
```
- 只做有来源支撑的声明
- 区分 [事实] 与 [观点/分析]
- 显式标注不确定性
- 不用煽动性语言（吹捧/恐慌）
- 引用格式统一: [作者/机构, 标题, 日期, URL]
```

## 3. 搜索策略矩阵

| 公司类型 | 首选策略 | 备选 |
|---------|---------|------|
| A 股（存续） | 中文全称 + 代码 | cninfo.com.cn 申报文件 |
| 港股（存续） | 公司名 + 代码 | HKEX 披露易、招股书 |
| 美股（存续） | Ticker + "stock analysis" | SEC 文件、Yahoo Finance |
| 已退市 | "[名称] 退市 原因" | Wikipedia、破产新闻 |
| 过小/无流动性 | 行业 + 地区对比 | 相似公司分析 |
| 陌生/私营 | 公司官网 | LinkedIn、行业数据库 |

数据源优先级：公司官网投资者关系 → 监管申报（cninfo / hkexnews.hk / sec.gov）→ 金融数据站（东方财富、新浪财经、Yahoo Finance）→ Wikipedia（背景）。

## 4. 搜索失败恢复（禁止提前退出！）

**❌ 错误行为**：搜索空结果 → 生成 20 行 README → 退出。
**✅ 正确行为**：搜索失败 → 备用策略 → 记录尝试 → 产出完整报告。

**Step 1 — 至少 3 组查询变体**：
```bash
python search_wrapper.py "[原查询]"
python search_wrapper.py "[公司全称] 财报 业务"
python search_wrapper.py "[股票代码] 营收 利润 财务数据"
python search_wrapper.py "[Company English Name] stock analysis financial"   # 中国公司试英文
python search_wrapper.py "[行业] [地区] 龙头公司 分析报告"
```
（无 wrapper 时用 WebSearch 执行同等变体。）

**Step 2 — 切换数据源**：按上方矩阵逐级尝试。

**Step 3 — 记录全部尝试** 到 `sources/search_attempts.md`：
```markdown
| Attempt | Query | Results | Status |
|---------|-------|---------|--------|
| 1 | "[查询]" | 0 | FAILED |
| 3 | "[查询]" | 3 | PARTIAL |
```

**Step 4 — 产出"数据受限报告"**：即使数据有限也必须生成完整结构，说明找到什么、找不到什么及原因、搜索尝试统计。

### 已退市公司最低产出（5 文件、总计 ≥200 行）

1. `00_Executive_Summary.md`（150+ 行）：退市日期与原因、公司史概要、分析受限原因、投资教训
2. `01_Company_History.md`（100+ 行）：创立与上市、商业模式、历史表现、失败原因
3. `02_Delisting_Analysis.md`（100+ 行）：退市时间线、原因（财务/监管/主动）、股东后果、现状
4. `03_Financial_History.md`（100+ 行）：可得历史数据、最后财报、退市前趋势
5. `04_Lessons_Learned.md`（50+ 行）：被错过的警示信号、投资者本应关注什么

## 5. 结果收集与矛盾处理

1. **汇总**：每 agent 2 句摘要 + 共识发现（2+ agent 提及）vs 独有发现分开列
2. **解矛盾**：逐条比对来源质量（A-E）与发布日期；高质量更新者优先；无法裁决则并列呈现并解释
3. **成文**：共识先行 → 独有洞察 → 矛盾说明 → 缺口与不确定性（详见 synthesizer.md）

## 6. 进度跟踪

用 TodoWrite 全程记录，支持中断恢复：
```
[ ] 阶段 0：需求澄清与指令生成
[ ] 阶段 1：执行计划确认
[ ] 阶段 2.1-2.8：八个研究阶段（逐个标记）
[ ] 阶段 3：综合汇整
[ ] 阶段 4：引用验证
[ ] 阶段 5：交付与 README
```
GoT 增强时追加：Generate(k) / Score / KeepBestN(n) / Aggregate(k) / Refine(1) 各操作项。

## 7. 常见问题排查

| 问题 | 对策 |
|------|------|
| agent 结果互相矛盾 | 比对来源质量与日期 → 高质量新数据优先 → 双视角并列 + 归因解释 |
| 找不到信息 | 换查询 → 换语言（外资公司）→ 查监管申报而非新闻 → 报告中显式声明局限 |
| 搜索质量差 | 更精确关键词 → `site:` 与 `filetype:pdf` 过滤 → 同义词替换 → 确认公司未更名/换代码 |
| 信息过载 | 只用最权威来源 → 聚焦近 3-5 年（除非需历史背景）→ 聚焦重大信息 → 跨源综合而非罗列 |
