# 通用主题 7 阶段深度研究流程

整合原 `research-executor`。适用于非投资类开放主题（行业、技术、政策、市场等）。股票尽调请改用 [stock-research-phases.md](stock-research-phases.md)。

## 7 阶段流程

### Phase 1：问题界定
验证结构化提示词（来自阶段 0 的通用模板）完整性；关键信息缺失则追问。

### Phase 2：检索规划
1. 将主问题按 SPECIFIC QUESTIONS 拆解为 3-7 个子主题
2. 为每个子主题生成具体搜索查询（含变体）
3. 按 CONSTRAINTS 确定数据源
4. 制定执行计划并呈现给用户确认

### Phase 3：迭代查询（多智能体执行）
单条消息并行启动：
- **Web 研究 agent（3-5 个）**：时效信息、趋势、新闻、行业报告
- **学术/技术 agent（1-2 个）**：论文、技术规格、方法论
- **交叉验证 agent（1 个）**：事实核查与验证

### Phase 4：来源三角验证
跨来源比对发现、验证声明。来源质量评级（A-E，标准见 [citation-validator.md](citation-validator.md)）：A=同行评审/系统综述/监管申报；B=队列研究/权威机构分析/行业协会；C=专家观点/公司白皮书/主流新闻；D=预印本/博客/众包内容；E=匿名内容/明显偏见/失效链接。

### Phase 5：知识综合
结构化撰写研究章节，**每条声明带内联引用**（作者/机构、日期、标题、URL/DOI、页码）。综合方法见 [synthesizer.md](synthesizer.md)。

### Phase 6：质量保证（链式验证）
1. 生成初始发现
2. 为每条关键声明构造验证问题
3. 用 WebSearch 独立检索证据
4. 将验证结果与原始发现交叉比对
完整流程见 [citation-validator.md](citation-validator.md)。

### Phase 7：输出打包
```
[output_directory]/[topic_name]/
├── README.md            # 导航
├── executive_summary.md # 执行摘要
├── full_report.md       # 完整报告
├── data/                # 数据文件
├── visuals/             # 图表说明
├── sources/             # 来源清单
├── research_notes/      # 研究笔记（含 GoT 图状态，如启用）
└── appendices/          # 附录
```

## 成功标准

- [ ] 100% 声明有可验证引用
- [ ] 关键发现有多来源支撑
- [ ] 矛盾被承认并解释
- [ ] 输出符合指定格式
- [ ] 研究不越界（时间、地域、来源类型约束）

## 与 GoT 的集成点

- Phase 2：Generate 拆分子主题
- Phase 3：Generate + Score 多智能体部署
- Phase 4：Aggregate 合并发现
- Phase 5：Aggregate + Refine 综合
- Phase 6：Score + Refine 质量保证

操作细节见 [got-controller.md](got-controller.md)。
