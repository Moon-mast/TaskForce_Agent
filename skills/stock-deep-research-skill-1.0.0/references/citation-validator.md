# 引用验证与质量保证手册

整合原 `citation-validator`。用于：每份报告定稿前、发布/分享前、或审查他人研究时。你是防止错误信息与幻觉的**最后一道防线**。

## 引用五要素（每条引用必须齐备）

1. **作者/机构** — 内容生产者
2. **发布日期** — 至少年份（YYYY）
3. **来源标题** — 报告/文章/文件名
4. **URL/DOI** — 可直接核验的链接
5. **页码** — PDF 与长文档适用

**格式示例**：

学术：
```
(Smith et al., 2023, p. 145)
Smith, J., Johnson, K., & Lee, M. (2023). "Title of Paper." Journal Name, 45(3), 140-156. https://doi.org/10.xxxx/xxxxx
```

行业报告：
```
(Gartner, 2024, "Cloud Computing Forecast")
Gartner. (2024). "Cloud Computing Market Forecast, 2024." Retrieved [日期] from https://www.gartner.com/en/research/xxxxx
```

财务数据（股票流程标准示例）：
```markdown
据 2023 年年报，贵州茅台营收同比增长 18.2% 至 1275 亿元 [贵州茅台股份有限公司, 2023 年年度报告, 2024-04,
https://www.cninfo.com.cn/new/disclosure/detail?stockCode=600519&announcementId=xxx]
```

## 来源质量评级（A-E）

| 等级 | 标准 |
|------|------|
| **A 优秀** | 同行评审期刊（含影响因子）、荟萃分析、RCT、政府监管机构、公司监管申报（年报/10-K/20-F） |
| **B 良好** | 队列研究、临床指南、权威机构分析（Gartner/Forrester）、政府网站、行业协会报告、公司投资者关系材料 |
| **C 可接受** | 专家观点、案例报告、公司白皮书、主流媒体报道、公司新闻稿 |
| **D 弱** | 预印本、会议摘要、无编辑审核的博客、众包内容、社交媒体（需一手源验证） |
| **E 极差** | 匿名内容、明显偏见或未披露利益冲突、过时来源、失效/可疑链接 |

## 七步验证流程

1. **声明检测**：扫描全部事实性声明——统计数字、日期、技术规格、市场数据、业绩声明、引语、因果表述
2. **存在性检查**：每条声明是否有引用
3. **完整性检查**：五要素是否齐备
4. **质量评估**：逐条标注 A-E
5. **准确性验证**：用 WebSearch/WebFetch 找到并核对原始来源
6. **幻觉检测**（红旗）：
   - 事实声明无引用
   - 引用不存在（URL 无效）
   - 引用存在但不支持该声明
   - 数字异常精确却无来源
   - 泛化来源（"据行业报告"）无具体出处
7. **关键声明链式验证**（Chain-of-Verification）：高利害声明（医疗/法律/财务）须找到 2-3 个独立来源；检查来源间共识；矛盾须标注；优先 A-B 级；来源分歧时说明不确定性

## 域特异要求

| 领域 | 要求 |
|------|------|
| 金融/市场数据 | 一手来源（SEC/交易所/公司申报）；注明报告期；区分 GAAP vs 非 GAAP |
| 医疗健康 | 同行评审（A-B 级）；验证 PubMed ID；区分"已证实"与"初步" |
| 法律监管 | 引用一手法律文件；含文号；注明司法辖区 |

## 验证报告模板

```markdown
# Citation Validation Report

## Executive Summary
- **Total Claims Analyzed**: [数]
- **Claims with Citations**: [数] ([%])
- **Complete Citations**: [数] ([%])
- **Accurate Citations**: [数] ([%])
- **Potential Hallucinations**: [数]
- **Overall Quality Score**: [分]/10

## Critical Issues (Immediate Action Required)
[幻觉或严重准确性问题]

## Detailed Findings
[逐声明分析]

## Recommendations
[按优先级排列的修正项]
```

落盘至 `sources/citation_validation_report.md`。

## 质量分解读与成功标准

| 分数 | 解读 |
|------|------|
| 9-10 | 专业研究水准 |
| 7-8 | 多数用途可接受 |
| 5-6 | 需改进 |
| 0-4 | 不可信 |

- [ ] 100% 事实性声明有引用
- [ ] 100% 引用五要素完整
- [ ] 95%+ 引用准确
- [ ] 无未解释的幻觉
- [ ] 平均来源质量 ≥ B
- [ ] 总分 ≥ 8/10

**记住：一条有据可查的声明，价值远超十条无支撑的断言。**
