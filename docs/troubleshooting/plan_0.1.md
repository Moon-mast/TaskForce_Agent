# plan_0.1:Plan-and-Execute — 问题与解决记录

> 时间:2026-09-24 | 真模型冒烟首轮暴露,修复含回归测试(tests/test_tasks.py、tests/test_plan_contract.py、tests/test_plan_confirm.py)。
> 方案与进度:`docs/new_module/plan_0.1/`,ADR-0012。

## 1. 计划推进对账断:子图自造 task_id,后台不打标 → 计划卡死 running

- **现象**:真模型冒烟,s1 完成后进度快照仍是 `▶ s1`(running),后续步不解锁、失败不级联、永不收口;用户追问时 answer 答"计划没有在执行"。
- **根因**:三个子图的 finalize 各自 `uuid.uuid4().hex[:8]` **自造** task_id(`research.py:63`、`executor.py:119`、`retriever.py:66`——单子图直挂 Send 的年代无人消费该字段,坏值长期潜伏);`TaskManager._run` 正常路径拿到 summary 直接入库不打标(仅 except 分支填了)。drain 回来的 `ResultSummary.task_id` 与 submit 分配的 id 对不上,`Plan.mark_by_task` 永远 miss。
- **解决**:`tasks.py _run` 在 try 内对 summary `model_copy(update={"task_id": task_id})`,统一覆盖为 submit 分配的 id;`tests/test_tasks.py::test_run_backfills_task_id` 锁死。
- **关联**:`agent/tasks.py`、`agent/subagents/research.py` 等、`agent/contracts/plan.py`(`mark_by_task`)、plan_0.1 文档 03 §4。
- **教训**:集成测试的 fake(直接拿正确 task_id 喂 `complete()`)会绕过真实回填链路——**fake 要模拟"最脆的接缝",不是理想路径**;契约字段"是否有人消费"决定坏值能潜伏多久。

## 2. `need_clarification` 被映射为 failed → 级联杀掉补救步(语义缺陷)

- **现象**:s1 检索无果(need_clarification)按原映射记 failed,`cascade_skip` 跳过 s2"联网补充核实"——补救路径被计划自己放弃,用户观感即"一步失败全都失败"。
- **根因**:把"缺澄清"与"硬失败"混为一谈。级联的本意是防止下游对**根本没执行**的步空跑;need_clarification 是"执行了但缺输入",conclusion 本身就是缺口说明,下游(联网核实)完全可以接力。
- **解决**:`mark_by_task` 只有 `failed` 记 failed,success/partial/need_clarification 都记 done,digest 带缺口说明;级联语义收窄为"硬失败才级联"。
- **关联**:`agent/contracts/plan.py`(`mark_by_task`)、`tests/test_plan_contract.py`(映射用例)、plan_0.1 文档 01 §1.3。
- **教训**:结果状态 → 计划状态的映射,要按"下游还能不能走"来定,不是按"结果是否完美"。

## 3. 下游步契约拿不到前序产出 → 解锁了也算不出

- **现象**:即使 s3 被解锁,其契约 task 只有静态 detail,"计算差值"步拿不到 s1/s2 查到的数字。
- **根因**:子图只看契约四件套、没有对话历史;"detail 自包含"只保证下游**知道要什么**,不保证**拿到上文**——depends_on 解决了控制流,信息流没管。
- **解决**:`_step_contract` 把 depends_on 步的 `result_digest` 拼进任务文本(`(前序步骤结论:s1 结论:…)`);`tests/test_plan_confirm.py::test_step_contract_carries_prior_digest` 锁死。
- **关联**:`agent/planner.py`(`_step_contract`)、plan_0.1 文档 02 §3.3。
- **教训**:计划式多智能体里,信息流与控制流要分开设计——depends_on 只管控制流,前序结论要显式随契约传递。

## 4. answer 工具循环漏带 tool_calls 的 assistant 消息 → OpenAI 400(既有 bug 被时钟工具炸出)

- **现象**:REPL 问"现在是几点",answer 调 `get_current_time` 后第二次 invoke 即 400:`Messages with role 'tool' must be a response to a preceding message with 'tool_calls'`,整轮崩溃。
- **根因**:`_answer_with_memory` 回填工具结果时只追加 ToolMessage(`[*messages, *tool_msgs]`),漏了 `res` 本身(带 tool_calls 的 AIMessage)。该 bug 自 ADR-0011 引入记忆工具就存在,但 memory_search 极少被真模型调用;"现在是几点"是第一个必调工具的请求,首次走进循环即炸——**新工具不是引入者,是暴露者**。
- **解决**:`messages = [*messages, res, *tool_msgs]`(answer.py:119);`tests/test_memory_inject.py::test_answer_tool_loop_appends_assistant_before_toolmessage` 断言每个 ToolMessage 前紧邻同 tool_call_id 的 AIMessage。
- **关联**:`agent/answer.py`(`_answer_with_memory`)、`tools/tool/clock.py`、ADR-0011。
- **教训**:fake 模型不校验 OpenAI 协议,消息序列错了也能全绿——**协议类约束(消息结构、tool 消息依附关系)要在测试里显式断言序列结构**,不能只断言"某类消息出现过"。这是本模块第二个"fake 绕过真实约束"的盲区(坑 1 的 task_id 同型)。
