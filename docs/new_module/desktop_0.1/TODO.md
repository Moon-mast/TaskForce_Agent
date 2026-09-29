# desktop_0.1 进度

> 口径:本文件记 desktop_0.1 的里程碑与勾选;跨模块总进度仍以 [docs/dev/TODO.md](../../dev/TODO.md)
> 与 [ROADMAP.md](../../dev/ROADMAP.md) 为准。
> 方案与代码见 [00-总览](00-总览.md) / [01-路径层重构](01-路径层重构.md) / [02-桌面壳](02-桌面壳.md)。

**当前状态**:方案与全部代码已交付(文档 + spec + 构建脚本);**环境被一个路径编码问题阻塞**,
阶段 0.5 的基线还没跑出来。下一步是「移项目到纯 ASCII 路径」,再由你誊写代码。

---

## 阶段 0:环境恢复(阻塞一切,不通过不往下走)

- [x] **0.1 装 uv + `uv sync --extra rag2 --frozen`**
      落地:`winget install --id=astral-sh.uv -e` → uv 0.12.20;venv 建在 `.venv`,
      `import langgraph, docling, pymilvus, psycopg, fastapi` 全部通过。
      注意 uv 装在 `%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_...\uv.exe`,
      **PATH 需重开 shell 才生效**;本机 bash 里要用 `/c/Users/.../uv.exe` 这种 Unix 风格路径。
- [x] **0.2 前端构建** `cd dev/front && npm install && npm run build` → `dev/front/dist/` 已生成
      (223 modules, 4.73s)。顺带解决了风险 #14「dist 不存在」。
- [ ] **0.3 起外部服务** `docker compose up -d`(PG 5433)+ WSL 起 Milvus(19530)
      **当前:两者都没起**(实测 5433 / 19530 均 closed),docker 也不在本机 bash 的 PATH 里。
- [~] **0.4 写 `.env`** 已由 `.env.example` 生成**占位版**(值是模板原文,不是真 key);
      **需要你换成真实 LLM / Embedding / RAG2 key**。
- [ ] **0.5 记录基线** `uv run pytest -q` → **479 绿**
      **当前被阻塞**:见下面的「阻塞项」,`uv run pytest` 连收集都过不了。

### ⛔ 阻塞项:项目路径含中文导致 editable 安装静默失效

**现象**:`uv run pytest -q` → 37 个收集错误,`ModuleNotFoundError: No module named 'api'`。

**根因**(已逐层实测确认):uv/hatchling 把 editable 安装写成
`.venv/Lib/site-packages/_editable_impl_taskforce.pth`,内容是本项目 `src/` 的**绝对路径**,
编码为 **UTF-8**。而 CPython 3.12 的 `site.addpackage` 用
`io.TextIOWrapper(io.open_code(fullname), encoding="locale")` 读它 —— Windows 中文 locale 下
`encoding="locale"` 解析为 **cp936**。于是 `D:\游戏\...` 的 UTF-8 字节被按 GBK 解码成
`D:\娓告\...`(U+6E38 U+620F → U+5A13 U+544A),`os.path.exists` 为假,**该行被静默跳过**。
结果:本地七个包全部 import 不到,**没有任何报错**。

**为什么两个常见绕过都无效**:`-X utf8` 与 `PYTHONUTF8=1` 都不起作用 —— `encoding="locale"`
走的是 `locale.getencoding()`,**不受 UTF-8 模式影响**。`uv run` 也一样无效(实测)。
本机卷还禁用了 8.3 短名生成(`GetShortPathName` 原样返回),短路径绕过也不可用。

**解决(待执行)**:把项目移到**纯 ASCII 路径**。这是根治 —— 移完 `.pth` 内容全 ASCII,
任何编码下都能正确解码,uv run / pytest / CLI / desktop 四个入口都不需要额外设置。

---

## 阶段 1:路径层 + 壳(不冻结)

> 代码已全部写在 [01](01-路径层重构.md) / [02-桌面壳](02-桌面壳.md) 里(完整代码 + 改动点清单),
> **按教学模式纪律由你亲手誊写**。测试与打包脚手架由智能体写。

- [ ] **1.1 `src/settings/appdirs.py`**(你誊写,见 01 篇 §2)
      **代码已预先验证**:把 01 篇的代码原样放到一个同构临时布局里跑了 30 项断言(开发态取值
      逐字一致 / `TASKFORCE_HOME` 覆盖 / 冻结态布局与 spec 的 dest 一一对应 / 播种时只清空
      `LLM_BASE_URL`+`LLM_API_KEY`+`LLM_MODEL` 三项、注释行与 `DATABASE_URL` 保留、
      `# LLM_MODEL=...` 这类注释行不被误清 / 幂等不覆盖用户 `.env` 与自装技能 /
      `BACKEND.md` 用户目录优先),**全部通过**。所以你照抄即可,不必担心照抄出错。
- [ ] **1.2 七组调用点改造**(你誊写,见 01 篇 §3.1~§3.7)
      `config.py` / `memory_ctx.py` / `loader.py` / `skills/loader.py` / `mcp/config.py` /
      `api/main.py` / `model_overrides.py` / `session.py`;含 `session.py` 补 `parents=True`、
      `mcp/config.py` 的 `save` 补 `mkdir`
- [ ] **1.3 stdout 防护**(你誊写,见 01 篇 §3.6)`api/main.py:8`、`cli/repl.py:17`
- [ ] **1.4 `tools/sandbox/client.py` 抽公开 `shutdown()`**(你誊写,见 01 篇 §3.8)
- [ ] **1.5 `src/desktop/` 八个文件**(你誊写,见 02 篇)
- [x] **1.6 `pyproject.toml`** 已改:加 `src/desktop` 第 8 包、`pywebview` + `python-dotenv`
      运行期依赖、`pyinstaller` dev 依赖;并在 packages 注释里记下那个 `.pth` 编码坑。
      **待执行**:`uv lock && uv sync --extra rag2`(移动目录后一起做)
- [ ] **1.7 人工冒烟** `uv run python -m desktop` → 窗口起来 → 发一轮对话 →
      设置页改配置 → 重启仍生效 → 关窗无残留进程

## 阶段 2:PyInstaller onedir 冻结

- [x] **2.1/2.2 spec 与构建脚本** 已写完:`packaging/TaskForce.spec`
      (一个 spec 经 `TF_CONSOLE` / `TF_NAME` 环境变量产两个变体,不写两份)、`packaging/build.ps1`
      (先 npm build → `uv sync --frozen` → 跑两遍 pyinstaller)。
      **注意目录叫 `packaging/` 而不是 `build/`** —— `build/` 在 `.gitignore` 里,spec 会入不了库。
- [ ] **2.3 首次构建**并核对 `_internal/prompts/` 下有 md、`_internal/web/` 下有 index.html
- [ ] **2.4 hiddenimports 迭代**(最可能反复的一步;rag_v01 全是懒 import)
- [ ] **2.5 在干净目录跑产物**,按 02 篇 §13 的验收表 + 故障注入矩阵逐条过

## 阶段 3:文档与登记

- [x] **ADR-0013**(`docs/adr/0013-desktop-shell.md`)
- [x] **`packaging/` 登记**入 `docs/index.md`(含「为何不叫 build/」的理由)
- [x] **`.env.example`** 补 `RAG2_*` 段(缺了它知识库上传/检索直接 `ValueError`)
- [x] **`AGENTS.md` 七包 → 八包**(代码布局树加 `desktop/` 与 `packaging/`、依赖方向加 `desktop`、
      `settings/` 行补 `appdirs`、`uv sync` 注释)+ 就地记下「项目路径必须纯 ASCII」
- [x] **`README.md`** 技术栈表七包 → 八包
- [x] **`docs/troubleshooting/common.md` 第 7 条**:`.pth` 编码坑(现象/根因/绕过为何无效/解决),
      并同步 `troubleshooting/README.md` 索引
- [ ] `docs/design/ARCHITECTURE.md` 补第三个入口(桌面壳)+ 路径层一节
- [ ] `docs/design/DESIGN.md` 第 158 行「hatchling 打包 src 下六个包」也已过期(rag_v01 之后就没更新过)
- [ ] `docs/dev/ROADMAP.md` 模块表 + 契约归属表加 `appdirs` 一行 + §7 登记
- [ ] `README.md` 加「桌面应用」一节(**等阶段 2 真的跑通再写**,现在写等于承诺没验证的东西)
- [ ] `src/agent/studio.py` docstring 里「config.py 按绝对路径锚定项目根」那句要改

---

## 待定(需你决定,未开工)

- **`/health` 的云沙箱计费隐患**:前端 `NavRail.vue` 挂载即调 `useHealthPoll()`,
  而 `/health` 会真的 `Sandbox.create()` —— **光把界面打开就占住一个云沙箱**,
  TTL 3600s 按存活时长计费。桌面应用会长时间开着窗口,把它放大了。
  建议给 `/health` 加 `?sandbox=` 显式开关、前端默认不探。
  代价:属于契约变更,要同步 `dev/front/docs/API-CONTRACT.md` 的 types/api/文档三处。见 ADR-0013 被否方案。
