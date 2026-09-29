# 0013 - 桌面应用外壳(pywebview 单窗口 + PyInstaller onedir)

- 状态:✅ 已接受(2026-09-29)
- 关联:ADR-0004(本地部署、skills/MCP 走文件系统)、ADR-0007(轻量执行沙箱)
- 方案与进度:`docs/new_module/desktop_0.1/`

## 背景

TaskForce 目前起步要四条命令(`docker compose up -d` → `npm run build` → `uv run uvicorn` → 浏览器打开),
作为**交付物**不合格:拿到项目的人要装 uv、Python 3.12、Node、Docker。

同时,冻结成单体应用这件事暴露出一层此前被掩盖的**结构性假设**:全仓 11 处运行时路径锚点
都依赖「CWD 是仓库根」或「`__file__` 上溯能到仓库根」,两者在冻结后都不成立。其中三处会
**静默失效不报错**(`memory_ctx.py` 的 `BACKEND.md` 降级为占位串、`loader.py` 的提示词、
`skills/loader.py` 的技能目录),这是本 ADR 要优先解决的问题。

## 决策

1. **壳用 pywebview 单窗口,不引入第二套语言栈**。项目已有「FastAPI 同源托管 dist」形态
   (`api/main.py:36`),壳只需指向 `127.0.0.1:8010`。Electron 要多维护一套 Node 壳,
   Tauri 要装 Rust + MSVC;两者收益(托盘、自动更新)对本项目不抵成本。

2. **引入 `settings/appdirs.py` 作为路径的唯一出处,按 `sys.frozen` 分双态**。
   **关键约束:开发态 `data_root()` 仍等于仓库根** —— 于是全部调用点在开发态的取值与重构前
   **逐字节相同**,479 条后端基线与 185 条前端基线不受影响,改动可逐处 review。
   冻结态才切到 `%LOCALAPPDATA%\TaskForce` 与 `sys._MEIPASS`。
   `TASKFORCE_HOME` 作部署逃生口 + 单测 seam。

3. **`rag_v01` 保持零改动,路径由启动器注入环境变量**。`rag_v01` 的定位是「独立可搬」
   (AGENTS.md),不能反向 `import settings`;它的 `config.py` 明写「不读 .env,载入由入口负责」,
   只认进程环境变量。于是 `RAG2_IMAGE_DIR` / `HF_HOME` 由 `desktop/bootstrap.py` 在
   **import 任何 app 模块之前** `setdefault` 注入。
   **顺序是硬约束**:`settings/config.py` 的 `_ENV_FILE` 是模块级常量,且全项目有 6 个
   `lru_cache`(`get_settings` / `embedding_dim` / `mcp_meta` / `_skills_meta` /
   `load_agents_md` / `rag_v01._cfg`)—— 任何一个在错误环境下被调过一次,进程内就永久锁死。

4. **提示词改读磁盘路径,不再依赖 `importlib.resources`**。后者的可用性取决于包解析与 zip 语义
   (包能否 import、md 是否落在真实目录而非 `base_library.zip`、`__init__` 与 md 是否同目录),
   冻结下任一不满足即 `FileNotFoundError`;而 `load_prompt` **没有 try/except,缺 md 直接崩图**。
   换成「文件存在性」判断后不确定性归零。附带好处:`prompts_dir()` 将来可改成「用户目录优先」
   以支持自定义提示词(本期不做)。

5. **端口固定 8010,占用即明确报错,不做顺延**。前端把会话 id 等状态存 `localStorage`,
   而 **localStorage 按 origin 隔离** —— 端口一变等于换了个应用,每次启动丢状态。
   宁可报错也不静默换端口。

6. **就绪判定用 `GET /`,不用 `/health`**。`/health` 会真的调 `sandbox_health()` →
   `Sandbox.create()`,耗时数秒且**开始计费**(云端实例 TTL 3600s)。轮询根路径一次同时验证
   「uvicorn 已监听」与「前端产物已挂载」;dist 没收集进包时表现为一直 404,直接指向根因。

7. **退出采用分层兜底,以 `os._exit` 收口**。`agent/tasks.py` 的 `ThreadPoolExecutor` worker
   **非 daemon**,而 CPython 3.9+ 在解释器退出时经 `threading._register_atexit` join 所有 worker;
   `tasks.py` 的 `_run()` 又是同步 `graph.invoke` **不可取消**,子图每轮可能跑最长 60s 的沙箱调用。
   `cancel_futures=True` **只取消尚未启动的任务**,所以「优雅关闭」单独用不够。
   **代价:硬兜底会跳过 `atexit`,而云沙箱回收正靠 `atexit`** —— 因此必须先把
   `tools/sandbox/client.py` 的回收抽成公开的 `shutdown()` 显式调用,否则漏回收并继续计费到 TTL。

8. **PyInstaller 用 onedir,产 windowed + console 两个变体**。onedir 避免 onefile 每次启动
   解压 GB 级内容;两个变体是因为 windowed 下 `sys.stdout` 是 `None`,**没有任何输出**,
   console 变体是唯一排障入口(也是 `--tf-smoke` 自检要看输出的那个)。

9. **`src/desktop/` 作为第 8 个包**。沿用「一个入口一个包」的既有逻辑
   (`cli/` = 终端入口、`api/` = HTTP 入口、`studio.py` = 调试入口)。它只做壳:
   图构建仍走 `build_graph()`,单轮仍走 `run_turn()`(ADR-0009 与双入口红线不变)。

## 被否方案

- **启动时 `os.chdir(data_root)` 解决全部相对路径**。它能一次性解决 `BACKEND.md`、
  `.taskforce/`、`RAG2_IMAGE_DIR` 这些**裸 CWD 相对**的锚点,看似最省事。否决理由:
  它解决不了 `.env`、`skills/`、`mcp_config.json`、dist 这些 **`__file__` 锚定**的锚点,
  两套机制并存比一套更糟;且 chdir 会改变用户传入相对路径(如 `rag_v01 ingest <路径>`)
  的解析基准,副作用外溢。
- **内嵌 PostgreSQL + 砍掉 Milvus 做到真正开箱即用**。Milvus 没有 Windows 版 Lite,
  知识库必然降级;且嵌入式 PG 与现有 `database_url` 配置/迁移逻辑要重新设计。
  本 ADR 选择「保持外部依赖 + 启动器把故障翻译成中文提示」。
- **端口占用则顺延**。见决策 5。
- **onefile 单 exe**。见决策 8。
- **给 `/health` 加 `?sandbox=` 开关、前端默认不探沙箱**。这是决策 6 顺带发现的现存计费隐患:
  前端 `NavRail.vue` 挂载即调 `useHealthPoll()`,**光把界面打开就会占住一个云沙箱**。
  桌面应用会长时间开着窗口,把它放大了。**本 ADR 未采纳,记为待定**:它属于契约变更,
  要按纪律同步 `dev/front/docs/API-CONTRACT.md` 三处,超出「打包」范围,留待单独决定。
- **用 8.3 短路径绕过中文路径问题**。本机卷禁用了短名生成(`GetShortPathName` 原样返回),
  不可用;根治办法是项目放纯 ASCII 路径,见 `docs/troubleshooting/common.md`。

## 后果

- `src/settings/appdirs.py` 新增(路径唯一出处);`src/desktop/` 新增 8 个文件。
- `pyproject.toml`:`packages` 七包 → **八包**;新增 `pywebview` / `python-dotenv` 运行期依赖
  与 `pyinstaller` dev 依赖。
- 七组调用点改造(`config.py` / `memory_ctx.py` / `loader.py` / `skills/loader.py` /
  `mcp/config.py` / `api/main.py` / `model_overrides.py` / `session.py`),
  外加 `session.py` 的 `mkdir` 补 `parents=True`、`mcp/config.py` 的 `save` 补 `mkdir`。
- `tools/sandbox/client.py` 新增公开 `shutdown()`。
- 新增 `packaging/`(spec + 构建脚本)。**注意不叫 `build/`** —— 那在 `.gitignore` 里。
- 文档同步:`AGENTS.md`(七包 → 八包、命令区)、`docs/design/ARCHITECTURE.md`、
  `docs/dev/ROADMAP.md`(模块表 + 契约归属表 + §7 登记)、`docs/index.md`、`README.md`、
  `.env.example`(补 `RAG2_*` 段)。
- **已知取舍**:硬兜底退出会丢弃正在执行的子图任务。若关闭时正好有计划在推进,
  该步骤会停在 running,下次启动不自动续跑。与 REPL 里 Ctrl+C 的后果一致,本期不引入断点续跑。
- **未解风险**:PyInstaller 全量冻结 docling + torch 需要实测迭代 `hiddenimports`,
  极可能反复几轮。若长期不通过,退回选项是排除 docling(知识库上传降级)。
