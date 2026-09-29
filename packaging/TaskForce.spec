# -*- mode: python ; coding: utf-8 -*-
"""TaskForce 桌面应用打包 spec(onedir;windowed 与 console 两个变体共用本文件)。

**不要直接手敲 pyinstaller** —— 走 `packaging/build.ps1`,它会先出前端产物、再按变体开关
跑两遍。手敲容易漏掉 `--workpath` 而把中间产物倒进仓库。

变体开关(环境变量,由 build.ps1 设置):
    TF_CONSOLE=1   → console 变体(带控制台窗口,排障用)
    TF_NAME        → 产物名,默认 TaskForce

选 onedir 而不是 onefile:包体 2-3GB 时 onefile 每次启动都要解压到临时目录,
首启十几秒到一分钟,且体积大时崩溃率更高。
"""
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

SPEC_DIR = Path(SPECPATH).resolve()
PROJECT = SPEC_DIR.parent
SRC = PROJECT / "src"
FRONT_DIST = PROJECT / "dev" / "front" / "dist"

CONSOLE = os.environ.get("TF_CONSOLE") == "1"
NAME = os.environ.get("TF_NAME", "TaskForce")

if not (FRONT_DIST / "index.html").is_file():
    raise SystemExit(
        f"前端产物不存在:{FRONT_DIST}\n"
        f"先跑 cd dev/front && npm install && npm run build(或让 build.ps1 代跑)"
    )

# ── 只读资源 ──────────────────────────────────────────────────────────────
# 每个 dest 都必须与 settings/appdirs.py 里对应 *_dir() 的**冻结态返回值逐字对应**。
# 对不上不会在构建期报错,只会运行时 FileNotFoundError 或白屏 —— 所以两边改动要同步。
datas = [
    # static_dir()      → bundle_root()/"web"
    (str(FRONT_DIST), "web"),
    # prompts_dir()     → bundle_root()/"prompts"
    # (源是目录时,PyInstaller 把**内容**拷进 dest,所以是 _MEIPASS/prompts/*.md 而不是 prompts/prompts/)
    (str(SRC / "prompts"), "prompts"),
    # _bundle_skills_dir() → bundle_root()/"default_skills"
    (str(PROJECT / "skills"), "default_skills"),
    # backend_md_path() 的包内回落位 → bundle_root()/"BACKEND.md"
    (str(PROJECT / "BACKEND.md"), "."),
    # _seed_env() 的模板 → bundle_root()/".env.example"
    (str(PROJECT / ".env.example"), "."),
]

# ── 隐藏导入 ──────────────────────────────────────────────────────────────
# 为什么必须手写:rag_v01 承诺「无 import 副作用:不加载 docling / pymilvus / sentence-transformers」,
# 于是它(以及 tools/、agent/ 的部分)的子模块**全部在函数体内按需 import**,
# 静态依赖图看不见它们。漏掉的后果是最难查的一种:构建成功、启动正常、
# **第一次上传文件或第一次用记忆才 ModuleNotFoundError**。
_OUR_LAZY_PACKAGES = ("rag_v01", "tools", "settings", "agent", "api", "desktop")

hiddenimports = []
for _pkg in _OUR_LAZY_PACKAGES:
    hiddenimports += collect_submodules(
            _pkg,
            # tools/sandbox/test/ 下有一个会连真实 e2b 的脚本,收进来会让构建期去 import 它
            # (实测直接抛 AuthenticationError,整个构建失败)
            filter=lambda mod: ".test" not in mod and not mod.endswith(".test"),
    )

# 第三方里这几处是**动态 import / 插件式加载**,静态分析抓不全,显式列出。
# 这份清单是「按实测补」的:冻结后第一次跑到某个功能报 ModuleNotFoundError 时,
# 把那个模块名加到这里再重建 —— 这是本 spec 唯一需要反复迭代的地方。
hiddenimports += [
    "langchain_openai",
    "langchain_mcp_adapters.client",
    "langgraph.checkpoint.postgres",
    "langgraph.checkpoint.serde.jsonplus",
    "e2b_code_interpreter",
    "pymilvus",
    "dashscope",
]

# 带**数据文件**的第三方库:只收模块不收数据会运行时缺文件。
# jieba 没有 dict.txt 直接不可用;tiktoken 缺 .tiktoken 会去联网下载(断网即崩)。
for _pkg in ("jieba", "tiktoken", "trafilatura", "docling", "docling_core"):
    try:
        datas += collect_data_files(_pkg)
    except Exception as exc:  # 包不在环境里就跳过,别让整次构建失败
        print(f"[spec] 跳过 {_pkg} 的数据文件收集:{exc}")

# ── 排除 ──────────────────────────────────────────────────────────────────
# 只排「确定无人用」的。**不要凭体积猜测去排 pandas / matplotlib 之类** ——
# docling 的版面分析管线可能间接用到,排错的表现是运行时才炸。
excludes = [
    "tkinter",              # GUI 走 pywebview,不用 tk
    "pytest",
    "_pytest",
    "IPython",
    "jupyter",
    "notebook",
    "langgraph_cli",        # pyproject 的运行期依赖,但 src/ 无人 import(只有 `uv run langgraph dev` 用)
    "rag_v01.evaluation",   # ragas 装在独立的 eval/ 环境,主环境根本没有它
    # 注意:cli 包与 rich 不在这里 —— 桌面入口不 import cli,静态依赖图自然带不进它们,
    # 不需要也不该手工去排(排错了将来有人 import 就炸)
]

a = Analysis(
        [str(SRC / "desktop" / "__main__.py")],
        pathex=[str(SRC)],
        binaries=[],
        datas=datas,
        hiddenimports=hiddenimports,
        hookspath=[],
        runtime_hooks=[],
        excludes=excludes,
        noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        # UPX 关闭:**torch 的 DLL 被 UPX 压过之后加载会失败**,而且压缩后的 exe
        # 极易被国产杀软误报。省下的那点体积不值得。
        upx=False,
        console=CONSOLE,
        disable_windowed_traceback=False,
)

coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name=NAME,
)
