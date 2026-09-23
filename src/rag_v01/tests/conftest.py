"""模块 02 单测的公共准备。

新模块是**已安装的顶层包** `rag_v01`(见 pyproject 的 `hatch packages`; 源码目录
`src/rag_v01/`), 所以测试直接 `from rag_v01.parsers import ...` 即可 ——
不需要任何 sys.path 注入, IDE 也能正常解析。

历史: 目录原名 `rag_0.1`(带点号 → 不能作为包路径, 只能靠"顶层模块 + 注入 sys.path"),
2026-09-20 改名为 `rag_v01` 以满足包名规则; 2026-09-22 从 `src/new_module/rag_v01/`
挪到 `src/rag_v01/`, 与其它六个包同深度, 才能进 `hatch packages`(混深度会让
editable 安装的 .pth 失效, 见 07 篇 §4.6.1)。
"""
from __future__ import annotations

from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]        # .../src/rag_v01
PROJECT_ROOT = MODULE_ROOT.parents[1]                    # .../my_pro_3
SAMPLES_DIR = PROJECT_ROOT / "data" / "rag2_samples"


@pytest.fixture(scope="session")
def samples_dir() -> Path:
    """真格式样例语料(data/rag2_samples/, 由智能体维护); 缺失则跳过相关用例。"""
    if not SAMPLES_DIR.is_dir():
        pytest.skip(f"缺少样例语料目录: {SAMPLES_DIR}")
    return SAMPLES_DIR

