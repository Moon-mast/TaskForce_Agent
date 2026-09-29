"""desktop_0.1 路径层单测:appdirs 的双态解析、首启播种、以及「开发态零变化」。

不依赖真实冻结:冻结态用 monkeypatch 造假 `sys.frozen` / `sys._MEIPASS` / `LOCALAPPDATA`,
可写目录用 `TASKFORCE_HOME` 指到 tmp_path。

用例文件顶部用 importorskip 守护:`appdirs.py` 属 desktop_0.1 的誊写清单,落地前本文件整篇
skip —— 这样它不会在代码到手之前把全量基线弄花(收集期 ERROR 会让「479 绿」这个口径失效)。
"""
import sys
from pathlib import Path

import pytest

appdirs = pytest.importorskip("settings.appdirs", reason="desktop_0.1 尚未落地 appdirs.py")

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dev_mode_keeps_repo_root():
    """开发态 data_root() 必须仍是仓库根 —— 这是「基线零回归」的全部依据。"""
    assert appdirs.is_frozen() is False
    assert appdirs.data_root() == REPO_ROOT
    assert appdirs.bundle_root() == REPO_ROOT


def test_dev_mode_paths_match_pre_refactor():
    """七组调用点在开发态的位置必须与重构前逐字一致。"""
    assert appdirs.env_file() == REPO_ROOT / ".env"
    assert appdirs.state_dir() == REPO_ROOT / ".taskforce"
    assert appdirs.skills_dir() == REPO_ROOT / "skills"
    assert appdirs.mcp_config_path() == REPO_ROOT / "mcp_config.json"
    assert appdirs.static_dir() == REPO_ROOT / "dev" / "front" / "dist"
    assert appdirs.backend_md_path() == REPO_ROOT / "BACKEND.md"
    # prompts 在开发态位于 src/ 下(与其它包同深度),不在仓库根
    assert appdirs.prompts_dir() == REPO_ROOT / "src" / "prompts"


def test_home_env_overrides_both_modes(monkeypatch, tmp_path):
    """TASKFORCE_HOME 优先于开发态与冻结态的默认值(部署逃生口 + 单测 seam)。"""
    monkeypatch.setenv(appdirs.HOME_ENV, str(tmp_path))
    assert appdirs.data_root() == tmp_path
    assert appdirs.env_file() == tmp_path / ".env"
    assert appdirs.state_dir() == tmp_path / ".taskforce"

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    assert appdirs.data_root() == tmp_path


def test_frozen_mode_layout(monkeypatch, tmp_path):
    """冻结态:可写数据落 LOCALAPPDATA,只读资源落 _MEIPASS,dest 与 spec 的 datas 对应。"""
    monkeypatch.delenv(appdirs.HOME_ENV, raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))

    data = tmp_path / "local" / appdirs.APP_NAME
    bundle = tmp_path / "bundle"
    assert appdirs.is_frozen() is True
    assert appdirs.data_root() == data
    assert appdirs.bundle_root() == bundle
    # 这三个 dest 必须与 packaging/TaskForce.spec 的 datas 一一对应,改一处就要改另一处
    assert appdirs.static_dir() == bundle / "web"
    assert appdirs.prompts_dir() == bundle / "prompts"
    assert appdirs.backend_md_path() == bundle / "BACKEND.md"
    assert appdirs.skills_dir() == data / "skills"


def test_backend_md_prefers_user_copy(monkeypatch, tmp_path):
    """BACKEND.md 用户目录优先(可自定义),否则回落包内。"""
    monkeypatch.setenv(appdirs.HOME_ENV, str(tmp_path))
    assert appdirs.backend_md_path() == REPO_ROOT / "BACKEND.md"   # 用户目录还没有 → 回落
    custom = tmp_path / "BACKEND.md"
    custom.write_text("自定义背景", encoding="utf-8")
    assert appdirs.backend_md_path() == custom


def test_ensure_data_layout_creates_dirs_and_seeds(monkeypatch, tmp_path):
    """首启:建三个可写目录,并从包内资源播种 skills/ 与 .env。"""
    monkeypatch.setenv(appdirs.HOME_ENV, str(tmp_path))
    appdirs.ensure_data_layout()

    assert appdirs.state_dir().is_dir()
    assert appdirs.skills_dir().is_dir()
    # skills 播种:仓库里已有 hello-world,应被复制过来
    assert (appdirs.skills_dir() / "hello-world" / "SKILL.md").is_file()
    assert appdirs.env_file().is_file()


def test_seeded_env_blanks_llm_keys(monkeypatch, tmp_path):
    """播种的 .env 必须把 LLM 三项清空。

    清空才能落到 assert_ready 的现成中文报错上;留着模板里的 "sk-your-llm-api-key"
    这种假值会「启动成功、第一次调用模型才报 401」,对首启体验是倒退。
    """
    monkeypatch.setenv(appdirs.HOME_ENV, str(tmp_path))
    appdirs.ensure_data_layout()

    text = appdirs.env_file().read_text(encoding="utf-8")
    for key in appdirs.BLANK_ON_SEED:
        assert f"{key}=\n" in text
    # 模板里的注释行要原样保留(用户靠它知道每个变量填什么)
    assert "# ── 主模型(LLM)" in text
    # DATABASE_URL 照抄模板:它已是 docker-compose 的默认值,首启不该要求用户填
    assert "DATABASE_URL=postgresql://" in text


def test_ensure_data_layout_never_overwrites(monkeypatch, tmp_path):
    """幂等且不覆盖:用户的 .env 与自装技能必须保住。"""
    monkeypatch.setenv(appdirs.HOME_ENV, str(tmp_path))
    custom_env = tmp_path / ".env"
    custom_env.write_text("LLM_MODEL=我的模型\n", encoding="utf-8")
    own_skill = tmp_path / "skills" / "my-skill"
    own_skill.mkdir(parents=True)
    (own_skill / "SKILL.md").write_text("---\nname: my-skill\n---\n", encoding="utf-8")

    appdirs.ensure_data_layout()

    assert custom_env.read_text(encoding="utf-8") == "LLM_MODEL=我的模型\n"
    assert (own_skill / "SKILL.md").read_text(encoding="utf-8").startswith("---")
