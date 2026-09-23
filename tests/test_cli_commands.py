"""CLI 管理命令单测(c4):/kb 与 /memory 退化为纯渲染,打桩点在 rag_v01 facade / memory_ctx。"""

import io
from types import SimpleNamespace

from rich.console import Console

import agent.memory_ctx as mcx
import rag_v01
from cli.commands.knowledge import cmd_kb
from cli.commands.memory import cmd_memory
from cli.context import ReplContext


def _ctx():
    """最小 ReplContext:捕获渲染输出到 StringIO(无色,断言裸文本)。"""
    buf = io.StringIO()
    console = Console(file=buf, width=160, color_system=None, force_terminal=False)
    ctx = ReplContext(
        console=console,
        settings=SimpleNamespace(),
        graph=None,
        checkpointer=None,
        sessions=None,
        usage=None,
    )
    return ctx, buf


def test_cmd_kb_list_renders_table(monkeypatch):
    monkeypatch.setattr(rag_v01, "list_docs", lambda: [
        {"doc_id": "d1", "filename": "a.md", "chunks": 3, "created_at": None},
    ])
    ctx, buf = _ctx()
    cmd_kb("list", ctx)
    out = buf.getvalue()
    assert "a.md" in out and "d1" in out


def test_cmd_kb_list_empty(monkeypatch):
    monkeypatch.setattr(rag_v01, "list_docs", lambda: [])
    ctx, buf = _ctx()
    cmd_kb("list", ctx)
    assert "(知识库为空)" in buf.getvalue()


def test_cmd_kb_delete_delegates_existence_to_facade(monkeypatch):
    """存在性判定在 facade:入口只按布尔渲染(不存在 → 文档不存在)。"""
    deleted: list[str] = []
    monkeypatch.setattr(rag_v01, "delete_doc", lambda did: deleted.append(did) or did == "d1")
    ctx, buf = _ctx()
    cmd_kb("delete d1", ctx)
    assert "已删除" in buf.getvalue() and deleted == ["d1"]
    cmd_kb("delete nope", ctx)
    assert "文档不存在" in buf.getvalue() and deleted == ["d1", "nope"]


def test_cmd_kb_upload_created_and_duplicate(monkeypatch, tmp_path):
    """上传路径引号剥离 + 原文件名/内容透传 facade + created 两种文案。"""
    results = [
        {"ok": True, "error": "", "doc_id": "d9", "created": True, "name": "x.md"},
        {"ok": True, "error": "", "doc_id": "d9", "created": False, "name": "x.md"},
    ]
    seen: dict = {}

    def fake_upload(content, filename):
        seen["content"], seen["filename"] = content, filename
        return results.pop(0)

    monkeypatch.setattr(rag_v01, "upload", fake_upload)
    f = tmp_path / "报告 2024.md"
    f.write_text("内容", encoding="utf-8")

    ctx, buf = _ctx()
    cmd_kb(f'upload "{f}"', ctx)  # 路径包引号防空格,应被剥掉
    assert "已上传" in buf.getvalue()
    assert seen["filename"] == "报告 2024.md"
    assert seen["content"] == "内容".encode()

    cmd_kb(f'upload "{f}"', ctx)
    assert "内容重复" in buf.getvalue()


def test_cmd_kb_upload_failure_prints_reason(monkeypatch, tmp_path):
    """facade 判失败(如 20MB 上限):入口原样渲染 error。"""
    monkeypatch.setattr(
        rag_v01, "upload",
        lambda c, n: {"ok": False, "error": "文件超过 20MB 上限",
                      "doc_id": "", "created": False, "name": n},
    )
    f = tmp_path / "big.md"
    f.write_text("x", encoding="utf-8")
    ctx, buf = _ctx()
    cmd_kb(f"upload {f}", ctx)
    out = buf.getvalue()
    assert "入库失败" in out and "20MB" in out


def test_cmd_kb_upload_missing_file_reports_read_error(tmp_path):
    ctx, buf = _ctx()
    cmd_kb(f"upload {tmp_path / '不存在.md'}", ctx)
    assert "读取失败" in buf.getvalue()


def test_cmd_memory_list_and_delete(monkeypatch):
    """记忆管理双入口同源:渲染 memory_ctx.list_memories/delete_memory 的结果。"""
    monkeypatch.setattr(mcx, "list_memories", lambda: [
        {"key": "k1", "content": "用户偏好中文",
         "source": "explicit", "created_at": "2026-09-23 10:00:00+00:00"},
    ])
    deleted: list[str] = []
    monkeypatch.setattr(mcx, "delete_memory", lambda key: deleted.append(key))

    ctx, buf = _ctx()
    cmd_memory("list", ctx)
    out = buf.getvalue()
    assert "k1" in out and "用户偏好中文" in out

    cmd_memory("delete k1", ctx)
    assert "已删除" in buf.getvalue() and deleted == ["k1"]
