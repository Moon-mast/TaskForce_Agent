"""desktop_0.1 预检单测:把各类失败翻译成可读中文结论,且不误判成功。

全部用替身,不连真实 LLM / DB:数据库那层 monkeypatch `ensure_vector_ext` 抛错,
端口那层真开一个本地监听套接字(这是唯一可靠地制造「端口被占」的办法)。

顶部 importorskip 守护理由同 test_appdirs.py:代码落地前整篇 skip,不污染全量基线。
"""
import socket
from types import SimpleNamespace

import psycopg
import pytest
from pydantic import BaseModel, ValidationError

preflight = pytest.importorskip("desktop.preflight", reason="desktop_0.1 尚未落地")


class _FakeSettings:
    """`get_settings` 的替身。

    必须同时提供 `cache_clear` —— 真身是 lru_cache 包装的,而 `_check_config` 会先清缓存。
    替身少了这个方法,测试会以 AttributeError 失败,而不是测出真实行为。
    """

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.clear_count = 0

    def cache_clear(self) -> None:
        self.clear_count += 1

    def __call__(self):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(database_url="postgresql://fake/db")


def _validation_error() -> ValidationError:
    """造一个真的 pydantic ValidationError(而不是它的替身),确保映射逻辑对真类型成立。"""

    class _Probe(BaseModel):
        required_field: int

    try:
        _Probe()
    except ValidationError as exc:
        return exc
    raise AssertionError("应该抛 ValidationError")   # pragma: no cover


def test_check_config_reuses_assert_ready_message(monkeypatch):
    """assert_ready 的中文报错要原样透出,不重写。"""
    fake = _FakeSettings(RuntimeError("配置缺失,请在 .env 中填写以下变量:LLM_API_KEY"))
    monkeypatch.setattr(preflight, "get_settings", fake)

    result = preflight._check_config()

    assert result is not None
    assert result.ok is False
    assert "LLM_API_KEY" in result.detail
    assert result.hint


def test_check_config_maps_validation_error(monkeypatch):
    """.env 缺失/写坏时抛的是 ValidationError(Settings() 先于 assert_ready),要给自己的说明。"""
    monkeypatch.setattr(preflight, "get_settings", _FakeSettings(_validation_error()))

    result = preflight._check_config()

    assert result is not None
    assert "配置" in result.title
    assert result.detail          # 原始报错保留,便于定位


def test_check_config_clears_settings_cache(monkeypatch):
    """回归守卫:不清缓存的话,用户改完 .env 点「重试」拿到的还是旧的失败结果。

    这条如果被改坏,表现为「配置文件明明改了却没用」,而且完全不报错 —— 所以单独立一个用例。
    """
    fake = _FakeSettings()
    monkeypatch.setattr(preflight, "get_settings", fake)

    assert preflight._check_config() is None
    assert fake.clear_count == 1


def test_check_config_passes_when_ok(monkeypatch):
    monkeypatch.setattr(preflight, "get_settings", _FakeSettings())
    assert preflight._check_config() is None


def test_check_port_detects_occupied(monkeypatch):
    """端口被占必须报错而不是顺延:前端状态存在 localStorage,按 origin 隔离。"""
    with socket.socket() as listener:
        listener.bind((preflight.const.HOST, 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        monkeypatch.setattr(preflight.const, "PORT", port)

        result = preflight._check_port()

        assert result is not None
        assert str(port) in result.title
        assert result.hint


def test_check_port_passes_when_free(monkeypatch):
    """先占再放,拿到一个刚空出来的端口号。"""
    with socket.socket() as probe:
        probe.bind((preflight.const.HOST, 0))
        port = probe.getsockname()[1]
    monkeypatch.setattr(preflight.const, "PORT", port)

    result = preflight._check_port()

    assert result is None


def test_check_database_reports_pg_down(monkeypatch):
    """PG 连不上:给 docker compose 的下一步动作,而不是把 psycopg 堆栈丢给用户。"""

    def _boom(_url):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(preflight, "ensure_vector_ext", _boom)

    result = preflight._check_database("postgresql://fake/db")

    assert result is not None
    assert "PostgreSQL" in result.title
    assert "docker compose" in result.hint


def test_check_database_passes(monkeypatch):
    monkeypatch.setattr(preflight, "ensure_vector_ext", lambda _url: None)
    assert preflight._check_database("postgresql://fake/db") is None


def test_run_short_circuits_on_first_failure(monkeypatch):
    """第一个失败即返回,不再往下探(否则 PG 没起时还要白等一次端口探测)。"""
    failed = preflight.Result(ok=False, title="配置未完成")
    monkeypatch.setattr(preflight, "_check_config", lambda: failed)

    def _must_not_run():
        raise AssertionError("配置都没过,不该再去探端口")

    monkeypatch.setattr(preflight, "_check_port", _must_not_run)

    assert preflight.run() is failed


def test_run_returns_ok_when_all_pass(monkeypatch):
    # get_settings 也要替身:run() 会把它的 database_url 当实参求值,真的会读 .env
    monkeypatch.setattr(preflight, "get_settings", _FakeSettings())
    monkeypatch.setattr(preflight, "_check_config", lambda: None)
    monkeypatch.setattr(preflight, "_check_port", lambda: None)
    monkeypatch.setattr(preflight, "_check_database", lambda _url: None)

    assert preflight.run().ok is True
