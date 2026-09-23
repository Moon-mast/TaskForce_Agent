"""模块 04 embed 层单测(公共件 / `_ApiBackend` / `_LocalBackend` / 门面与 `embed_chunks`)。

**全程不联网**: 第 1 段用假实现替换 `MultiModalEmbedding.call`, 第 2 段整段换掉
`get_embedder` 与 `_cfg`(以及假的 `sentence_transformers` 模块), 真的 HTTP 请求一次也不发。

为什么这一层值得细测: 分批次数、顺序对齐、图片降级、维度自检这几件事一旦错, 现象都是
"检索结果莫名变差"而不是报错(04 篇坑位 10 / 坑位 5 / 坑位 13), 所以断言必须落到
"第 i 条内容拿到了第 i 个向量"这种逐条映射上, 不能只看"返回了 N 个向量"。

维度统一用小值(8): `cfg.embed_dim` 本来就是配置项, 用小维度让断言好读, 与真 1024 无关。
"""
from __future__ import annotations

import sys
import types

import pytest

# dashscope 只在 api 模式需要(rag2 extra); 没装就整文件跳过, 不让单测卡在依赖上
dashscope = pytest.importorskip("dashscope")

from rag_v01 import embed as embed_mod  # noqa: E402  (importorskip 之后才能 import 被测模块)
from rag_v01.config import Config  # noqa: E402
from rag_v01.contracts import TYPE_IMAGE, TYPE_TABLE, TYPE_TEXT, ChildChunk  # noqa: E402
from rag_v01.embed import (  # noqa: E402
    _ApiBackend,
    _check_dim,
    _l2_normalize,
    _LocalBackend,
    _to_data_uri,
    embed_chunks,
    embed_documents,
    embed_images,
    embed_query,
    embedding_dim,
)

DIM = 8
TEXTS_45 = [f"文本{i}" for i in range(45)]
IMG_12 = [f"图{i}" for i in range(12)]


# ---------------------------------------------------------------- 造数据与假模型


class FakeMM:
    """假的 `MultiModalEmbedding`: 记录每次入参, 按内容给一条**可区分**的向量。

    向量取 `[tag, 1, 1, ...]`, tag 是"该内容首次出现时的序号" —— 相同内容同向量、不同内容必
    不同向量。这样"顺序有没有错"就能逐条验, 而不是靠"看起来像对"。
    """

    def __init__(self, *, dim: int = DIM, status: int = 200, reverse_index: bool = False) -> None:
        self.calls: list[dict] = []
        self.dim = dim
        self.status = status
        self.reverse_index = reverse_index
        self._tags: dict[str, int] = {}

    def probe(self, key: str) -> list[float]:
        """内容 → 未归一化的原始向量(测试侧算期望值用, 与 `call` 内部同一套规则)。"""
        if key not in self._tags:
            self._tags[key] = len(self._tags) + 1
        return [float(self._tags[key])] + [1.0] * (self.dim - 1)

    def call(self, **kwargs) -> object:
        self.calls.append(kwargs)
        if self.status != 200:
            return _resp(status=self.status, output=None, code="InvalidApiKey", message="key 无效")
        items = [
            {"index": i, "embedding": self.probe(_key_of(content))}
            for i, content in enumerate(kwargs["input"])
        ]
        if self.reverse_index:
            items = list(reversed(items))       # 模拟服务端把顺序打乱
        return _resp(status=200, output={"embeddings": items})

    def sent(self) -> list[list[dict]]:
        """每次请求的 contents 列表(按调用顺序)。"""
        return [c["input"] for c in self.calls]

    def batch_sizes(self) -> list[int]:
        return [len(items) for items in self.sent()]


def _key_of(content: dict) -> str:
    """一条 content 的身份: 文本取文本, 图片取 data URI 前缀之后的字节。"""
    if "text" in content:
        return content["text"]
    return content["image"].split(",", 1)[1]


def _resp(*, status: int, output: object, code: str = "", message: str = "") -> object:
    """伪造 DashScopeAPIResponse: 六个字段与 SDK 一一对应(_call 只读这几个)。"""
    return types.SimpleNamespace(
        status_code=status, code=code, message=message, request_id="req-fake", output=output
    )


def api_cfg(**kw) -> Config:
    base = {
        "embed_mode": "api",
        "embed_api_key": "fake-key",
        "embed_dim": DIM,
        "embed_batch": 20,
        "embed_img_batch": 5,
    }
    base.update(kw)
    return Config(**base)


@pytest.fixture
def fake_mm(monkeypatch):
    """装假模型: `mm = fake_mm(reverse_index=True)`。patch 的是**类属性**, 与构造顺序无关。"""

    def _install(**kw) -> FakeMM:
        fake = FakeMM(**kw)
        monkeypatch.setattr(dashscope.MultiModalEmbedding, "call", fake.call)
        return fake

    return _install


@pytest.fixture
def pngs(tmp_path):
    """12 张内容互不相同的假 PNG(内容不同 → 向量可区分)。"""
    paths = []
    for i in range(len(IMG_12)):
        p = tmp_path / f"img{i}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([i]) * (i + 1))
        paths.append(str(p))
    return paths


# ---------------------------------------------------------------- 公共件


def test_l2_normalize_makes_unit_vector():
    vec = _l2_normalize([3.0, 4.0])
    assert abs(sum(x * x for x in vec) - 1.0) < 1e-9
    assert vec == pytest.approx([0.6, 0.8])


def test_l2_normalize_zero_vector_passthrough():
    """零向量必须原样返回: 除以 0 会得到 nan, nan 进库不报错、只污染检索。"""
    assert _l2_normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_to_data_uri_png_and_jpg_mime(tmp_path):
    png = tmp_path / "a.png"
    png.write_bytes(b"\x89PNG")
    assert _to_data_uri(str(png)) == "data:image/png;base64,iVBORw=="

    jpg = tmp_path / "b.JPG"                     # 大写后缀也要认(02 篇 RAG2_IMAGE_FORMAT)
    jpg.write_bytes(b"\xff\xd8")
    assert _to_data_uri(str(jpg)).startswith("data:image/jpeg;base64,")


def test_to_data_uri_rejects_unsupported_suffix(tmp_path):
    bmp = tmp_path / "c.bmp"
    bmp.write_bytes(b"BM")
    with pytest.raises(ValueError, match="不支持的后缀|不支持的图片后缀"):
        _to_data_uri(str(bmp))


def test_to_data_uri_rejects_oversize(tmp_path, monkeypatch):
    """超限判断用改小上限来测, 不真写 11MB 文件。"""
    from rag_v01 import embed

    monkeypatch.setattr(embed, "_MAX_IMAGE_BYTES", 8)
    big = tmp_path / "big.png"
    big.write_bytes(b"\x89PNG" + b"x" * 16)
    with pytest.raises(ValueError, match="超过单图"):
        _to_data_uri(str(big))


def test_to_data_uri_missing_file_raises(tmp_path):
    """文件不存在要抛, 不能悄悄返回 None —— 降级判断由调用方(第 2 步)用 is_file 先做。"""
    with pytest.raises(FileNotFoundError):
        _to_data_uri(str(tmp_path / "nope.png"))


def test_check_dim_names_the_env_var():
    with pytest.raises(ValueError, match="RAG2_EMBED_DIM"):
        _check_dim([[0.0] * 768], 1024, what="文本 embedding")


def test_check_dim_accepts_matching_and_empty():
    assert _check_dim([], 1024, what="文本 embedding") == []
    vec = [0.0] * DIM
    assert _check_dim([vec], DIM, what="文本 embedding") == [vec]


# ---------------------------------------------------------------- API 后端构造


def test_api_backend_requires_api_key():
    with pytest.raises(ValueError, match="RAG2_EMBED_API_KEY"):
        _ApiBackend(api_cfg(embed_api_key=""))


def test_api_backend_rejects_non_positive_batch():
    """batch=0 会让 range(..., 0) 抛一句看不懂的报错, 在构造处就拦掉。"""
    with pytest.raises(ValueError, match="必须是正整数"):
        _ApiBackend(api_cfg(embed_batch=0))


# ---------------------------------------------------------------- 文本路


def test_encode_text_batches_20_and_passes_dimension(fake_mm):
    fake = fake_mm()
    backend = _ApiBackend(api_cfg())

    out = backend.encode(TEXTS_45)

    assert fake.batch_sizes() == [20, 20, 5]              # 45 条 → 3 次请求
    assert len(out) == 45
    # dimension 必须是**顶层参数**: 单请求里带 parameters= 会被服务端静默忽略、返回默认 2560
    assert [c["dimension"] for c in fake.calls] == [DIM, DIM, DIM]
    assert [c["model"] for c in fake.calls] == ["qwen3-vl-embedding"] * 3
    assert all("text" in item and "image" not in item for items in fake.sent() for item in items)


def test_encode_text_order_is_one_to_one(fake_mm):
    """逐条映射: 第 i 条文本必须拿到第 i 条文本自己的向量(错位=向量写错块, 静默)。"""
    fake = fake_mm()
    out = _ApiBackend(api_cfg()).encode(TEXTS_45)

    expected = [_l2_normalize(fake.probe(t)) for t in TEXTS_45]
    assert out == expected
    assert len({tuple(v) for v in expected}) == 45        # 假向量两两不同, 上一条断言才有意义


def test_encode_text_resorts_by_index(fake_mm):
    """服务端把 index 打乱返回时, 仍要按 index 归位。"""
    fake = fake_mm(reverse_index=True)
    out = _ApiBackend(api_cfg()).encode(TEXTS_45)

    assert out == [_l2_normalize(fake.probe(t)) for t in TEXTS_45]


def test_encode_empty_texts_makes_no_request(fake_mm):
    fake = fake_mm()
    assert _ApiBackend(api_cfg()).encode([]) == []
    assert fake.calls == []


def test_encode_raises_readable_error_on_dim_mismatch(fake_mm):
    """真配置 1024 维、模型返回 8 维: 要在 embed 层就炸, 且点出改哪个变量。"""
    fake_mm(dim=8)
    with pytest.raises(ValueError, match="RAG2_EMBED_DIM=1024"):
        _ApiBackend(api_cfg(embed_dim=1024)).encode(["一段中文"])


def test_encode_raises_runtime_error_on_api_failure(fake_mm):
    """SDK 不抛异常, 只把错误放进响应对象: 不显式抛, 上层会崩在 output 为 None 上。"""
    fake_mm(status=401)
    with pytest.raises(RuntimeError, match="InvalidApiKey"):
        _ApiBackend(api_cfg()).encode(["一段中文"])


# ---------------------------------------------------------------- 图片路


def test_encode_images_batches_5_and_sends_base64(fake_mm, pngs):
    fake = fake_mm()
    out = _ApiBackend(api_cfg()).encode_images(pngs)

    assert fake.batch_sizes() == [5, 5, 2]                # 12 张 → 3 次请求(服务端限图片 ≤5/请求)
    assert len(out) == 12
    sent = [item for items in fake.sent() for item in items]
    assert all(item["image"].startswith("data:image/png;base64,") for item in sent)
    assert all("text" not in item for item in sent)
    assert out == [_l2_normalize(fake.probe(_key_of(item))) for item in sent]


def test_encode_images_resorts_by_index(fake_mm, pngs):
    fake = fake_mm(reverse_index=True)
    out = _ApiBackend(api_cfg()).encode_images(pngs)

    sent = [item for items in fake.sent() for item in items]
    assert out == [_l2_normalize(fake.probe(_key_of(item))) for item in sent]


# ---------------------------------------------------------------- 第 2 段: 门面与 embed_chunks


class FakeBackend:
    """假后端(替掉 `get_embedder`): 记录每次调用的入参, 按内容给可区分向量。

    与 `FakeMM` 同一套路但更靠上 —— 这一段的被测对象是"分组与拼回"的编排逻辑, 不该再牵
    批处理: 后端返回什么就照单收下, 因此 `drop_last=True` 能造出"后端少返向量"的坏情形。
    """

    def __init__(
        self, *, supports_images: bool = True, dim: int = DIM, drop_last: bool = False
    ) -> None:
        self.supports_images = supports_images
        self.dim = dim
        self.drop_last = drop_last
        self.text_calls: list[list[str]] = []
        self.image_calls: list[list[str]] = []
        self._tags: dict[str, int] = {}

    def vec(self, key: str) -> list[float]:
        """可区分向量: 首元素是内容序号(首次出现的顺序), 其余补 1 —— 未归一化, 便于算期望。"""
        if key not in self._tags:
            self._tags[key] = len(self._tags) + 1
        return [float(self._tags[key])] + [1.0] * (self.dim - 1)

    def _emit(self, keys: list[str]) -> list[list[float]]:
        vectors = [_l2_normalize(self.vec(k)) for k in keys]
        return vectors[:-1] if self.drop_last and vectors else vectors

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.text_calls.append(list(texts))
        return self._emit(texts)

    def encode_images(self, paths: list[str]) -> list[list[float]]:
        self.image_calls.append(list(paths))
        return self._emit(paths)


def row(
    chunk_id: str,
    *,
    chunk_type: str = TYPE_TEXT,
    image_path: str | None = None,
) -> ChildChunk:
    """造一个子块: text 默认可从 chunk_id 区分, 便于断言"哪个块拿到了哪个向量"。"""
    return ChildChunk(
        chunk_id=chunk_id,
        text=f"内容-{chunk_id}",
        chunk_type=chunk_type,
        image_path=image_path,
    )


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """把门面的两个环境依赖换成假的: `get_embedder`(后端) 与 `_cfg`(配置)。"""

    def _install(backend: FakeBackend, *, image_dir=None) -> FakeBackend:
        backend._tags = {}
        monkeypatch.setattr(embed_mod, "get_embedder", lambda: backend)
        monkeypatch.setattr(embed_mod, "_cfg", lambda: Config(image_dir=str(image_dir or tmp_path)))
        return backend

    return _install


@pytest.fixture
def fake_st(monkeypatch):
    """假 `sentence_transformers` 模块: 测 local 后端不装 torch。"""
    FakeST.instances.clear()
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return FakeST


class FakeST:
    """假 SentenceTransformer: 只实现 encode, 把入参原样记下来供断言。"""

    instances: list[FakeST] = []

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls: list[dict] = []
        FakeST.instances.append(self)

    def encode(self, texts, batch_size, normalize_embeddings):  # noqa: ANN001, ANN201
        self.calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        raw = [3.0, 4.0] + [0.0] * 6  # 模长恒为 5, 归一化后 [0.6, 0.8, 0, ...]
        # 照真模型的语义来: normalize_embeddings=True 时由**模型**返回单位向量(所以 local 路
        # 不再手算归一化); 万一以后忘了传这个开关, 期望值断言会立刻失败
        vec = _l2_normalize(raw) if normalize_embeddings else raw
        return [list(vec) for _ in texts]


def test_api_backend_declares_image_capability():
    assert _ApiBackend.supports_images is True
    assert _LocalBackend.supports_images is False


def test_embed_chunks_routes_and_preserves_order(wired, tmp_path):
    """混排(正文/表格/有图/无图/正文): 两组分别调, 结果按**原下标**拼回。"""
    root = tmp_path / "imgroot"
    (root / "d1").mkdir(parents=True)
    png = root / "d1" / "p0001_i001.png"
    png.write_bytes(b"\x89PNG\x00")
    rows = [
        row("text0"),
        row("table1", chunk_type=TYPE_TABLE),
        row("img2", chunk_type=TYPE_IMAGE, image_path="d1/p0001_i001.png"),
        row("img3", chunk_type=TYPE_IMAGE),  # md: 图片块没有本地图
        row("text4"),
    ]
    backend = wired(FakeBackend(), image_dir=root)

    out = embed_chunks(rows)

    assert backend.text_calls == [["内容-text0", "内容-table1", "内容-img3", "内容-text4"]]
    assert backend.image_calls == [[str(png)]]
    expected = [
        _l2_normalize(backend.vec("内容-text0")),
        _l2_normalize(backend.vec("内容-table1")),
        _l2_normalize(backend.vec(str(png))),
        _l2_normalize(backend.vec("内容-img3")),
        _l2_normalize(backend.vec("内容-text4")),
    ]
    assert len({tuple(v) for v in expected}) == 5  # 期望值两两不同, 上面的顺序断言才有意义
    assert out == expected


def test_embed_chunks_resolves_image_path_against_image_dir(wired, tmp_path):
    """`image_path` 是相对 `RAG2_IMAGE_DIR` 的: 拼错基准 → 所有图片块静默退化成文本。"""
    root = tmp_path / "imgroot"
    (root / "d1").mkdir(parents=True)
    png = root / "d1" / "a.png"
    png.write_bytes(b"\x89PNG\x00")
    backend = wired(FakeBackend(), image_dir=root)

    out = embed_chunks([row("img0", chunk_type=TYPE_IMAGE, image_path="d1/a.png")])

    assert backend.image_calls == [[str(png)]]
    assert backend.text_calls == []
    assert out == [_l2_normalize(backend.vec(str(png)))]


def test_embed_chunks_degrades_and_warns_when_image_root_is_wrong(wired, tmp_path):
    """根目录配错 = 图片都在但一律判"不在盘上": 走文本路, 并且要让人看见。"""
    root = tmp_path / "imgroot"
    (root / "d1").mkdir(parents=True)
    (root / "d1" / "a.png").write_bytes(b"\x89PNG\x00")
    warnings: list[str] = []
    backend = wired(FakeBackend(), image_dir=tmp_path / "wrong-root")

    rows = [row("img0", chunk_type=TYPE_IMAGE, image_path="d1/a.png")]
    out = embed_chunks(rows, warnings=warnings)

    assert backend.image_calls == []
    assert backend.text_calls == [["内容-img0"]]
    assert len(out) == 1
    assert warnings == ["1 个图片块有 image_path 但文件不在盘上, 已按文本路嵌入"]


def test_embed_chunks_md_image_without_path_is_silent(wired, tmp_path):
    """md 的图片块 image_path=None 是常态(02 篇实测), 降级但不刷警告。"""
    warnings: list[str] = []
    backend = wired(FakeBackend(), image_dir=tmp_path)

    out = embed_chunks([row("img0", chunk_type=TYPE_IMAGE)], warnings=warnings)

    assert backend.text_calls == [["内容-img0"]]
    assert backend.image_calls == []
    assert warnings == []
    assert len(out) == 1


def test_embed_chunks_local_backend_sends_everything_as_text(wired, tmp_path):
    """local 模式没有图片能力: 有本地图的图片块也走文本路(04 篇 §2.1)。"""
    root = tmp_path / "imgroot"
    (root / "d1").mkdir(parents=True)
    (root / "d1" / "a.png").write_bytes(b"\x89PNG\x00")
    backend = wired(FakeBackend(supports_images=False), image_dir=root)
    rows = [row("img0", chunk_type=TYPE_IMAGE, image_path="d1/a.png"), row("text1")]

    out = embed_chunks(rows)

    assert backend.image_calls == []
    assert backend.text_calls == [["内容-img0", "内容-text1"]]
    assert len(out) == 2


def test_embed_chunks_raises_when_backend_returns_fewer_vectors(wired, tmp_path):
    """后端少返向量必须立刻抛: `zip` 默认按短的截断, 之后的块会整体错位一格。"""
    wired(FakeBackend(drop_last=True), image_dir=tmp_path)

    with pytest.raises(RuntimeError, match="不匹配"):
        embed_chunks([row("t0"), row("t1")])


def test_embed_chunks_empty_rows_make_no_call(wired, tmp_path):
    backend = wired(FakeBackend(), image_dir=tmp_path)

    assert embed_chunks([]) == []
    assert backend.text_calls == []
    assert backend.image_calls == []


def test_embed_documents_and_query_delegate_to_backend(wired, tmp_path):
    backend = wired(FakeBackend(), image_dir=tmp_path)

    assert embed_documents(["a", "b"]) == [
        _l2_normalize(backend.vec("a")),
        _l2_normalize(backend.vec("b")),
    ]
    assert embed_query("q") == _l2_normalize(backend.vec("q"))


def test_embed_images_raises_on_local_backend(wired, tmp_path):
    """local 模式下调图片入口是**用错了**: 报错, 而不是悄悄走文本。"""
    wired(FakeBackend(supports_images=False), image_dir=tmp_path)

    with pytest.raises(ValueError, match="不支持图片"):
        embed_images(["a.png"])


def test_embedding_dim_measures_once_then_caches(wired, tmp_path):
    backend = wired(FakeBackend(), image_dir=tmp_path)
    embedding_dim.cache_clear()
    try:
        assert embedding_dim() == DIM
        assert embedding_dim() == DIM
    finally:
        embedding_dim.cache_clear()

    assert backend.text_calls == [["维度探针"]]  # 只发了一条探针


def test_get_embedder_rejects_unknown_mode(monkeypatch):
    monkeypatch.setattr(embed_mod, "_cfg", lambda: Config(embed_mode="bogus"))
    embed_mod.get_embedder.cache_clear()
    try:
        with pytest.raises(ValueError, match="只能是 api"):
            embed_mod.get_embedder()
    finally:
        embed_mod.get_embedder.cache_clear()


def test_local_backend_encodes_with_normalization(fake_st):
    backend = _LocalBackend(
        Config(embed_mode="local", embed_model="BAAI/bge-m3", embed_dim=DIM, embed_batch=7)
    )

    out = backend.encode(["一", "二"])

    st = fake_st.instances[0]
    assert st.model_name == "BAAI/bge-m3"
    assert st.calls == [
        {"texts": ["一", "二"], "batch_size": 7, "normalize_embeddings": True}
    ]
    assert out == [[0.6, 0.8] + [0.0] * 6] * 2


def test_local_backend_reports_missing_dependency(monkeypatch):
    """缺 torch/sentence-transformers 时报错要带装法, 不能只丢一句 ModuleNotFoundError。"""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # 让 import 直接失败

    with pytest.raises(ImportError, match="rag2-local"):
        _LocalBackend(Config(embed_mode="local"))
