"""embed.py —— 双模式 embedding: 文本与图片 → 1024 维 dense 向量(04 篇)。

两条路共用同一个多模态模型、同一个向量空间(qwen3-vl-embedding):
- 文本路: text / table 块, 以及取不到本地图的 image 块(降级) → {"text": ...}
- 图片路: 有本地图的 image 块 → {"image": "data:image/png;base64,..."}
查询侧(用户打的问题)永远走文本路 —— 靠同一向量空间去跨模态命中图片块。

本文件只 import 同级的 config / contracts, 不碰项目其它包, 整目录可搬走。
"""

from __future__ import annotations

import base64
import math
from functools import lru_cache
from pathlib import Path

from .config import Config, load_config
from .contracts import TYPE_IMAGE, ChildChunk

# 单图 10MB 是百炼接口硬限制, 超了服务端直接拒; 与其等一个看不懂的 4xx, 不如本地拦下
_MAX_IMAGE_BYTES = 10 * 1024 * 1024

# 只认 02 篇落盘时可能出现的两种后缀(RAG2_IMAGE_FORMAT = png|jpg)
_MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

@lru_cache(maxsize=1)
def _cfg()->Config:
    """配置懒加载(04 篇骨架写的是模块级 `_DIM = cfg.embed_dim`, 这里改成懒加载)。

    模块级取值会在 `import embed` 那一刻就把环境锁死: 单测想注假配置、入口想先载 .env
    再 import, 都变成"import 顺序"问题; lru_cache 又保证只读一次环境, 不重复开销。
    """
    return load_config()

def _l2_normalize(vec: list[float]) -> list[float]:
    """L2 归一化, 让 ‖v‖=1(04 篇 §2.2)。

    对 COSINE 的排序没有任何影响(Milvus 自己会归一化), 做它是为了: 两个后端输出尺度一致、
    分数落进 [-1,1] 便于肉眼调试、将来换 IP 度量时排名不变。
    """
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec  # 零向量: 不要除以 0(得到 nan, nan 进库是静默污染)
    return [x / norm for x in vec]

def _to_data_uri(path: str) -> str:
    """本地图片 → `data:image/png;base64,...`(百炼收 Base64, 不必自建图床)。

    文件不存在时抛 FileNotFoundError 是**有意的**: "图没了就走文本路"这个降级判断留给
    调用方(第 2 步的 embed_chunks 先判 is_file 再分组), 本函数只做纯粹的转换。
    """
    p = Path(path)
    mime = _MIME_BY_SUFFIX.get(p.suffix.lower())
    if mime is None:
        raise ValueError(f"不支持的图片后缀 {p.suffix!r}: {p}(02 篇只落 png/jpg)")
    raw = p.read_bytes()
    if len(raw) > _MAX_IMAGE_BYTES:
        raise ValueError(f"图片过大: {p} {len(raw) / 1048576:.1f}MB, 超过单图 10MB 上限")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

def _check_dim(vectors: list[list[float]], dim: int, *, what: str) -> list[list[float]]:
    """维度自检(04 篇坑位 5): 刚拿到向量就验, 不等 Milvus 来报错。

    Milvus 的维度不一致要拖到建索引/插入阶段才炸, 且报错不指向配置项; 这里出错直接点名
    `RAG2_EMBED_DIM` 与"改维度必须重建 collection", 省掉一轮排查。
    """
    for vec in vectors:
        if len(vec) != dim:
            raise ValueError(
                f"{what} 返回 {len(vec)} 维, 与 RAG2_EMBED_DIM={dim} 不一致; "
                "两者必须相同, 且改维度后必须重建 collection(04 篇坑位 6)"
            )
    return vectors

def _image_file(row:ChildChunk,cfg:Config)->Path | None:
    """把 `image_path` 还原成真实路径; 取不到或盘上没有 → None。

    坑在这里: `image_path` 是**相对 `RAG2_IMAGE_DIR`** 的路径(02 篇 §4.5), 不是相对 cwd ——
    直接 `Path(row.image_path).is_file()` 会一律判假, 于是所有图片块悄悄退化成文本。
    """
    if not row.image_path:
        return None
    path=Path(cfg.image_dir)/row.image_path
    return path if path.is_file() else None

class _ApiBackend:
    """百炼多模态模式(定稿路径, 04 篇 §2.1)。

    必须走 DashScope SDK: OpenAI 兼容端点不支持图片输入, 所以这里不能用 openai.OpenAI,
    也不能靠改 base_url 绕过去。
    """
    supports_images = True  # 能力声明: embed_chunks 据此决定图片块走图片路还是文本路

    def __init__(self, cfg: Config) -> None:
        if not cfg.embed_api_key:
            raise ValueError("RAG2_EMBED_MODE=api 需要 RAG2_EMBED_API_KEY(01 篇 §三)")
        if min(cfg.embed_batch, cfg.embed_img_batch) <= 0:
            raise ValueError("RAG2_EMBED_BATCH / RAG2_EMBED_IMG_BATCH 必须是正整数")
        import dashscope  # 懒 import: local 模式不必装 dashscope
        from dashscope import MultiModalEmbedding

        dashscope.base_http_api_url = cfg.embed_api_base  # 国际站/代理: 靠这个模块级全局切端点
        self._mm = MultiModalEmbedding
        self.model = cfg.embed_model
        self.api_key = cfg.embed_api_key
        self.dim = cfg.embed_dim
        self.batch = cfg.embed_batch  # 文本批 ≤20
        self.img_batch = cfg.embed_img_batch  # 图片批 ≤5(服务端限制)

    def _call(self, contents: list[dict]) -> list[list[float]]:
        """一次请求。contents 如 [{"text": "..."}] / [{"image": "data:...;base64,..."}]。"""
        resp = self._mm.call(
            api_key=self.api_key,
            model=self.model,
            input=contents,
            # dimension 必须走**顶层参数**: 写成 parameters={"dimension": ...} 会被静默忽略,
            # 返回默认 2560 维(dashscope 1.27.6 实测), 最后死在 embed 层的维度自检上。
            dimension=self.dim,
        )
        if resp.status_code != 200:
            # SDK 不抛异常, 失败只体现在响应对象里: 不在这里抛, 上层会崩在 output 为 None 上
            raise RuntimeError(
                f"embedding 失败: status_code={resp.status_code} code={resp.code} "
                f"message={resp.message} request_id={resp.request_id}"
            )
        items = resp.output["embeddings"]
        if all("index" in it for it in items):
            # 响应带 index: 有就按 index 排一遍, 免得多批之后顺序被服务端重排(错序=向量错位)
            items = sorted(items, key=lambda it: it["index"])
        return [_l2_normalize(list(it["embedding"])) for it in items]

    def encode(self, texts: list[str]) -> list[list[float]]:
        """文本批: 每 `self.batch` 条发一次请求, 返回顺序与入参严格一一对应。"""
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            out += self._call([{"text": t} for t in texts[i : i + self.batch]])
        return _check_dim(out, self.dim, what="文本 embedding")

    def encode_images(self, paths: list[str]) -> list[list[float]]:
        """图片批: 每 `self.img_batch` 张发一次请求(比文本批小, 因服务端限一次 ≤5 张图)。"""
        out: list[list[float]] = []
        for i in range(0, len(paths), self.img_batch):
            batch = paths[i : i + self.img_batch]
            out += self._call([{"image": _to_data_uri(p)} for p in batch])
        return _check_dim(out, self.dim, what="图片 embedding")

class _LocalBackend:
    """本地 sentence-transformers 模式(离线备选, 04 篇 §2.1)。

    只认文本: bge-m3 这类文本模型没有图片能力, 图片块由 `embed_chunks` 统一退回文本路,
    所以这里**不提供** `encode_images` —— 能力有无用 `supports_images` 显式声明, 不做
    `hasattr` 那种动态判断(那种写法在换后端时会静默失效)。
    """

    supports_images = False

    def __init__(self, cfg: Config) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # torch 与权重都不在定稿依赖里, 缺了要能自己说清怎么装
            raise ImportError(
                "RAG2_EMBED_MODE=local 需要 sentence-transformers: uv sync --extra rag2-local"
            ) from exc
        self.model = SentenceTransformer(cfg.embed_model)  # 首次会联网拉权重(见 01 篇 HF_ENDPOINT)
        self.dim = cfg.embed_dim
        self.batch = cfg.embed_batch

    def encode(self, texts: list[str]) -> list[list[float]]:
        """`normalize_embeddings=True` 让 local 路与 api 路尺度一致(04 篇 §2.2), 不用再手算。"""
        arr = self.model.encode(texts, batch_size=self.batch, normalize_embeddings=True)
        return _check_dim([list(v) for v in arr], self.dim, what="文本 embedding(local)")

@lru_cache(maxsize=1)
def get_embedder() -> _ApiBackend | _LocalBackend:
    """单例后端(懒加载): `RAG2_EMBED_MODE=api|local` 二选一。

    做成单例是因为 api 模式的构造有副作用(设 `base_http_api_url`)、local 模式要加载权重;
    每次调用都新建一遍会很贵, 而单例又不会绑死环境(第 1 步已经解释了为什么不用模块级常量)。
    """
    cfg = _cfg()
    if cfg.embed_mode == "api":
        return _ApiBackend(cfg)
    if cfg.embed_mode == "local":
        return _LocalBackend(cfg)
    raise ValueError(f"RAG2_EMBED_MODE 只能是 api|local, 收到 {cfg.embed_mode!r}")


def embed_documents(texts: list[str]) -> list[list[float]]:
    """批量文本 → 向量(顺序与入参一致)。纯文本入口; 子块列表请用 `embed_chunks`。"""
    return get_embedder().encode(texts)


def embed_images(paths: list[str]) -> list[list[float]]:
    """批量图片路径 → 向量。local 模式下明确报错, 不静默降级(降级是 `embed_chunks` 的决定)。"""
    backend = get_embedder()
    if not backend.supports_images:
        raise ValueError("当前后端不支持图片向量(local 模式): 图片块应走文本路, 不该调到这里")
    return backend.encode_images(paths)


def embed_query(text: str) -> list[float]:
    """单条查询向量: 与入库**同一后端、同一模式**(查询侧永远送文本, 无指令前缀)。"""
    return get_embedder().encode([text])[0]


@lru_cache(maxsize=1)
def embedding_dim() -> int:
    """实测维度(首次调用真发一条最小请求, 之后缓存)。

    为什么不用 `cfg.embed_dim` 直接返回: 向量维度是**服务端**决定的, 配置只是我们的期望;
    真量一次才能发现"配置 1024、实际 2560"这种偏差(顶层参数写错位置就是这个后果)。
    不一致由 `_check_dim` 抛错, 这里不吞。
    """
    return len(embed_query("维度探针"))


def embed_chunks(
    rows: list[ChildChunk], *, warnings: list[str] | None = None
) -> list[list[float]]:
    """子块列表 → 向量列表: 长度与顺序与入参**严格一一对应**(04 篇坑位 10)。

    分组的两个理由: (1) 文本与图片的 payload 形态和批大小都不同(文本 ≤20 / 图片 ≤5);
    (2) image 块要**逐条**判断"图还在不在盘上", 不在就退回文本路 —— 逐条的事只能分组做。

    顺序是这条流水线的生命线: 分组 → 分别调 → **按原下标拼回**。少了最后一步, 向量就写到
    别人的 chunk 上, 而 Milvus 照收不报错, 检索从此静默变差(最难查的一类故障)。

    `warnings` 是可选的收集口: 传进来就把"图片有路径但文件不在盘上"这类异常记进去(07 篇的
    ingest 报告用); 不传就只影响返回值。md 的图片块本来没有本地图(02 篇实测), 属正常降级,
    不记警告 —— 记了会把警告刷满。
    """
    backend, cfg = get_embedder(), _cfg()
    text_idx: list[int] = []
    img_idx: list[int] = []
    img_paths: list[str] = []
    dropped: list[str] = []  # image_path 有值但盘上找不到: 异常, 该让人看见

    for i, row in enumerate(rows):
        if row.chunk_type != TYPE_IMAGE or not backend.supports_images:
            text_idx.append(i)  # 正文 / 表格 / local 模式下的图片块: 都走文本路
            continue
        path = _image_file(row, cfg)
        if path is None:
            if row.image_path:
                dropped.append(row.chunk_id)
            text_idx.append(i)  # 占位文本 + 图注仍能参与检索, 总比丢块强
            continue
        img_idx.append(i)
        img_paths.append(str(path))

    placed: dict[int, list[float]] = {}
    if text_idx:
        _place(placed, text_idx, backend.encode([rows[i].text for i in text_idx]), "文本路")
    if img_idx:
        _place(placed, img_idx, backend.encode_images(img_paths), "图片路")
    if warnings is not None and dropped:
        warnings.append(f"{len(dropped)} 个图片块有 image_path 但文件不在盘上, 已按文本路嵌入")
    return [placed[i] for i in range(len(rows))]


def _place(
    placed: dict[int, list[float]], indexes: list[int], vectors: list[list[float]], what: str
) -> None:
    """把一批向量按**原下标**放回结果字典; 条数对不上立刻抛。

    为什么要单独一个函数 + 一次条数校验: `zip` 是"短的截断长的", 后端少返一条不会报错,
    只会让后面整体错位一格 —— 静默错位的代价远大于多写这四行。
    """
    if len(vectors) != len(indexes):
        raise RuntimeError(f"{what}返回 {len(vectors)} 条向量, 与 {len(indexes)} 条内容不匹配")
    placed.update(zip(indexes, vectors, strict=True))
