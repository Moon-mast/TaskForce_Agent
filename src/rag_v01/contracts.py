"""rag_0.1 的数据契约(唯一出处)。

规则: 所有模块共享的类型只在这里定义, 其它模块一律 `from contracts import ...`,
不许各自复制一份(改字段时只改这一处)。
当前已落: 解析侧 ParsedDoc / ParsedItem、切块侧 ParentChunk / ChildChunk、检索侧 RetrievedChunk /
ChildHit(05 篇)、入库报告 IngestReport(07 篇)、评估报告 EvalReport(06 篇)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 取值用常量而不是裸字符串: 03/04 篇要按 type 分支(表格不切、图片走多模态嵌入),
# 字符串拼错不会报错、只会悄悄走错分支, 所以集中定义。
TYPE_TEXT = "text"
TYPE_TABLE = "table"
TYPE_IMAGE = "image"
TYPES = (TYPE_TEXT, TYPE_TABLE, TYPE_IMAGE)

@dataclass(frozen=True, slots=True)
class ParsedItem:
    """解析后的一条结构条目(最小字段集, 见 02 篇 §三)。

    level 与 heading_path 的分工(最容易混):
    - `level`: 只有标题条目有值 —— 0 = 文档标题, 1/2/... = 各级小标题;
        正文 / 表格 / 图片条目恒为 None;
    - `heading_path`: 该条目"所处位置"的标题面包屑, 只含上级、不含自身 ——
        所以标题条目自己的 heading_path 也不含自己的文本, 03 篇的标题栈可直接 push。
    """

    type: str
    text: str  # 正文原文 / 整表 Markdown / 图片占位文本, 恒非空
    heading_path: str = ""      # 所处位置的标题面包屑(只含上级不含自身); 03 篇直接拿它当父块面包屑
    page_no: int | None = None  # pdf 有页码; docx/md 无页码概念 → None(落库时 04 篇写 -1)
    level: int | None = None    # 仅标题条目有值(0=文档标题, 1/2=各级小标题); 其余条目恒 None
    image_path: str | None = None  # 仅 type=image: 相对 RAG2_IMAGE_DIR 的路径; md 取不到 → None

    @property
    def is_heading(self) -> bool:
        """标题条目 = type=text 且有 level(02 篇 §三 接缝约定 1)。"""
        return self.type == TYPE_TEXT and self.level is not None

@dataclass(frozen=True, slots=True)
class ParsedDoc:
    """一份文件解析(并清洗)后的全部条目与溯源信息。"""

    doc_id: str                    # 文件字节流 sha256 前 16 位(内容寻址: 同内容 → 同 id)
    source: str                    # 相对语料根的 POSIX 路径; 单文件入库时就是文件名
    items: list[ParsedItem] = field(default_factory=list)   # 按阅读顺序
    meta: dict = field(default_factory=dict)     # 后端/页数/耗时/warnings, 由 ingest 汇总


@dataclass(frozen=True,slots=True)
class Chunk:
    """父块 / 子块共用的九个元信息字段 + 正文(03 篇 §3.1, 命名固定不得增删改名)。

    父块与子块的字段完全相同, 差别只在取值约定:
    - 父块: `parent_id` 为空串, `chunk_id` 就等于子块 `parent_id` 所指的那个 id 串;
    - 子块: `parent_id` 指向父块主键, `chunk_id` 再追加 `:c{序号}`。

    `char_len` 由 `text` 派生(见 `__post_init__`), 调用方不用传 —— 九字段里它最容易写错
    (重叠拼接、去空白之后忘了回填), 派生了就不会与 `text` 脱节。
    子类 `ParentChunk` / `ChildChunk` 只是给 04 篇的 `upsert_parents` / `upsert_children` 一个
    能进类型注解的区分, 不新增字段。
    """

    chunk_id:str           # 唯一键: 进库后靠它定位"这一块"(子块形如 f"{parent_id}:c{序号}")
    text:str               # 块的正文(已含标题前缀; 子块含重叠): 进向量与 BM25 的就是它
    parent_id:str=""        # 归属: 子块填父块主键, 父块自己留空串(父块没有父亲)
    doc_id:str=""           # 来源文档的内容 id: 按文档整体删除/重灌靠它(04 篇)
    source:str=""           # 来源文件路径: 检索命中时告诉用户"依据来自哪个文件"
    page_no:int|None=None   # 页码: pdf 有; docx/md 恒 None(入库前 04 篇换 -1 哨兵)
    heading_path:str=""     # 所属章节的面包屑(父块取开块单元的位置)
    chunk_type:str=TYPE_TEXT  # text/table/image: 子块层据此分支, 04 篇据此选嵌入方式
    char_len:int=0          # 由 text 量出, 别手填(见 __post_init__)
    created_at:str=""       # 切块时刻 UTC ISO 8601; 不参与 ID 生成, 不破坏确定性
    # 以下为**透传字段**, 不属于九字段(02 篇 §三 接缝约定 2): 04 篇把它收进 Milvus 的
    # metadata(JSON), 并作为读取图片本体做多模态嵌入的入口。

    image_path:str|None=None  # 图片块专属的落盘相对路径(表格/正文块为 None)

    def __post_init__(self)->None:
        # frozen 挡不住 object.__setattr__ —— 这是 dataclass 的官方逃逸口, 只在这里用:
        # 把派生量钉死, 免得字符串切片/重叠拼接之后 char_len 忘了同步。
        object.__setattr__(self, "char_len", len(self.text))

@dataclass(frozen=True,slots=True)
class ParentChunk(Chunk):
    "父块: 进 parents collection(仅主键点查, 无向量索引), 命中子块后回溯它取上下文。"

@dataclass(frozen=True, slots=True)
class ChildChunk(Chunk):
    """子块: 进 children collection 承担 dense + BM25 匹配(04 篇)。"""

@dataclass(frozen=True, slots=True)
class ChildHit:
    """一条命中子块明细(05 篇 §3.2): 挂在父块条目下, 说明"为什么这段语境被选中"。

    两个名次是这块子块在**两路各自**的位置, `None` = 该路没进前 N(名次从 1 起, 不用 0 哨兵,
    否则与"未命中"歧义)。它们进了契约而不是只留在 retrieve 内部, 是因为 06 篇评估要看
    "某路召回强不强"、07 篇调试也要打印 —— 融合后再也拿不到这两个数。
    """

    chunk_id: str
    text: str
    chunk_type: str
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """`search()` 的返回单元: **一个父块一条**(05 篇 §3.2)。

    `hits` 允许为空列表吗? 不允许 —— 只有被至少一个子块命中的父块才会出现在结果里, 所以
    `hits` 恒 ≥ 1 条。`rrf_rank` 取"该父块里最佳子块的名次", 于是结果列表的顺序 == 最佳子块的
    融合顺序, 而每个父块的命中明细全留在 `hits` 里, 信息不丢。
    """

    parent_id: str
    parent_text: str          # 父块全文: 这才是最终喂给模型/给人读的上下文主体
    rrf_rank: int             # 1 起
    hits: list[ChildHit]      # 本父块下进入 top-N 的命中子块(≥1 条)
    doc_id: str = ""
    source: str = ""
    page_no: int | None = None
    heading_path: str = ""
    metadata: dict = field(default_factory=dict)   # 父块透传元信息(含图片块的 image_path)

@dataclass(frozen=True,slots=True)
class IngestReport:
    """一次 `ingest()` 的结果(07 篇 §3.2): CLI 打印它, 调用方按它判断成败。"""

    ok_sources: list[str]  # 成功入库的 source(相对语料根的 POSIX 路径)
    failed: dict[str, str]  # source -> 失败原因(解析失败 / 入库失败)
    total_parents: int = 0
    total_children: int = 0
    warnings: dict[str, list[str]] = field(default_factory=dict)  # source -> 警告(解析/清洗/入库)


@dataclass(frozen=True, slots=True)
class EvalReport:
    """一次 `evaluate()` 的结果(06 篇 §3.4): 契约只留程序要用的最小面, 明细在报告文件里。

    为什么明细不进契约: 逐题分数、judge 型号、版本号、参数快照这些是"给人读的产物", 放在
    `report_dir` 下的 json/markdown 里; 契约只保留调用方(CLI、上层)真正需要的最小面。
    """

    metrics: dict[str, float]  # 四项指标的均值, 键用 06 篇 §2.2 的口径名; NaN 的题不计入均值
    question_count: int  # 参与评估的题目数(NaN 的也计入, 报告里单独统计)
    report_dir: str  # json + markdown 报告的落盘目录
