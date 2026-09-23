"""docling 解析 Word(.docx)示例。

运行:
    uv run python src/rag_v01/load.py

要点:
- docling 按后缀自动挑后端:.docx -> MsWordDocumentBackend,纯结构解析、不下载模型;
  只有 PDF/图片才走 ML 版面分析管线,首次运行会从 HuggingFace 拉模型。
- DocumentConverter 构造一次即可复用,批量转换时别写在循环里。
- docx 里的图片被解析成 PictureItem,image.uri 已是 base64 data URI;
  导出 markdown 必须显式给 image_mode,默认 PLACEHOLDER 只留 <!-- image -->。
- 传路径一律用绝对路径:相对路径会被 docling 按 md 所在目录再拼一次,套娃出多层目录。
"""

from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc.base import ImageRefMode

# 语料与生成物统一放仓库根 data/rag2_samples/(c8:代码包零数据,语料唯一口径);
# 生成物(sample1.md / sample1_embedded.md / sample1_artifacts/)已入 .gitignore,重跑本脚本即再生
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/rag_v01/load.py -> 仓库根
SAMPLES_DIR = PROJECT_ROOT / "data" / "rag2_samples"
DOC_PATH = SAMPLES_DIR / "doc" / "sample1.docx"
MD_PATH = SAMPLES_DIR / "md" / "sample1.md"
EMBEDDED_PATH = SAMPLES_DIR / "md" / "sample1_embedded.md"
ARTIFACTS_DIR = SAMPLES_DIR / "md" / "sample1_artifacts"


def main() -> None:
    if not DOC_PATH.exists():
        raise FileNotFoundError(f"没找到待解析文件: {DOC_PATH}")

    # 1) 转换器:内部管理后端与(按需加载的)模型,复用同一个实例
    converter = DocumentConverter()

    # 2) 解析:返回 ConversionResult;解析失败时 status 是 FAILURE,不抛异常
    result = converter.convert(DOC_PATH)
    print(f"解析状态: {result.status}")
    doc = result.document
    print(f"来源文件: {doc.origin.filename if doc.origin else '-'}")
    print(f"页数: {doc.num_pages()}  (.docx 无页概念,恒为 0)")
    print(f"图片数: {len(doc.pictures)}")

    # 3) 三种图片处理方式,按用途挑一个

    # 3a) PLACEHOLDER(默认):图片位置只留 <!-- image -->,喂纯文本 RAG 最省 token
    md_plain = doc.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)
    print(f"占位符版长度: {len(md_plain)} 字符")

    # 3b) EMBEDDED:base64 内嵌,md 单文件自包含,适合上传/喂多模态模型(体积大)
    md_embedded = doc.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
    print(f"内嵌版长度:   {len(md_embedded)} 字符(含 base64)")

    # 3c) REFERENCED:md 里写相对链接,图片另存为 png 文件,适合人工查看/换机器不失效
    #     image_uri_prefix 是原样拼在文件名前的,结尾必须带 "/",否则会粘成 xxximage_000000.png
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    md_referenced = doc.export_to_markdown(
        image_mode=ImageRefMode.REFERENCED,
        image_dir=ARTIFACTS_DIR,
        image_uri_prefix=f"{ARTIFACTS_DIR.name}/",
    )
    MD_PATH.write_text(md_referenced, encoding="utf-8")
    pngs = sorted(ARTIFACTS_DIR.glob("*.png"))
    print(f"已写入: {MD_PATH}")
    print(f"图片目录: {ARTIFACTS_DIR}  ({len(pngs)} 张 png)")

    EMBEDDED_PATH.write_text(md_embedded, encoding="utf-8")

    # 4) 要做切片/入库就改用结构化导出:每个文本块自带层级与来源定位信息
    data = doc.export_to_dict()
    print(f"结构化导出: {len(data.get('texts', []))} 个文本块, {len(data.get('tables', []))} 张表")

    if doc.pictures:
        first = doc.pictures[0]
        print(f"首图: {first.image.size} 像素, uri 前缀 {str(first.image.uri)[:32]}…")


if __name__ == "__main__":
    main()
