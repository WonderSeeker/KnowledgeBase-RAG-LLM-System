"""
命令行入库脚本：把 txt / pdf 批量写入 Chroma 向量库

为什么需要它：
    原项目只能通过 Streamlit 页面上传，无法自动化、无法复现。
    做基线记录和 RAGAS 评测都需要「同样的语料、同样的入库过程」，
    所以把入库抽成命令行脚本，绕开浏览器。

用法（在项目根目录执行）：
    # 默认入库 assets/ 下全部 txt（项目自带的 3 份中文示例文本）
    .\\.venv\\Scripts\\python.exe scripts\\ingest.py

    # 指定文件或目录
    .\\.venv\\Scripts\\python.exe scripts\\ingest.py data\\手册.pdf
    .\\.venv\\Scripts\\python.exe scripts\\ingest.py data

前提：.env 里的 DASHSCOPE_API_KEY 已填好（入库要用通义 embedding 接口）
"""
import sys
import time
from pathlib import Path

# 让脚本无论从哪里启动，都能 import 到项目根目录下的模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_base import KnowledgeBaseService, extract_text  # noqa: E402

SUPPORTED_SUFFIX = {".txt", ".pdf"}


def collect_files(target: Path) -> list[Path]:
    """收集待入库文件：传文件就返回它本身，传目录就递归找 txt/pdf"""
    if target.is_file():
        return [target] if target.suffix.lower() in SUPPORTED_SUFFIX else []
    return sorted(p for p in target.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIX)


def main() -> int:
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else PROJECT_ROOT / "assets"
    if not target.exists():
        print(f"[错误] 路径不存在：{target}")
        return 1

    files = collect_files(target)
    if not files:
        print(f"[跳过] {target} 下没有可入库的 txt/pdf 文件")
        return 0

    print(f"[入库] 目标：{target}")
    print(f"[入库] 待处理 {len(files)} 个文件\n")

    service = KnowledgeBaseService()      # 建对象即会新建/连接chroma_db
    success = 0
    for file_path in files:
        start_time = time.perf_counter()
        text = extract_text(file_path.read_bytes(), file_path.name)
        result = service.upload_by_str(text, file_path.name)
        cost = time.perf_counter() - start_time
        print(f"  → {file_path.name}：{result}（含读取，本文件共 {cost:.2f}s）")
        if "Success" in result:
            success += 1

    print(f"\n[入库] 完成：共 {len(files)} 个文件，成功 {success} 个，跳过 {len(files) - success} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
