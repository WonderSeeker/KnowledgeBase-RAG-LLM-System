"""构建**医疗**评测语料：从中文维基百科抓取若干独立医学条目，转成纯文本。

与 scripts/build_corpus.py 同构（那边抓 Python 官方文档），差别只在来源与许可：
    来源：zh.wikipedia.org（MediaWiki API，prop=extracts 取纯文本）
    许可：**CC BY-SA 4.0** —— 使用时须署名并以相同协议共享（见 corpus_medical/SOURCES.md）

★ 为什么单独一份语料、不覆盖原 corpus/：
    corpus/（12 篇 Python 文档）上已经跑过一整套结论（3.8 / 4.10~4.12）；
    换语料 = 换评测前提，两套结论**不可混用**。所以医疗语料独立成目录、独立集合、独立评测集。

★ 语料选择原则（与 build_corpus.py 一致）：
    1. 来源可复现、可追溯（URL + 抓取时间 + 许可都写进 SOURCES.md）；
    2. 多个**相互独立**的主题 —— 否则「文档级」检索评测（hit@k / MRR）没有区分度；
    3. 不含个人信息（维基条目是公开百科内容）。

用法：
    .\\.venv\\Scripts\\python.exe scripts\\build_corpus_medical.py            # 抓取并重建
    .\\.venv\\Scripts\\python.exe scripts\\build_corpus_medical.py --list     # 只看清单
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

try:                                    # 可选依赖：没装也能跑，只是不做繁简兜底
    from zhconv import convert as _zh_convert
except ImportError:                     # pragma: no cover
    _zh_convert = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "corpus_medical"
API_URL = "https://zh.wikipedia.org/w/api.php"
UA = "kb-rag-eval-corpus-builder/1.0 (local retrieval evaluation; contact: local only)"

# 12 个**相互独立**的常见医学主题（条目质量较稳定的中文维基条目）
SOURCES: list[tuple[str, str]] = [
    ("01_hypertension", "高血压"),
    ("02_type2_diabetes", "2型糖尿病"),
    ("03_pneumonia", "肺炎"),
    ("04_anemia", "贫血"),
    ("05_asthma", "支气管哮喘"),
    ("06_depression", "重性抑郁障碍"),
    ("07_coronary_heart_disease", "冠心病"),
    ("08_stroke", "脑卒中"),
    ("09_osteoporosis", "骨质疏松症"),
    ("10_hypothyroidism", "甲状腺功能减退"),
    ("11_gastritis", "胃炎"),
    ("12_urticaria", "荨麻疹"),
    # ---- 第二批（2026-10-16 扩充）：覆盖呼吸/感染/肾脏/代谢/神经/精神/心血管 ----
    ("13_influenza", "流行性感冒"),
    ("14_tuberculosis", "肺结核"),
    ("15_hepatitis_b", "乙型肝炎"),
    ("16_chronic_kidney_disease", "慢性肾脏病"),
    ("17_gout", "痛风"),
    ("18_rheumatoid_arthritis", "类风湿性关节炎"),
    ("19_migraine", "偏头痛"),
    ("20_epilepsy", "癫痫"),
    ("21_alzheimer", "阿尔茨海默病"),
    ("22_anxiety_disorder", "焦虑症"),
    ("23_hyperlipidemia", "高脂血症"),
    ("24_heart_failure", "心力衰竭"),
]


def fetch_extract(title: str, timeout: int = 45) -> str:
    """经 MediaWiki API 取条目的纯文本正文（prop=extracts&explaintext=1）"""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",          # 纯文本，不要 HTML
        "redirects": "1",            # 接受重定向（标题拼写差异）
        "format": "json",
        "formatversion": "2",
        "titles": title,
        "variant": "zh-cn",          # ★ 繁简转换：zh.wikipedia 默认可能返回繁体，这里统一成简体
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    pages = (payload.get("query") or {}).get("pages") or []
    if not pages:
        raise RuntimeError("API 未返回页面")
    page = pages[0]
    if page.get("missing"):
        raise RuntimeError(f"条目不存在：{title}")
    extract = page.get("extract") or ""
    if not extract.strip():
        raise RuntimeError(f"条目正文为空：{title}")
    return to_simplified(clean(extract))


def to_simplified(text: str) -> str:
    """繁转简兜底。

    ★ 为什么不能只靠 API 的 `variant=zh-cn`：
      MediaWiki 的简繁转换对**嵌入模板 / 未标记内容**会漏转 —— 实测 12 篇里有 1 篇
      （`09_osteoporosis.txt`）混进 57 个繁体字（骨質疏鬆 / 髖關節骨 / 檢測 …）。
      混排会让「关键词字面匹配」这类口径悄悄失准，所以这里再兜一层。
    """
    if _zh_convert is None:
        return text
    return _zh_convert(text, "zh-cn")


def clean(text: str) -> str:
    """去掉维基 extract 里常见的小节标题噪声与多余空行，便于分块与检索"""
    lines: list[str] = []
    for raw in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw).strip()
        # extract 会把小节标题输出成 "== 症状 ==" 这种形式；去掉等号但保留标题文字
        line = re.sub(r"^=+\s*(.*?)\s*=+$", r"\1", line)
        if line:
            lines.append(line)
    deduped: list[str] = []
    for line in lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)
    return "\n".join(deduped)


def build(verbose: bool = True, skip_existing: bool = False) -> int:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, int, str, str]] = []      # (文件名, 字数, 条目标题, URL)
    for name, title in SOURCES:
        target = CORPUS_DIR / f"{name}.txt"
        if skip_existing and target.exists():
            # 已存在就跳过抓取，但仍要并入 SOURCES.md（否则清单会丢一半语料）
            size = len(target.read_text(encoding="utf-8"))
            rows.append((f"{name}.txt", size, title,
                         f"https://zh.wikipedia.org/wiki/{urllib.parse.quote(title)}"))
            if verbose:
                print(f"[跳过] {name}.txt（已存在，{size} 字）")
            continue
        text = None
        last_error: Exception | None = None
        for _attempt in (1, 2):                     # 维基偶发读超时，重试一次
            try:
                text = fetch_extract(title)
                break
            except (urllib.error.URLError, OSError, RuntimeError) as exc:
                last_error = exc
                time.sleep(2)
        if text is None:
            print(f"[失败] {name}（{title}）：{last_error}")
            continue
        if len(text) < 3000:                        # 太短的条目做检索评测区分度差，显式提示
            print(f"[警告] {name}（{title}）只有 {len(text)} 字，偏短，建议换更充实的条目")
        (CORPUS_DIR / f"{name}.txt").write_text(text, encoding="utf-8")
        page_url = f"https://zh.wikipedia.org/wiki/{urllib.parse.quote(title)}"
        rows.append((f"{name}.txt", len(text), title, page_url))
        if verbose:
            print(f"[OK] {name}.txt  {len(text):>7} 字  <- {title}")

    stamp = datetime.now().strftime("%Y-%m-%d")
    sources_md = [
        "# 医疗评测语料来源",
        "",
        f"抓取时间：{stamp}（重新抓取用 `scripts/build_corpus_medical.py`）",
        "",
        "| 文件 | 字数 | 条目 | 来源 |",
        "|---|---|---|---|",
    ]
    sources_md += [f"| `{name}` | {size} | {title} | {url} |" for name, size, title, url in rows]
    sources_md += [
        "",
        "## 许可与用途",
        "",
        "- 内容来自**中文维基百科**（zh.wikipedia.org），适用 **知识共享 署名-相同方式共享 4.0（CC BY-SA 4.0）**。",
        "- 本仓库**仅将这些条目用作检索评测语料**；版权归各条目贡献者所有，署名与许可信息见上表链接。",
        "- CC BY-SA 要求：使用时署名、并以相同协议共享 —— 若本仓库对外发布，请保留本说明与许可声明。",
        "",
        "## 为什么单独一份语料",
        "",
        "`corpus/`（Python 文档）上已跑过 3.8 / 4.10~4.12 的结论。换语料等于换评测前提，",
        "**两套结论不可相互引用**；因此医疗语料独立成目录、独立向量库集合、独立评测集。",
        "",
        "## 局限（引用前必读）",
        "",
        "- 这是**公开百科条目**，不是临床指南/病历，**不能代表真实医疗文档分布**；",
        "  真做医疗 RAG 还需脱敏、术语/缩写表、以及「答案可溯源到原文哪一句」的判分口径。",
        "- 条目内容会随时间被编辑；本目录的 txt 是抓取时点的快照（时间见上）。",
        "",
    ]
    (CORPUS_DIR / "SOURCES.md").write_text("\n".join(sources_md), encoding="utf-8")

    total = sum(size for _, size, _, _ in rows)
    print(f"\n[语料] 共 {len(rows)} 个文档，合计 {total} 字，写入 {CORPUS_DIR}")
    return 0


def reconvert() -> int:
    """把已有语料就地繁转简（不重新抓取）—— 用于修复历史遗留的混排"""
    fixed = 0
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        converted = to_simplified(text)
        if converted != text:
            path.write_text(converted, encoding="utf-8")
            print(f"[繁转简] {path.name}（{len(text)} -> {len(converted)} 字）")
            fixed += 1
    print(f"[繁转简] 完成：改写了 {fixed} 个文件")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="构建医疗评测语料（中文维基医学条目）")
    parser.add_argument("--list", action="store_true", help="只打印清单，不抓取")
    parser.add_argument("--reconvert", action="store_true",
                        help="不联网，只把已有 txt 做一次繁转简（修复混排）")
    parser.add_argument("--add", action="store_true",
                        help="只抓清单里**还没有**的条目，已存在的跳过（扩充语料时用）")
    args = parser.parse_args()
    if args.list:
        for name, title in SOURCES:
            print(f"{name:<28} {title}")
        return 0
    if args.reconvert:
        return reconvert()
    started = time.perf_counter()
    code = build(skip_existing=args.add)
    print(f"[语料] 耗时 {time.perf_counter() - started:.1f}s")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
