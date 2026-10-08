"""构建评测语料：从 Python 官方中文文档抓取若干独立主题页，转成纯文本

为什么要有这个脚本（而不是手工塞几个 txt）：
    语料的**来源与许可是评测可信度的一部分**，必须可复现、可追溯。
    脚本记下每个文件的来源 URL、抓取时间与许可，任何人重跑都能得到同一份语料。

★ 语料选择原则（刻意避开的两类东西）
    1. 不用本机 `D:\\Projects` 下的规划类文档 —— 那些含个人信息，作品集仓库不能公开。
    2. 不用单一长文 —— 要能支撑"文档级"检索评测，必须是**多个相互独立的主题**，
       否则 doc-level hit@k 没有区分度。

数据来源：docs.python.org 中文文档（Python 官方文档的中文翻译）
许可：Python 软件基金会许可协议（PSF License）；本仓库仅作评测语料示例之用，署名见 corpus/SOURCES.md

用法：
    .\\.venv\\Scripts\\python.exe scripts\\build_corpus.py            # 抓取并重建语料
    .\\.venv\\Scripts\\python.exe scripts\\build_corpus.py --list     # 只看清单不抓取
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "corpus"
BASE_URL = "https://docs.python.org/zh-cn/3"

# 刻意选择**相互独立**的主题：12 个文档 → 文档级检索评测才有区分度
SOURCES: list[tuple[str, str]] = [
    ("01_python_intro", "/tutorial/introduction.html"),
    ("02_controlflow", "/tutorial/controlflow.html"),
    ("03_datastructures", "/tutorial/datastructures.html"),
    ("04_modules", "/tutorial/modules.html"),
    ("05_inputoutput", "/tutorial/inputoutput.html"),
    ("06_errors", "/tutorial/errors.html"),
    ("07_classes", "/tutorial/classes.html"),
    ("08_stdlib_overview", "/tutorial/stdlib.html"),
    ("09_json", "/library/json.html"),
    ("10_sqlite3", "/library/sqlite3.html"),
    ("11_pathlib", "/library/pathlib.html"),
    ("12_regex", "/library/re.html"),
]

SKIP_TAGS = {"script", "style", "nav", "head"}
BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "pre", "dt", "dd", "tr", "section"}
# 这些 class 的内容一律丢弃：
#   toctree-wrapper 是正文顶部的内嵌目录，sphinxsidebar/related 是导航 ——
#   ★ 它们在 12 个文档里逐字相同，留着会让"文档级"检索评测失效（样板文字人人都有）
NOISE_CLASSES = {"toctree-wrapper", "sphinxsidebar", "related", "headerlink", "header",
                 "footer", "mobile-nav", "searchbox", "ethical-sidebar", "ethical-footer",
                 "clearer", "announcement", "topic", "contents"}


class MainContentExtractor(HTMLParser):
    """只取正文 div 里的文字，丢掉导航/侧栏/内嵌目录等噪声

    docs.python.org 的正文是 `<div class="body" role="main">`。
    ⚠️ 不能用 `"body" in class` 判定 —— 外层还有 `class="bodywrapper"` 会先命中。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = -1                # 正文 div 内的嵌套层数；-1 表示还没进入
        self.skip_depth = 0            # 处在 script/style/噪声块内的层数
        self.parts: list[str] = []

    @staticmethod
    def _classes(attrs: list[tuple[str, str | None]]) -> set[str]:
        return set(((dict(attrs).get("class") or "")).split())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = self._classes(attrs)
        if self.depth < 0:
            if tag == "div" and "body" in classes:      # 精确匹配正文 div
                self.depth = 0
            return

        if tag == "div":
            self.depth += 1
        if tag in SKIP_TAGS or (classes & NOISE_CLASSES):
            self.skip_depth += 1
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.depth < 0:
            return
        if self.skip_depth > 0:
            self.skip_depth -= 1
        if tag == "div":
            self.depth -= 1
            if self.depth < 0:
                self.depth = -1            # 正文结束
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.depth >= 0 and self.skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = raw.replace("\xa0", " ")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw.split("\n")]
        noise = ("上一页", "下一页", "快速搜索", "模块代码", "跳到主要内容", "报告代码错误",
                 "显示源码", "显示翻译源", "当前页")
        kept = [line for line in lines if line and not any(word in line for word in noise)]
        deduped: list[str] = []
        for line in kept:
            if not deduped or deduped[-1] != line:
                deduped.append(line)
        return "\n".join(deduped)


def fetch(url: str, timeout: int = 30) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (kb-rag corpus builder)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def build(verbose: bool = True) -> int:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, int, str]] = []
    for name, path in SOURCES:
        url = BASE_URL + path
        try:
            html = fetch(url)
        except (urllib.error.URLError, OSError) as exc:
            print(f"[失败] {name}: {exc}")
            continue
        parser = MainContentExtractor()
        parser.feed(html)
        text = parser.text()
        (CORPUS_DIR / f"{name}.txt").write_text(text, encoding="utf-8")
        rows.append((f"{name}.txt", len(text), url))
        if verbose:
            print(f"[OK] {name}.txt  {len(text):>7} 字  <- {url}")

    stamp = datetime.now().strftime("%Y-%m-%d")
    sources_md = [
        "# 评测语料来源",
        "",
        f"抓取时间：{stamp}（重新抓取用 `scripts/build_corpus.py`）",
        "",
        "| 文件 | 字数 | 来源 |",
        "|---|---|---|",
    ]
    sources_md += [f"| `{name}` | {size} | {url} |" for name, size, url in rows]
    sources_md += [
        "",
        "## 许可与用途",
        "",
        "- 内容来自 Python 官方文档的中文翻译（docs.python.org），适用 **Python 软件基金会许可协议（PSF License）**。",
        "- 本仓库**仅将这些文件用作检索评测语料**（占位示例），不作其它用途；版权归 Python Software Foundation 所有。",
        "- 若本仓库需要对外发布，请保留本说明与 PSF 许可声明。",
        "",
        "## 为什么选这些文档",
        "",
        "1. **题材中性**：纯技术文档，不含任何个人信息（刻意避开本机规划类文档）。",
        "2. **主题互相独立**：12 个不同主题，使「文档级」检索评测（hit@k / MRR）有区分度；",
        "   若只用一篇长文，文档级指标会失去意义。",
        "3. **可复现**：来源 URL 与抓取方式都写在本目录与 `scripts/build_corpus.py` 里。",
        "",
    ]
    (CORPUS_DIR / "SOURCES.md").write_text("\n".join(sources_md), encoding="utf-8")

    total = sum(size for _, size, _ in rows)
    print(f"\n[语料] 共 {len(rows)} 个文档，合计 {total} 字，写入 {CORPUS_DIR}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="构建评测语料（Python 中文文档）")
    parser.add_argument("--list", action="store_true", help="只打印清单，不抓取")
    args = parser.parse_args()
    if args.list:
        for name, path in SOURCES:
            print(f"{name:<22} {BASE_URL + path}")
        return 0
    started = time.perf_counter()
    code = build()
    print(f"[语料] 耗时 {time.perf_counter() - started:.1f}s")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
