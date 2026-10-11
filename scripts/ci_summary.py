"""把 junit.xml 变成 CI 的 job summary

★ 为什么这段逻辑要放在脚本里、而不是写进 workflow 的内联 heredoc：
   因为这里的"声明"本身也需要被测试。
   CI 页面上那句「本 job 不跑 IR 评测」如果哪天腐化掉了，
   外人看到一个绿勾就会以为所有测试都过了 —— 那是最危险的一种"谎言"。
   所以 `summarize()` 是纯函数，`tests/test_ci_summary.py` 会钉住它的输出。

用法（CI 里）：
    python -m pytest --junitxml=junit.xml
    python scripts/ci_summary.py junit.xml
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# 这段声明是"诚实性"的核心：公开的绿勾必须说明它**没有**覆盖什么。
SCOPE_STATEMENT = [
    "### 本次 CI 的范围（诚实声明）",
    "",
    "- 本 job 只跑**确定性单元测试**：不联网、不需要 API Key、不需要本地嵌入服务。",
    "- **不跑 IR 评测**：IR 指标本身确定，但**向量腿不是** —— chromadb 走 HNSW 近似检索，"
    "索引图每个进程重建，实测同一配置跨进程有 24/36 题的候选集合不同（CHANGES §4.10.5、§4.11.2）。"
    "把它做成门禁会**随机变红**，所以它只在需要时人工跑，且必须带 `--repeat 3` 输出实测噪声地板。",
    "- **不跑 RAGAS**：4 个指标需要 LLM 裁判（`DASHSCOPE_API_KEY`），仓库里不存密钥。",
]


def summarize(junit_path: str | Path) -> str:
    """读 junit.xml，产出 Job Summary 文本

    解析失败（文件不存在 / 不是合法 XML）时**不抛异常** —— CI 的 summary 步骤
    不该把一个已经失败的 job 变成"两个失败"。它只会如实说明"拿不到统计"。
    """
    lines: list[str] = []
    path = Path(junit_path)

    total = failures = errors = skipped = 0
    skipped_names: list[str] = []
    parsed = False

    if path.exists():
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            root = None
        if root is not None:
            parsed = True
            for case in root.iter("testcase"):
                total += 1
                if case.find("failure") is not None:
                    failures += 1
                elif case.find("error") is not None:
                    errors += 1
                if case.find("skipped") is not None:
                    skipped += 1
                    skipped_names.append(case.get("name") or "?")

    lines += ["### 用例统计", ""]
    if parsed:
        passed = total - failures - errors - skipped
        lines.append(f"总计 **{total}** 条：{passed} 通过 / {failures} 失败 / "
                     f"{errors} 错误 / {skipped} 跳过")
        if skipped_names:
            lines += ["", "跳过的用例：" + "、".join(f"`{name}`" for name in skipped_names)]
    else:
        lines.append("拿不到统计（`junit.xml` 不存在或不是合法 XML）—— 请直接看上面的测试日志。")

    lines += [""] + SCOPE_STATEMENT
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    junit = argv[0] if argv else "junit.xml"
    text = summarize(junit)

    print(text)
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print("[提示] 没有 GITHUB_STEP_SUMMARY 环境变量，只在终端打印（本地跑就是这样）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
