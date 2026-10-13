"""端到端问答冒烟：真检索 + 真生成，问几条，断言「有答案、且答案带依据」。

为什么是单独脚本、不进 pytest / 不进 CI（与「IR / RAGAS 不进 CI」同一个理由）：
    它需要**真实生成模型**（DeepSeek 或 DashScope Key）与**本地嵌入服务**，还要联网 ——
    做成 CI 门禁会随机变红。这里是「发版前手动跑一次」的入口。

它校验的是 pytest 那 119 条**测不到**的东西：整条链路（检索 → 提示词 → 生成）真的能出答案，
而不是「模块能 import、RagService 能构造」（那是 `test_import_smoke.py`）。

用法（默认 DeepSeek 生成 + 本地嵌入，不需要 DashScope）：

    $env:CHAT_BACKEND='openai-compatible'
    .\\.venv\\Scripts\\python.exe scripts\\e2e_smoke.py                 # 默认问前 3 条
    .\\.venv\\Scripts\\python.exe scripts\\e2e_smoke.py --limit 5
    .\\.venv\\Scripts\\python.exe scripts\\e2e_smoke.py --question "身高168cm体重100斤穿什么尺码"

前置：本地嵌入服务在跑（`cd D:\\Projects ; .\\svc.ps1 start llm`）。
退出码：0 = 全部通过；1 = 有失败（断言或异常）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 两个路径都要加：PROJECT_ROOT 用来 import config_data / rag，evaluation 用来读评测集
for _path in (str(PROJECT_ROOT), str(PROJECT_ROOT / "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config_data as config                                # noqa: E402

DEFAULT_EVAL_SET = PROJECT_ROOT / "evaluation" / "eval_set_v2.jsonl"


def load_questions(path: Path, limit: int | None) -> list[dict]:
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if limit and len(items) >= limit:
                break
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="端到端问答冒烟（真检索 + 真生成；不进 CI）")
    parser.add_argument("--limit", type=int, default=3, help="问评测集里的前 N 条（默认 3）")
    parser.add_argument("--question", action="append", default=None,
                        help="自定义问题，可重复；给了它就不再读评测集")
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--min-contexts", type=int, default=1, help="每条至少要命中几段依据（默认 1）")
    parser.add_argument("--collection", default=None,
                        help="指定向量库集合（如 rag_local_corpus_c300_o30）；不指定则按嵌入后端派生命名")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖 config.retrieve_k")
    args = parser.parse_args(argv)

    # 必须在构造 RagService 之前把参数写回 config（检索器是在 RagService.__init__ 里建的）
    if args.collection:
        config.collection_override = args.collection
    if args.top_k is not None:
        config.retrieve_k = args.top_k

    if args.question:
        questions = [{"id": f"manual-{i + 1}", "question": q} for i, q in enumerate(args.question)]
    else:
        questions = load_questions(Path(args.eval_set), args.limit)
    if not questions:
        print("[E2E] 没有可问的问题（评测集为空？）")
        return 1

    print("=" * 74)
    print(f"[E2E] 生成后端={config.chat_backend}  嵌入后端={config.embedding_backend}  "
          f"策略={config.retrieval_method} k={config.retrieve_k} 候选={config.candidate_k}")
    print(f"[E2E] 重排={'开' if config.rerank_enabled else '关'}  问题数={len(questions)}")
    print(f"[E2E] 集合={config.collection_override or '（按后端派生）'}")
    print("=" * 74)

    # 生成侧 Key 必须先就位：否则构造/调用时才炸，错误信息离现场很远
    from run_ragas import require_chat_key          # 复用评测层同一套「按后端解析变量名」的逻辑
    require_chat_key()

    from rag import RagService
    rag = RagService()

    failures: list[tuple[str, str]] = []
    for item in questions:
        qid = item.get("id", "?")
        question = item.get("question", "")
        started = time.perf_counter()
        try:
            result = rag.answer(question, session_id=f"e2e_{qid}")
        except Exception as exc:                                  # 单条失败不要中断整轮
            failures.append((qid, f"抛异常：{exc}"))
            print(f"  [FAIL] {qid}  抛异常：{exc}")
            continue

        answer = (result.get("answer") or "").strip()
        contexts = result.get("contexts") or []
        elapsed = time.perf_counter() - started
        problems = []
        if not answer:
            problems.append("答案为空")
        if len(contexts) < args.min_contexts:
            problems.append(f"只命中 {len(contexts)} 段依据（<{args.min_contexts}）")
        ok = not problems
        if not ok:
            failures.append((qid, "；".join(problems)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {qid}  "
              f"答案 {len(answer)} 字 / 依据 {len(contexts)} 段 / {elapsed:.2f}s")

    print("=" * 74)
    if failures:
        print(f"[E2E] 失败 {len(failures)} 条：")
        for qid, why in failures:
            print(f"  - {qid}: {why}")
        return 1
    print(f"[E2E] 全部通过（{len(questions)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
