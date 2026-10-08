"""构建新评测集（12 篇 Python 中文文档）+ **可度量**的"语义改写"比例

为什么要有这个脚本，而不是直接手写 eval_set.jsonl：

    ★ ground truth 的构造方式决定结论可信度，必须留痕。
    如果问题直接从原文抄词，BM25 会天然占优，向量检索的优势被抹掉，结论反向失真。
    所以这里把"问题与原文的词汇重叠"变成**可计算的量**，而不是靠自我声明。

两种题型的**可操作定义**（由脚本计算，不靠人工标注）：

    术语型      问题与来源文档的内容词重叠率 ≥ 0.6
    语义改写型  问题与来源文档的内容词重叠率 < 0.6（即：用同义说法提问，刻意避开原词）

内容词的取法：jieba 分词后，去掉标点/纯数字/单字/停用词。

用法：
    .\\.venv\\Scripts\\python.exe evaluation\\build_eval_set.py            # 生成 + 打印统计
    .\\.venv\\Scripts\\python.exe evaluation\\build_eval_set.py --check     # 只校验不写文件
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jieba                                                       # noqa: E402

CORPUS_DIR = PROJECT_ROOT / "corpus"
OUT_PATH = EVAL_DIR / "eval_set_v2.jsonl"

REWRITE_THRESHOLD = 0.6      # 重叠率低于此值 → 归为"语义改写型"

STOPWORDS = set("""
什么 怎么 怎样 如何 哪些 哪个 为什么 是否 可以 能否 需要 应该 一个 一些 这些 那些 这个 那个
的 了 是 在 和 与 或 也 都 就 而 及 把 被 让 对 从 到 为 以 之 其 这 那 有 无 不 很 更 最 会 能
用 要 想 我 你 他 她 它 我们 你们 他们 它 里 中 上 下 时 时候 现在 之后 之前 分别 各自
下面 上面 以下 以上 其他 什么 时候 方式 方法 情况 地方 内容 部分 例子 例如 比如 如果 那么
""".split())


def content_words(text: str) -> set[str]:
    """内容词集合：去掉标点、纯数字、单字与停用词

    ★ 英文标识符（如 dumps / pathlib）必须保留 —— Python 文档里它们是最强的检索信号，
      只统计中文会高估"语义改写"比例。
    """
    tokens = jieba.lcut(text)
    words: set[str] = set()
    for token in tokens:
        token = token.strip()
        if not token or token in STOPWORDS:
            continue
        if re.fullmatch(r"[\W_]+", token):          # 纯标点
            continue
        if re.fullmatch(r"\d+", token):             # 纯数字
            continue
        if len(token) == 1 and not token.isascii():  # 单个汉字区分度太低
            continue
        if len(token) <= 1:
            continue
        words.add(token.lower())
    return words


def overlap_ratio(question: str, document: str) -> float:
    """问题的内容词中，有多少比例出现在来源文档里"""
    question_words = content_words(question)
    if not question_words:
        return 0.0
    document_words = content_words(document)
    return len(question_words & document_words) / len(question_words)


# --------------------------------------------------------------------------- 评测集
# 12 篇文档 × 3 题。每题都给了 answer_keywords，用于**程序化校验**答案确实在来源文档里。
QUESTIONS: list[dict] = [
    # ---- 01 Python 入门 ----
    dict(id="intro-01", source="01_python_intro.txt",
         question="Python 里 / 和 // 两个除号得到的结果有什么不一样？",
         ground_truth="用 / 做除法得到的永远是浮点数；用 // 做整除会向下取整，结果可能是整数。",
         tags=["术语型"], answer_keywords=["除法", "浮点数", "取整"]),
    dict(id="intro-02", source="01_python_intro.txt",
         question="字符串字面量前面加一个 r 有什么作用？",
         ground_truth="表示原始字符串，反斜杠不再被当成转义字符，例如 r'\\n' 里就是反斜杠和字母 n 两个字符。",
         tags=["术语型"], answer_keywords=["原始字符串", "转义"]),
    dict(id="intro-03", source="01_python_intro.txt",
         question="我想让一段文字占到好几行，不用一行行拼，能怎么表示？",
         ground_truth="用三重引号把文字括起来（\"\"\"...\"\"\" 或 '''...'''），行末的换行会自动包含在字符串里。",
         tags=["语义改写"], answer_keywords=["三重引号", "换行"]),

    # ---- 02 流程控制 ----
    dict(id="flow-01", source="02_controlflow.txt",
         question="range() 和 enumerate() 在 for 循环里分别解决什么问题？",
         ground_truth="range() 用来生成一串数字序列；enumerate() 在遍历序列的同时给出元素的下标位置。",
         tags=["术语型"], answer_keywords=["range", "enumerate"]),
    # 注：原拟问 *args/**kwargs，但实测该文档没有 kwargs 相关内容（0 次命中），
    #     已改为该文档确实覆盖的「可变默认值陷阱」—— 这正是答案关键词校验拦住的问题
    dict(id="flow-02", source="02_controlflow.txt",
         question="给函数参数设默认值时，如果默认值本身是可变的（比如列表），会有什么问题？怎么避免？",
         ground_truth="默认值只在函数定义时求值一次，所以多次调用会共享同一个列表、结果互相累积；"
                      "做法是把默认值写成 None，进入函数后再判断并新建列表。",
         tags=["术语型"], answer_keywords=["默认值", "None"]),
    dict(id="flow-03", source="02_controlflow.txt",
         question="定义函数的时候，怎么规定某个入参只能按顺序给、另一个只能用名字给？",
         ground_truth="在只能按顺序给的参数后面加一个斜杠 /，在只能用名字给的参数前面加一个星号 *。",
         tags=["语义改写"], answer_keywords=["/", "*"]),

    # ---- 03 数据结构 ----
    dict(id="ds-01", source="03_datastructures.txt",
         question="列表推导式的写法是怎样的？能举个例子吗？",
         ground_truth="形如 [表达式 for 元素 in 序列 if 条件]，例如 [x*2 for x in squares] 会对每个元素乘 2 生成新列表。",
         tags=["术语型"], answer_keywords=["列表推导式", "for"]),
    dict(id="ds-02", source="03_datastructures.txt",
         question="列表和元组有什么区别？",
         ground_truth="列表是可变序列，可以增删改元素；元组是不可变序列，创建后不能修改，可以包含不同类型的元素。",
         tags=["术语型"], answer_keywords=["列表", "元组", "不可变"]),
    dict(id="ds-03", source="03_datastructures.txt",
         question="手上有两个键值对表，想把它们并成一份，有什么简洁的做法？",
         ground_truth="用竖线运算符 | 合并两个字典，得到包含两边全部键值的新字典，例如 a | b。",
         tags=["语义改写"], answer_keywords=["|"]),

    # ---- 04 模块 ----
    dict(id="mod-01", source="04_modules.txt",
         question="if __name__ == '__main__': 这行判断有什么用？",
         ground_truth="让这段代码只在文件被直接运行时执行；当文件被当作模块导入时不会执行。",
         tags=["术语型"], answer_keywords=["__name__", "__main__"]),
    dict(id="mod-02", source="04_modules.txt",
         question="包目录里的 __init__.py 起什么作用？",
         ground_truth="它让 Python 把这个目录当成包；也可以在里面写包的初始化代码，或者用 __all__ 控制 from 包 import * 的行为。",
         tags=["术语型"], answer_keywords=["__init__", "包"]),
    dict(id="mod-03", source="04_modules.txt",
         question="不想把整个文件的东西都引进来，只想要里面某一个函数，可以怎么写？",
         ground_truth="用 from 模块名 import 函数名 的形式只导入需要的那个名字。",
         tags=["语义改写"], answer_keywords=["from", "import"]),

    # ---- 05 输入输出 ----
    dict(id="io-01", source="05_inputoutput.txt",
         question="open() 的常用模式参数有哪些？分别是什么含义？",
         ground_truth="r 只读（默认）、w 只写（覆盖原内容）、a 追加、b 二进制模式、+ 同时读写。",
         tags=["术语型"], answer_keywords=["r", "w", "a", "b"]),
    dict(id="io-02", source="05_inputoutput.txt",
         question="用 with 语句打开文件有什么好处？",
         ground_truth="文件用完会自动关闭，即使中途出现异常也会正确关闭，不需要手动调用 close()。",
         tags=["术语型"], answer_keywords=["with", "异常"]),
    dict(id="io-03", source="05_inputoutput.txt",
         question="我想把一系列数字存进磁盘上的文件、而不是打到屏幕上，该怎么做？",
         ground_truth="用 open() 打开文件拿到文件对象，再把数字转成字符串后用 write() 写进去；write 需要的是字符串而不是数字。",
         tags=["语义改写"], answer_keywords=["write", "open"]),

    # ---- 06 错误与异常 ----
    dict(id="err-01", source="06_errors.txt",
         question="try 语句里的 except、else、finally 分别在什么时候执行？",
         ground_truth="try 块正常结束时会执行 else 块；出现异常时执行匹配的 except 块；无论是否发生异常，finally 块最后都会执行。",
         tags=["术语型"], answer_keywords=["except", "else", "finally"]),
    dict(id="err-02", source="06_errors.txt",
         question="raise ... from ... 这种写法的作用是什么？",
         ground_truth="在抛出异常时显式指定它的原因（原始异常），便于追溯问题源头的调用链。",
         tags=["术语型"], answer_keywords=["raise", "from"]),
    dict(id="err-03", source="06_errors.txt",
         question="程序拿到不认识的输入时，我想主动中断并给出提示，该怎么写？",
         ground_truth="用 raise 抛出一个异常对象（例如 raise ValueError('提示信息')），调用方可以用 try/except 接住处理。",
         tags=["语义改写"], answer_keywords=["raise", "异常"]),

    # ---- 07 类 ----
    dict(id="cls-01", source="07_classes.txt",
         question="类变量和实例变量有什么区别？",
         ground_truth="类变量由所有实例共享，只在类里定义一次；实例变量属于具体对象，通常在 __init__ 里通过 self 赋值，互不影响。",
         tags=["术语型"], answer_keywords=["类变量", "实例变量", "self"]),
    dict(id="cls-02", source="07_classes.txt",
         question="子类里调用 super() 是做什么的？",
         ground_truth="调用父类（超类）的实现，常用于在子类重写方法后仍然复用父类的行为，例如 super().__init__()。",
         tags=["术语型"], answer_keywords=["super", "父类"]),
    dict(id="cls-03", source="07_classes.txt",
         question="不想让外面的代码随便改对象内部的数据，有什么约定或做法？",
         ground_truth="按约定把只给内部用的名字前面加一个下划线（如 _spam），把它当成私有实现细节，外部不要直接访问。",
         tags=["语义改写"], answer_keywords=["下划线", "_"]),

    # ---- 08 标准库概览 ----
    dict(id="std-01", source="08_stdlib_overview.txt",
         question="标准库概览里，日期和时间分别用哪个模块处理？",
         ground_truth="datetime 模块用于日期和时间的运算与格式化；time 模块用于获取时间戳、访问底层时间相关函数。",
         tags=["术语型"], answer_keywords=["datetime", "time"]),
    dict(id="std-02", source="08_stdlib_overview.txt",
         question="想从一串候选里随机抽几个，用标准库的哪个模块？",
         ground_truth="random 模块提供随机选择功能，例如 random.choice() 从序列里随机取一个元素。",
         tags=["术语型"], answer_keywords=["random", "choice"]),
    dict(id="std-03", source="08_stdlib_overview.txt",
         question="想按通配符把一堆文件筛出来，标准库里有什么现成工具？",
         ground_truth="glob 模块提供按 Unix 通配符风格（如 *.py）查找文件名列表的函数。",
         tags=["语义改写"], answer_keywords=["glob"]),

    # ---- 09 json ----
    dict(id="json-01", source="09_json.txt",
         question="json.dumps() 和 json.dump() 有什么区别？",
         ground_truth="dumps() 把对象序列化成字符串返回；dump() 直接把对象写进一个文件对象里。",
         tags=["术语型"], answer_keywords=["dumps", "dump"]),
    dict(id="json-02", source="09_json.txt",
         question="序列化出来的中文变成了 \\uXXXX，怎么让它保持原样？",
         ground_truth="把 ensure_ascii 参数设为 False，输出就会保留中文原字符。",
         tags=["术语型"], answer_keywords=["ensure_ascii"]),
    dict(id="json-03", source="09_json.txt",
         question="从网络收到一段文本，怎么把它变成 Python 里能直接用的字典或列表？",
         ground_truth="用 json.loads() 把符合 JSON 格式的字符串解析成 Python 对象（字典或列表）。",
         tags=["语义改写"], answer_keywords=["loads"]),

    # ---- 10 sqlite3 ----
    dict(id="sql-01", source="10_sqlite3.txt",
         question="用 sqlite3 查询数据库的基本步骤是什么？",
         ground_truth="先用 sqlite3.connect() 建立连接、用 connection.cursor() 创建游标，再执行 SQL 并取回结果。",
         tags=["术语型"], answer_keywords=["connect", "cursor"]),
    dict(id="sql-02", source="10_sqlite3.txt",
         question="为什么推荐用问号占位符传参，而不是把值拼进 SQL 字符串？",
         ground_truth="占位符传参能防止 SQL 注入；拼接字符串既容易注入，也可能因为引号转义出错。",
         tags=["术语型"], answer_keywords=["占位符", "注入"]),
    dict(id="sql-03", source="10_sqlite3.txt",
         question="要一次把一批数据写进表里，而不是一条一条来，用哪个方法？",
         ground_truth="用 cursor.executemany()，把一批参数一次性交给同一条 SQL 语句执行。",
         tags=["语义改写"], answer_keywords=["executemany"]),

    # ---- 11 pathlib ----
    dict(id="path-01", source="11_pathlib.txt",
         question="Path 对象怎么表示子路径或拼接路径？",
         ground_truth="用斜杠运算符 / 拼接，例如 Path('/etc') / 'passwd'；它会按当前系统自动使用合适的分隔符。",
         tags=["术语型"], answer_keywords=["Path", "/"]),
    dict(id="path-02", source="11_pathlib.txt",
         question="Path.glob() 和 Path.rglob() 有什么区别？",
         ground_truth="glob() 只在当前目录层级按模式匹配；rglob() 会递归进所有子目录匹配，等价于用 **/ 的模式。",
         tags=["术语型"], answer_keywords=["glob", "rglob", "递归"]),
    dict(id="path-03", source="11_pathlib.txt",
         question="想知道某个路径上到底存不存在东西，跨平台该怎么判断？",
         ground_truth="用 Path 对象的 exists() 方法判断路径是否存在；它不依赖具体操作系统，比手工拼字符串更稳。",
         tags=["语义改写"], answer_keywords=["exists"]),

    # ---- 12 re ----
    dict(id="re-01", source="12_regex.txt",
         question="re.match() 和 re.search() 有什么区别？",
         ground_truth="match() 只从字符串开头尝试匹配；search() 会扫描整个字符串，找到第一处匹配就返回。",
         tags=["术语型"], answer_keywords=["match", "search"]),
    dict(id="re-02", source="12_regex.txt",
         question="re.findall() 返回的是什么？",
         ground_truth="返回所有不重叠匹配组成的列表；如果模式里有分组，返回的是分组内容组成的列表。",
         tags=["术语型"], answer_keywords=["findall", "列表"]),
    dict(id="re-03", source="12_regex.txt",
         question="从一段文字里把所有形如 年-月-日 的内容都捞出来，该怎么写？",
         ground_truth="用 re.findall() 配合一个匹配四位数字、短横线、两位数字、短横线、两位数字的模式（如 \\d{4}-\\d{2}-\\d{2}）。",
         tags=["语义改写"], answer_keywords=["findall", "\\d"]),
]


def check(items: list[dict]) -> tuple[list[dict], list[str]]:
    """计算每题的词汇重叠率、判定题型、校验答案关键词确实在来源文档里"""
    problems: list[str] = []
    enriched: list[dict] = []

    for item in items:
        doc_path = CORPUS_DIR / item["source"]
        if not doc_path.exists():
            problems.append(f"{item['id']}: 语料文件不存在 {item['source']}")
            continue
        document = doc_path.read_text(encoding="utf-8")
        ratio = overlap_ratio(item["question"], document)
        measured_type = "语义改写" if ratio < REWRITE_THRESHOLD else "术语型"

        missing = [kw for kw in item["answer_keywords"] if kw not in document]
        if missing:
            problems.append(f"{item['id']}: 答案关键词不在来源文档里 -> {missing}")

        enriched.append({
            "id": item["id"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "source": item["source"],
            # ★ tags / 题型一律以**实测**为准（作者意图只作留痕，见 question_type_authored）
            "tags": [measured_type],
            "relevant_sources": [item["source"]],
            "question_type": measured_type,
            "question_type_authored": item["tags"][0],
            "question_source_overlap": round(ratio, 4),
        })
    return enriched, problems


def main() -> int:
    parser = argparse.ArgumentParser(description="构建评测集并度量语义改写比例")
    parser.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = parser.parse_args()

    enriched, problems = check(QUESTIONS)

    print("=" * 96)
    print(f"{'id':<10}{'来源文档':<24}{'重叠率':>8}  {'声明':<10}{'实测':<10}")
    print("-" * 96)
    mismatched = 0
    for row in enriched:
        same = row["question_type_authored"] == row["question_type"]
        mismatched += 0 if same else 1
        flag = "" if same else "  <- 与作者意图不同（以实测为准）"
        print(f"{row['id']:<10}{row['source']:<24}{row['question_source_overlap']:>8.2f}  "
              f"{row['question_type_authored']:<10}{row['question_type']:<10}{flag}")
    print("=" * 96)

    rewritten = [r for r in enriched if r["question_type"] == "语义改写"]
    print(f"[评测集] 共 {len(enriched)} 题，覆盖 {len({r['source'] for r in enriched})} 篇文档")
    print(f"[评测集] 实测语义改写（与来源文档内容词重叠 < {REWRITE_THRESHOLD}）："
          f"{len(rewritten)} 题，占 {len(rewritten) / len(enriched) * 100:.1f}%")
    print(f"[评测集] 其中 {mismatched} 题的实测类型与作者意图不同（作者意图仅作留痕，"
          f"下游一律用实测类型）")
    print("[评测集] 口径提示：重叠率是拿问题去比**整篇文档**，文档越长越容易命中，"
          "因此本比例是**保守估计**（真实改写比例不低于此值）")

    if problems:
        print(f"\n[校验] 发现 {len(problems)} 个问题：")
        for problem in problems:
            print("  -", problem)
    else:
        print("[校验] 通过：所有答案关键词都能在对应的来源文档里找到")

    if args.check:
        return 0 if not problems else 1

    OUT_PATH.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in enriched) + "\n",
        encoding="utf-8",
    )
    print(f"\n[评测集] 已写入 {OUT_PATH}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
