"""构建**医疗**评测集（12 篇中文医学文档 × 3 题）并做同样的程序化校验。

与 `build_eval_set.py` 同构（那边是 Python 文档），保留两条硬纪律：
    1. 每题的 `answer_keywords` 必须**逐字出现在来源文档里**（下面 `check()` 会拦下来）；
       否则 IR 评测的「对抗性口径」（纯词面判相关）会静默全 0。
    2. 题型（术语型 / 语义改写）由**实测的词汇重叠率**决定，不采信作者声明。

★ 与 Python 那份评测集的语料**不可混用**：换语料 = 换评测前提（见 corpus_medical/SOURCES.md）。

用法：
    .\\.venv\\Scripts\\python.exe evaluation\\build_eval_set_medical.py            # 生成 + 校验
    .\\.venv\\Scripts\\python.exe evaluation\\build_eval_set_medical.py --check     # 只校验
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

CORPUS_DIR = PROJECT_ROOT / "corpus_medical"
OUT_PATH = EVAL_DIR / "eval_set_medical.jsonl"

REWRITE_THRESHOLD = 0.6

STOPWORDS = set("""
什么 怎么 怎样 如何 哪些 哪个 为什么 是否 可以 能否 需要 应该 一个 一些 这些 那些 这个 那个
的 了 是 在 和 与 或 也 都 就 而 及 把 被 让 对 从 到 为 以 之 其 这 那 有 无 不 很 更 最 会 能
用 要 想 我 你 他 她 它 我们 你们 他们 里 中 上 下 时 时候 现在 之后 之前 分别 各自
下面 上面 以下 以上 其他 时候 方式 方法 情况 地方 内容 部分 例子 例如 比如 如果 那么
患者 病人 疾病 可能 通常 一般 常见 出现 引起 导致 造成 治疗 诊断 症状 情况 发生 使用
""".split())


def content_words(text: str) -> set[str]:
    """内容词集合：去掉标点、纯数字、单字与停用词"""
    tokens = jieba.lcut(text)
    words: set[str] = set()
    for token in tokens:
        token = token.strip()
        if not token or token in STOPWORDS:
            continue
        if re.fullmatch(r"[\W_]+", token):
            continue
        if re.fullmatch(r"\d+", token):
            continue
        if len(token) == 1 and not token.isascii():
            continue
        if len(token) <= 1:
            continue
        words.add(token.lower())
    return words


def overlap_ratio(question: str, document: str) -> float:
    question_words = content_words(question)
    if not question_words:
        return 0.0
    document_words = content_words(document)
    return len(question_words & document_words) / len(question_words)


# --------------------------------------------------------------------------- 评测集
# 12 篇医疗文档 × 3 题。答案与关键词都取自对应文档的原文。
QUESTIONS: list[dict] = [
    # ---- 01 高血压 ----
    dict(id="htn-01", source="01_hypertension.txt",
         question="高血压可分为哪两类？各自在全部病例中大约占多少比例？",
         ground_truth="高血压可分为原发性高血压和继发性高血压，其中约 90-95% 为原发性高血压，其余 5-10% 为继发性高血压。",
         tags=["术语型"], answer_keywords=["原发性高血压", "继发性高血压", "90-95%"]),
    dict(id="htn-02", source="01_hypertension.txt",
         question="成年人的血压高到什么程度就算是得了高血压？",
         ground_truth="成人若血压持续超过 130/80 或 140/90 毫米汞柱（收缩压/舒张压），即有可能是高血压。",
         tags=["语义改写"], answer_keywords=["130/80", "收缩压", "舒张压"]),
    dict(id="htn-03", source="01_hypertension.txt",
         question="SPRINT 研究对把血压控制目标降到多少毫米汞柱得出了结论？",
         ground_truth="美国 SPRINT 研究认为应将血压控制标准降至 120 mmHg。",
         tags=["术语型"], answer_keywords=["SPRINT", "120 mmHg"]),

    # ---- 02 2 型糖尿病 ----
    dict(id="dm2-01", source="02_type2_diabetes.txt",
         question="2 型糖尿病过去（旧称）被称为什么？",
         ground_truth="2 型糖尿病旧称非胰岛素依赖型糖尿病（NIDDM），也称成人发病型糖尿病。",
         tags=["术语型"], answer_keywords=["非胰岛素依赖型糖尿病", "NIDDM", "成人发病型"]),
    dict(id="dm2-02", source="02_type2_diabetes.txt",
         question="空腹抽血时血糖高到多少，就可以诊断为糖尿病？",
         ground_truth="空腹血糖 ≥7.0 mmol/l（126 mg/dl）即达到糖尿病的诊断标准。",
         tags=["语义改写"], answer_keywords=["7.0 mmol/l", "126 mg/dl", "空腹血糖"]),
    dict(id="dm2-03", source="02_type2_diabetes.txt",
         question="通常被作为 2 型糖尿病第一线治疗的药物是什么？",
         ground_truth="由于有证据表明二甲双胍可降低死亡率，通常将其作为 2 型糖尿病的第一线治疗药物。",
         tags=["术语型"], answer_keywords=["二甲双胍", "第一线"]),

    # ---- 03 肺炎 ----
    dict(id="pneu-01", source="03_pneumonia.txt",
         question="十九世纪的医生威廉·奥斯勒把肺炎称为什么？",
         ground_truth="肺炎被威廉·奥斯勒视为「死亡统帅」。",
         tags=["术语型"], answer_keywords=["威廉·奥斯勒", "死亡统帅"]),
    dict(id="pneu-02", source="03_pneumonia.txt",
         question="得了肺炎的人咳出的痰液可能有什么特殊的颜色或形状？",
         ground_truth="肺炎链球菌引起的肺炎可能带有锈色的痰液，而克雷伯氏肺炎菌引发的肺炎则可能产生类似「黑醋栗果冻」的血痰。",
         tags=["语义改写"], answer_keywords=["锈色", "黑醋栗果冻", "肺炎链球菌"]),
    dict(id="pneu-03", source="03_pneumonia.txt",
         question="临床上常用哪个分数来判断成年肺炎患者是否需要住院？",
         ground_truth="CURB-65 分数可以用来判断成年人肺炎患者是否需要住院。",
         tags=["术语型"], answer_keywords=["CURB-65", "住院"]),

    # ---- 04 贫血 ----
    dict(id="anemia-01", source="04_anemia.txt",
         question="临床上最常用来作为贫血指标的检测项目是什么？",
         ground_truth="临床上最常应用血红蛋白浓度作为贫血指标，红细胞比容次之。",
         tags=["术语型"], answer_keywords=["血红蛋白浓度", "红细胞比容"]),
    dict(id="anemia-02", source="04_anemia.txt",
         question="根据红细胞个头的大小，贫血可以分成哪几种？",
         ground_truth="依红细胞形态（主要参考平均红细胞体积）可分为大细胞性贫血、正细胞性贫血、小细胞性贫血。",
         tags=["语义改写"], answer_keywords=["平均红细胞体积", "大细胞性贫血", "小细胞性贫血"]),
    dict(id="anemia-03", source="04_anemia.txt",
         question="因缺乏内在因子导致维生素 B12 吸收不足而引起的贫血叫什么？",
         ground_truth="缺乏内在因子使维生素 B12 吸收不足所造成的缺乏又称恶性贫血。",
         tags=["术语型"], answer_keywords=["内在因子", "恶性贫血", "维生素B12"]),

    # ---- 05 支气管哮喘 ----
    dict(id="asthma-01", source="05_asthma.txt",
         question="哮喘属于哪一型超敏反应性疾病，其主要特征是什么？",
         ground_truth="哮喘是与气道高反应性相关的慢性非特异性炎症性疾病，属 I 型超敏反应性疾病。",
         tags=["术语型"], answer_keywords=["气道高反应性", "非特异性炎症", "I型超敏反应"]),
    dict(id="asthma-02", source="05_asthma.txt",
         question="医生用什么检查手段可以确诊哮喘？",
         ground_truth="呼吸量测定法是诊断哮喘的最佳测试法；吸入支气管扩张剂如沙丁胺醇后 FEV1 改善超过 12%，可支持哮喘诊断。",
         tags=["语义改写"], answer_keywords=["呼吸量测定法", "沙丁胺醇", "FEV1"]),
    dict(id="asthma-03", source="05_asthma.txt",
         question="被认为对哮喘病长期控制最有效的治疗是什么？",
         ground_truth="类固醇通常被认为是对哮喘病长期控制的最有效的治疗，一般会采用吸入的形式。",
         tags=["术语型"], answer_keywords=["类固醇", "长期控制", "吸入"]),

    # ---- 06 重性抑郁障碍 ----
    dict(id="mdd-01", source="06_depression.txt",
         question="诊断重性抑郁发作要求抑郁心境至少持续多久？",
         ground_truth="一次重性抑郁发作表现为严重的心境抑郁且持续两周以上。",
         tags=["术语型"], answer_keywords=["两周", "心境抑郁"]),
    dict(id="mdd-02", source="06_depression.txt",
         question="医生给抑郁症患者开药时，最常开哪一类抗抑郁药？",
         ground_truth="选择性 5-羟色胺再摄取抑制剂（SSRIs，例如舍曲林、氟西汀）往往是处方中主要的药物。",
         tags=["语义改写"], answer_keywords=["选择性5-羟色胺再摄取抑制剂", "舍曲林", "氟西汀"]),
    dict(id="mdd-03", source="06_depression.txt",
         question="研究发现人体内哪个基因会影响人在应对应激事件时是否持续抑郁？",
         ground_truth="人体内 5-羟色胺转运体（5-HTT）基因影响到人们是否在应对应激事件时持续体验到抑郁。",
         tags=["术语型"], answer_keywords=["5-羟色胺转运体", "5-HTT"]),

    # ---- 07 冠心病 ----
    dict(id="chd-01", source="07_coronary_heart_disease.txt",
         question="冠心病的型态包含哪几种？",
         ground_truth="冠心病型态包含稳定型心绞痛、非稳定型心绞痛、心肌梗塞和猝死。",
         tags=["术语型"], answer_keywords=["稳定型心绞痛", "心肌梗塞", "猝死"]),
    dict(id="chd-02", source="07_coronary_heart_disease.txt",
         question="冠心病突然发作时，应立即含服哪种急救药？",
         ground_truth="冠心病发作时须立即口含医师处方的硝酸甘油，其可扩张血管、增加冠状动脉血流量。",
         tags=["语义改写"], answer_keywords=["硝酸甘油", "冠状动脉血流量"]),
    dict(id="chd-03", source="07_coronary_heart_disease.txt",
         question="双联抗血小板治疗（DAPT）指的是哪类药物治疗？",
         ground_truth="双联抗血小板治疗（DAPT）是阿司匹林联合一种 P2Y12 受体拮抗剂的药物治疗。",
         tags=["术语型"], answer_keywords=["DAPT", "P2Y12", "阿司匹林"]),

    # ---- 08 脑卒中 ----
    dict(id="stroke-01", source="08_stroke.txt",
         question="中风按发病机制可分为哪两种类型？",
         ground_truth="一种是由血管阻塞所造成的缺血性脑中风，一种是由出血所造成的出血性脑中风。",
         tags=["术语型"], answer_keywords=["缺血性脑中风", "出血性脑中风"]),
    dict(id="stroke-02", source="08_stroke.txt",
         question="怀疑身边有人中风时，可以按什么口诀快速自查？",
         ground_truth="可用 FAST 测试（脸臂说时），台湾口诀为「微笑、举手、说你好」。",
         tags=["语义改写"], answer_keywords=["FAST", "微笑", "举手"]),
    dict(id="stroke-03", source="08_stroke.txt",
         question="缺血性中风患者最慢在发病后几小时内还可以接受静脉血栓溶解治疗？",
         ground_truth="缺血性中风患者若无禁忌症，最慢 4.5 小时内可接受静脉血栓溶解治疗。",
         tags=["术语型"], answer_keywords=["4.5小时", "静脉血栓溶解"]),

    # ---- 09 骨质疏松症 ----
    dict(id="osteo-01", source="09_osteoporosis.txt",
         question="诊断骨质疏松症的金标准是什么检测方法？",
         ground_truth="双能 X 线吸收法（DXA，DEXA）骨密度检测是诊断骨质疏松症的金标准。",
         tags=["术语型"], answer_keywords=["双能X线吸收法", "金标准"]),
    dict(id="osteo-02", source="09_osteoporosis.txt",
         question="骨密度低到什么程度就可以诊断为骨质疏松？",
         ground_truth="测得骨密度低于青年人的标准值 2.5 个标准方差值（T 值小于 -2.5），即可诊断为骨质疏松症。",
         tags=["语义改写"], answer_keywords=["T值", "骨密度", "-2.5"]),
    dict(id="osteo-03", source="09_osteoporosis.txt",
         question="骨质疏松症使哪些骨骼部位最容易发生骨折？",
         ground_truth="易于因骨质疏松而骨折的骨骼部位有脊椎、前臂骨、髋关节骨。",
         tags=["术语型"], answer_keywords=["脊椎", "前臂骨", "髋关节骨"]),

    # ---- 10 甲状腺功能减退 ----
    dict(id="hypo-01", source="10_hypothyroidism.txt",
         question="世界上造成甲状腺功能减退症最常见的病因是什么？",
         ground_truth="世界上食物中的碘缺乏是最常见造成甲状腺功能减退症的主要病因；在碘充足的国家则多为桥本氏甲状腺炎。",
         tags=["术语型"], answer_keywords=["碘缺乏", "桥本氏甲状腺炎"]),
    dict(id="hypo-02", source="10_hypothyroidism.txt",
         question="甲减用什么药物来补充激素、治疗疾病？",
         ground_truth="用左旋甲状腺素钠来补充甲状腺激素可以治疗甲状腺功能减退症。",
         tags=["语义改写"], answer_keywords=["左旋甲状腺素", "甲状腺激素"]),
    dict(id="hypo-03", source="10_hypothyroidism.txt",
         question="怀疑甲减时，目前认为最好的初步检测方法是什么？",
         ground_truth="目前认为甲减最好的初步检测方法是通过实验室测量血液中的促甲状腺激素（TSH）水平。",
         tags=["术语型"], answer_keywords=["促甲状腺激素", "TSH"]),

    # ---- 11 胃炎 ----
    dict(id="gast-01", source="11_gastritis.txt",
         question="被公认的慢性胃炎主要病因是什么？",
         ground_truth="幽门螺杆菌感染被认为是慢性胃炎的主要病因。",
         tags=["术语型"], answer_keywords=["幽门螺杆菌", "慢性胃炎"]),
    dict(id="gast-02", source="11_gastritis.txt",
         question="胃黏膜长期发炎不管，可能引起哪些严重后果？",
         ground_truth="胃炎的并发症可能有出血、胃溃疡、胃癌等；若引发自身免疫导致缺乏维生素 B12，可出现恶性贫血。",
         tags=["语义改写"], answer_keywords=["胃溃疡", "胃癌", "恶性贫血"]),
    dict(id="gast-03", source="11_gastritis.txt",
         question="若检测发现幽门螺旋杆菌，建议合并使用哪两种抗生素治疗？",
         ground_truth="如果检测发现幽门螺旋杆菌，建议合并使用阿莫西林和克拉霉素。",
         tags=["术语型"], answer_keywords=["阿莫西林", "克拉霉素"]),

    # ---- 12 荨麻疹 ----
    dict(id="urt-01", source="12_urticaria.txt",
         question="荨麻疹的急性与慢性如何区分？",
         ground_truth="急性荨麻疹一般只发作 1 至 3 天；慢性荨麻疹则指症状持续 6 周以上、可能反复发生。",
         tags=["术语型"], answer_keywords=["急性荨麻疹", "慢性荨麻疹", "6周"]),
    dict(id="urt-02", source="12_urticaria.txt",
         question="荨麻疹发作、身上又红又痒时，通常吃什么药来止痒？",
         ground_truth="通常会使用抗组织胺药；现今更倾向使用第二代抗组织胺药，例如非索非那定、乐雷他定和西替利嗪。",
         tags=["语义改写"], answer_keywords=["抗组织胺药", "西替利嗪", "非索非那定"]),
    dict(id="urt-03", source="12_urticaria.txt",
         question="因为抓挠或抚摸皮肤而在皮肤上出现风疹块、条痕的现象叫什么？",
         ground_truth="这叫皮肤划痕症，是「物理性荨麻疹」子集中最常见的形式。",
         tags=["术语型"], answer_keywords=["皮肤划痕症", "物理性荨麻疹"]),
]


def check(items: list[dict]) -> tuple[list[dict], list[str]]:
    """计算词汇重叠率、判定题型、校验关键词确实在来源文档里"""
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
            "tags": [measured_type],
            "relevant_sources": [item["source"]],
            "answer_keywords": item["answer_keywords"],
            "question_type": measured_type,
            "question_type_authored": item["tags"][0],
            "question_source_overlap": round(ratio, 4),
        })
    return enriched, problems


def main() -> int:
    parser = argparse.ArgumentParser(description="构建医疗评测集并度量语义改写比例")
    parser.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = parser.parse_args()

    enriched, problems = check(QUESTIONS)

    print("=" * 96)
    print(f"{'id':<11}{'来源文档':<32}{'重叠率':>8}  {'声明':<10}{'实测':<10}")
    print("-" * 96)
    mismatched = 0
    for row in enriched:
        same = row["question_type_authored"] == row["question_type"]
        mismatched += 0 if same else 1
        flag = "" if same else "  <- 与作者意图不同（以实测为准）"
        print(f"{row['id']:<11}{row['source']:<32}{row['question_source_overlap']:>8.2f}  "
              f"{row['question_type_authored']:<10}{row['question_type']:<10}{flag}")
    print("=" * 96)

    rewritten = [r for r in enriched if r["question_type"] == "语义改写"]
    print(f"[评测集] 共 {len(enriched)} 题，覆盖 {len({r['source'] for r in enriched})} 篇文档")
    if enriched:
        print(f"[评测集] 实测语义改写（与来源文档内容词重叠 < {REWRITE_THRESHOLD}）："
              f"{len(rewritten)} 题，占 {len(rewritten) / len(enriched) * 100:.1f}%")
    print(f"[评测集] 其中 {mismatched} 题的实测类型与作者意图不同（作者意图仅作留痕）")

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
