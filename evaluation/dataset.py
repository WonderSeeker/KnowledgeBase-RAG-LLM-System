"""评测集：问题 + 标准答案（ground truth）

为什么评测集只需要「问题 + 标准答案」两列：
    RAGAS 的 4 个指标里，
      - faithfulness / answer_relevancy 只看「模型答案 + 检索片段」；
      - context_recall / context_precision 需要「标准答案」来判断检索片段的覆盖度与纯度。
    检索片段和模型答案都在跑评测时由 RAG 链路现场产出，所以不必预先准备。

数据来源：本项目 assets/ 下 3 份中文语料（尺码推荐 / 洗涤养护 / 颜色推荐），共 29 条，
覆盖 4 类问题：精确数值查询、单点事实、跨段落归纳、否定性提问（"能不能/禁不禁"）。

`evaluation/eval_set.jsonl` 是权威数据（可直接手工编辑增删）；
本文件里的 BUILTIN_EVAL_SET 是它的种子，`python evaluation/dataset.py --init` 可重新生成。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
DEFAULT_EVAL_SET = EVAL_DIR / "eval_set.jsonl"


@dataclass
class EvalItem:
    """一条评测样本"""

    id: str
    question: str
    ground_truth: str
    source: str                      # 标准答案出自哪个语料文件
    tags: list[str] = field(default_factory=list)   # 问题类型标签，便于分难度看指标

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalItem":
        return cls(
            id=data["id"],
            question=data["question"],
            ground_truth=data["ground_truth"],
            source=data.get("source", ""),
            tags=list(data.get("tags") or []),
        )


def _item(idx: str, question: str, ground_truth: str, source: str, tags: list[str]) -> EvalItem:
    return EvalItem(id=idx, question=question, ground_truth=ground_truth, source=source, tags=tags)


# --------------------------------------------------------------------------- 内置评测集
BUILTIN_EVAL_SET: list[EvalItem] = [
    # ---------------- 尺码推荐.txt：精确数值查询（表格类数据最容易暴露「检索不全」问题）----------------
    _item("size-01", "身高168cm、体重100斤，建议穿什么尺码？",
          "建议M码。M码对应身高160-170cm、体重90-115斤。", "尺码推荐.txt", ["精确数值"]),
    _item("size-02", "身高176cm、体重140斤选哪个尺码？",
          "建议XL码。XL对应身高170-178cm、体重130-150斤。", "尺码推荐.txt", ["精确数值"]),
    _item("size-03", "身高192cm、体重220斤穿什么尺码？",
          "建议5XL。5XL对应身高190cm以上、体重210斤以上。", "尺码推荐.txt", ["精确数值"]),
    _item("size-04", "身高162cm、体重90斤，尺码怎么选？",
          "建议S码。S码对应身高155-165cm、体重75-95斤。", "尺码推荐.txt", ["精确数值"]),
    _item("size-05", "体重170斤、身高182cm，建议尺码？",
          "建议3XL。3XL对应身高178-185cm、体重160-180斤。", "尺码推荐.txt", ["精确数值"]),

    # ---------------- 洗涤养护.txt：单点事实 / 跨段落归纳 / 否定性提问 ----------------
    _item("wash-01", "真丝连衣裙应该怎么洗？",
          "建议干洗；手洗用真丝专用中性洗涤剂，水温不超过25℃，浸泡不超过15分钟，轻轻按压清洗，禁止搓揉、拧绞、漂白。",
          "洗涤养护.txt", ["跨段落归纳"]),
    _item("wash-02", "羊毛衫手洗时水温最高多少度？",
          "手洗水温不超过20℃，浸泡不超过15分钟，轻轻按压，禁止机洗、搓揉、拧绞。",
          "洗涤养护.txt", ["精确数值"]),
    _item("wash-03", "羽绒服怎么洗涤和晾晒？",
          "优先干洗；可水洗款用羽绒服专用洗涤剂，水温≤30℃，浸泡≤20分钟轻轻按压，机洗选羽绒服专用模式并放洗衣袋；"
          "在通风阴凉处阴干，晾晒时轻轻拍打恢复蓬松。",
          "洗涤养护.txt", ["跨段落归纳"]),
    _item("wash-04", "厚羊毛大衣可以水洗吗？",
          "不可以，必须干洗。干洗能保护羊毛纤维的弹性和柔软度，禁止水洗、机洗，避免纤维毡化和缩水。",
          "洗涤养护.txt", ["否定性提问"]),
    _item("wash-05", "针织棉机洗需要注意什么？",
          "机洗需用洗衣袋并选针织专用模式，水温≤25℃，用中性洗涤剂轻轻按压，禁止用力搓揉、拧绞；养护要平铺阴干，避免悬挂拉伸领口。",
          "洗涤养护.txt", ["跨段落归纳"]),
    _item("wash-06", "纯棉衣物首次洗涤怎么防止掉色？",
          "首次洗涤加少许盐固色，并且浅色与深色要分开洗。",
          "洗涤养护.txt", ["单点事实"]),
    _item("wash-07", "灯芯绒衣服怎么熨烫？",
          "用低温蒸汽熨烫，熨斗垫一层薄布，顺着绒面方向熨烫。",
          "洗涤养护.txt", ["单点事实"]),
    _item("wash-08", "冰丝材质最多能浸泡多久？",
          "浸泡不超过10分钟；水温≤30℃、用中性洗涤剂，机洗要放洗衣袋并选轻柔模式。",
          "洗涤养护.txt", ["精确数值"]),
    _item("wash-09", "羽绒服收纳要注意什么？",
          "完全干燥后收纳，压缩袋不要过度压缩，并定期取出拍打通风。",
          "洗涤养护.txt", ["单点事实"]),
    _item("wash-10", "薄牛仔（牛仔裤）需要多久洗一次？",
          "避免频繁清洗，1-2周洗一次即可；水温≤30℃、中性洗涤剂，翻面清洗减少褪色。",
          "洗涤养护.txt", ["精确数值"]),
    _item("wash-11", "厚牛仔首次洗如何固色？",
          "用盐水浸泡30分钟固色；水温≤30℃，翻面清洗可减少褪色和磨损。",
          "洗涤养护.txt", ["单点事实"]),
    _item("wash-12", "麂皮绒外套表面的灰尘怎么清理？",
          "用软毛刷轻轻刷除；养护要阴凉阴干、避免暴晒和高温，收纳时悬挂以避免折痕。",
          "洗涤养护.txt", ["单点事实"]),
    _item("wash-13", "衣服上沾了油渍该怎么处理？",
          "用洗洁精原液针对性处理，污渍要及时处理，时间越久越难去除。",
          "洗涤养护.txt", ["单点事实"]),
    _item("wash-14", "血渍应该怎么清洗？",
          "用冷水浸泡处理，并及时处理，时间越久越难去除。",
          "洗涤养护.txt", ["单点事实"]),
    _item("wash-15", "加绒牛仔裤洗涤有什么要注意的？",
          "水温≤30℃、中性洗涤剂，翻面清洗，机洗选轻柔模式，避免长时间浸泡以减少绒层脱落。",
          "洗涤养护.txt", ["跨段落归纳"]),

    # ---------------- 颜色推荐.txt ----------------
    _item("color-01", "冷白皮适合穿什么颜色？",
          "冷色调和暖色调都适合，亮色系（宝蓝、正红、薄荷绿）更显白皙透亮；深色系（黑色、深灰）可提升气场。",
          "颜色推荐.txt", ["单点事实"]),
    _item("color-02", "黄皮应该避免哪些颜色？",
          "避免冷调荧光色（荧光绿、冷粉），也要避免冷调粉色和紫色，容易显肤色蜡黄暗沉；浅米色、燕麦色可以柔和肤色。",
          "颜色推荐.txt", ["否定性提问"]),
    _item("color-03", "黑皮肤适合什么颜色？",
          "适合高饱和度亮色，如明黄、橙色、湖蓝，能突出健康肤色；避免暗沉的土黄色、灰褐色。",
          "颜色推荐.txt", ["单点事实"]),
    _item("color-04", "商务面试场合穿什么颜色比较合适？",
          "首选深色系，黑色、藏蓝、深灰，稳重专业；避免大面积亮色和花哨图案，保持简洁得体。",
          "颜色推荐.txt", ["单点事实"]),
    _item("color-05", "怎样通过颜色搭配显得更高？",
          "上浅下深形成视觉落差，例如浅粉上衣配黑色裤子；或者用同色系搭配（全身米色、全身黑色）让线条更流畅。",
          "颜色推荐.txt", ["跨段落归纳"]),
    _item("color-06", "怎样穿显得更瘦？",
          "优先选深色系（黑色、深灰、藏蓝）在视觉上收缩身形；避免大面积亮色和横条纹，容易显臃肿。",
          "颜色推荐.txt", ["跨段落归纳"]),
    _item("color-07", "冬季适合什么服装颜色？",
          "适合深色系和高饱和度亮色，如黑色、酒红、墨绿色，保暖且显气场；白色羽绒服也是冬季经典选择。",
          "颜色推荐.txt", ["单点事实"]),
    _item("color-08", "颜色搭配有哪些禁忌？",
          "避免大面积高饱和色撞色（红配绿、蓝配橙），可用黑白灰过渡；黄皮避免冷调粉色和紫色，黑皮避免荧光色；"
          "正式场合配色不超过三种。",
          "颜色推荐.txt", ["跨段落归纳"]),
    _item("color-09", "瘦小体型怎么选颜色？",
          "适合浅色系和暖色系，如白色、米色、浅黄，视觉上更显饱满；避免过于宽大的深色服装，容易显单薄。",
          "颜色推荐.txt", ["单点事实"]),
]


# --------------------------------------------------------------------------- 读写
def builtin_eval_set() -> list[EvalItem]:
    return [EvalItem.from_dict(item.to_dict()) for item in BUILTIN_EVAL_SET]


def save_eval_set(items: list[EvalItem], path: Path = DEFAULT_EVAL_SET) -> Path:
    """写成 JSONL：一行一条，方便手工增删和 git diff"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_eval_set(path: Path = DEFAULT_EVAL_SET) -> list[EvalItem]:
    """读评测集；文件不存在时用内置评测集并自动落盘"""
    if not path.exists():
        print(f"[评测集] {path} 不存在，使用内置评测集并写入该文件")
        save_eval_set(builtin_eval_set(), path)
    items: list[EvalItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(EvalItem.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"评测集第 {line_no} 行格式错误：{exc}") from exc
    if not items:
        raise ValueError(f"评测集为空：{path}")
    return items


def summarize(items: list[EvalItem]) -> str:
    """打印评测集构成，便于确认覆盖度"""
    by_source: dict[str, int] = {}
    by_tag: dict[str, int] = {}
    for item in items:
        by_source[item.source] = by_source.get(item.source, 0) + 1
        for tag in item.tags:
            by_tag[tag] = by_tag.get(tag, 0) + 1
    lines = [f"[评测集] 共 {len(items)} 条"]
    lines.append("  按语料：" + "，".join(f"{k} {v} 条" for k, v in sorted(by_source.items())))
    lines.append("  按类型：" + "，".join(f"{k} {v} 条" for k, v in sorted(by_tag.items())))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if "--init" in sys.argv or not DEFAULT_EVAL_SET.exists():
        save_eval_set(builtin_eval_set())
        print(f"[评测集] 已写入 {DEFAULT_EVAL_SET}")
    items = load_eval_set()
    print(summarize(items))
