"""工程卫生（与业务无关、但都是真实踩过的坑）

1. **配置路径必须绝对**：原版用 "./xxx" 相对当前工作目录 → 换目录启动会读写到**不同的向量库**
   （README 的 Q1 就是这个坑）。
2. **废弃配置不得再被引用**：`max_spliter_char_number` 保留只为兼容旧配置文件，
   逻辑里再引用它就会让"chunk_size 对短文档失效"那个缺陷复活。
3. **代码里的文档交叉引用必须真的存在**：报告层曾把"24 题集合不同"引到 §4.10.5，
   而那个数据其实在 §4.11.2 —— 这类腐化靠人眼很难发现，用测试兜住。
4. **`.env` 必须被 git 忽略**：这是密钥泄露问题，不是风格问题。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

import config_data as config

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- 路径
def test_state_paths_are_absolute_and_inside_project():
    assert Path(config.md5_path).is_absolute()
    assert Path(config.persist_directory).is_absolute()
    assert Path(config.persist_directory).parent == config.BASE_DIR
    assert Path(config.md5_path).name == "md5.text"


def test_retrieve_k_keeps_backward_compatible_alias():
    """旧名 similarity_threshold 仍可用，且与 retrieve_k 同源（原项目把它当 k 用）"""
    assert config.similarity_threshold == config.retrieve_k


# --------------------------------------------------------------------------- 废弃配置
def test_deprecated_split_threshold_is_no_longer_referenced():
    """★ 回归：判定逻辑里不得再出现 max_spliter_char_number

    它曾经的语义是"≤1000 字就不切分"，正是"chunk_size 对短文档完全失效"的根因（CHANGES §4.8）。
    配置项本身保留（兼容旧配置文件），但代码里不应再引用。
    """
    offenders = []
    for path in list(ROOT.glob("*.py")) + list((ROOT / "evaluation").glob("*.py")) \
            + list((ROOT / "scripts").glob("*.py")):
        if path.name == "config_data.py":
            continue
        text = path.read_text(encoding="utf-8")
        # 允许在注释里提到它（历史说明），但不允许出现在可执行代码里
        for lineno, line in enumerate(text.splitlines(), start=1):
            code = line.split("#", 1)[0]
            if "max_spliter_char_number" in code:
                offenders.append(f"{path.name}:{lineno}")
    assert not offenders, f"这些地方的代码又引用了废弃配置：{offenders}"


# --------------------------------------------------------------------------- 文档交叉引用
def test_changes_cross_references_in_code_exist():
    """代码注释里写的 `CHANGES §x.y.z`，必须在 CHANGES.md 里真的有这一节"""
    changes = (ROOT / "CHANGES.md").read_text(encoding="utf-8")
    headings = set(re.findall(r"^#{3,4}\s+([0-9]+(?:\.[0-9]+)*)", changes, flags=re.MULTILINE))

    missing = []
    pattern = re.compile(r"CHANGES\s*§\s*([0-9]+(?:\.[0-9]+)*)")
    for path in list(ROOT.glob("*.py")) + list((ROOT / "evaluation").glob("*.py")) \
            + list((ROOT / "scripts").glob("*.py")) + list(Path(__file__).parent.glob("*.py")):
        for number in pattern.findall(path.read_text(encoding="utf-8")):
            if number not in headings:
                missing.append(f"{path.name} → §{number}")
    assert not missing, f"这些交叉引用在 CHANGES.md 里找不到对应小节：{missing}"


# --------------------------------------------------------------------------- 密钥卫生
def test_env_file_is_git_ignored():
    """.env 里存的是 API Key —— 必须被 git 忽略（原版漏了这一条）"""
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", ".env"],
            cwd=ROOT, capture_output=True,
        )
    except FileNotFoundError:
        pytest.skip("环境里没有 git")
    if result.returncode == 127:
        pytest.skip("环境里没有 git")
    assert result.returncode == 0, ".env 没有被 git 忽略 —— 提交一次就会泄露 API Key"


def test_runtime_artifacts_are_git_ignored():
    """向量库、去重文件、评测中间产物都不该进版本库"""
    for name in ("chroma_db/", "md5.text", "evaluation/cache/", "evaluation/runs/"):
        result = subprocess.run(["git", "check-ignore", "-q", name], cwd=ROOT,
                                capture_output=True)
        assert result.returncode == 0, f"{name} 没有被 git 忽略"


# --------------------------------------------------------------------------- 文档存在性
@pytest.mark.parametrize("name", ["README.md", "CHANGES.md"])
def test_docs_exist(name):
    assert (ROOT / name).stat().st_size > 1000
