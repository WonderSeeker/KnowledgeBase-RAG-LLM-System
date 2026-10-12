"""生成/裁判后端可切换（`config.chat_backend` 与 `--judge-provider`）

为什么需要这层：RAGAS 的 `faithfulness` / `answer_relevancy` 需要**真实生成的答案**，
而裁判 LLM 与生成 LLM 可以是两个来源（例如"生成用 DeepSeek、裁判用另一个"）。
有了它，"本地 LLM 零成本对比"也不需要再改代码。

这里钉住三件事：

1. **默认必须仍然是 dashscope（通义千问）** —— 改造前的行为不能被悄悄改掉；
2. 换成 `openai-compatible` 时，base_url / 模型名 / **取 key 的变量名**全部来自 config（不写死厂商）；
3. 缺 key 时报错必须指出**准确的环境变量名** —— 原实现写死 `DASHSCOPE_API_KEY`，
   换成 DeepSeek 时会误报"缺 DASHSCOPE_API_KEY"，把人带偏。
"""

from __future__ import annotations

import pytest

import config_data as config
import rag
from run_ragas import (
    build_judge_llm,
    chat_api_key_env,
    judge_api_key_env,
    require_chat_key,
    require_judge_key,
)


def _require_langchain_openai():
    """走 OpenAI 兼容端点时必须要有 langchain-openai

    ★ CI 只装 `requirements.txt` + `requirements-optim.txt` + `requirements-dev.txt`，
      而 **langchain-openai 属于评测依赖**（在 `evaluation/requirements-eval.txt` 里）
      → 这几条测试在 CI 会自动 skip（并出现在 job summary 的跳过列表里），本地照常跑。
      这也是 CI 刻意"只跑确定性单测"的一部分：不为了覆盖率去装一堆重依赖。
    """
    pytest.importorskip("langchain_openai")


# --------------------------------------------------------------------------- 默认行为不变
def test_default_chat_backend_is_dashscope():
    assert config.chat_backend == "dashscope"


def test_rag_chat_model_defaults_to_tongyi(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dummy-construction-only")
    model = rag.build_chat_model()
    assert type(model).__name__ == "ChatTongyi"
    assert model.model_name == config.chat_model_name


# --------------------------------------------------------------------------- 切到 OpenAI 兼容端点
def test_rag_chat_model_switches_to_openai_compatible(monkeypatch):
    _require_langchain_openai()
    monkeypatch.setattr(config, "chat_backend", "openai-compatible", raising=False)
    monkeypatch.setenv(config.openai_compatible_api_key_env, "sk-dummy")

    model = rag.build_chat_model()
    assert type(model).__name__ == "ChatOpenAI"
    assert model.model_name == config.openai_compatible_model
    assert config.openai_compatible_base_url in str(model.openai_api_base)


def test_rag_chat_model_accepts_deepseek_alias(monkeypatch):
    _require_langchain_openai()
    monkeypatch.setattr(config, "chat_backend", "deepseek", raising=False)
    monkeypatch.setenv(config.openai_compatible_api_key_env, "sk-dummy")
    assert type(rag.build_chat_model()).__name__ == "ChatOpenAI"


def test_rag_chat_model_error_names_the_exact_env_var(monkeypatch):
    _require_langchain_openai()
    monkeypatch.setattr(config, "chat_backend", "openai-compatible", raising=False)
    monkeypatch.delenv(config.openai_compatible_api_key_env, raising=False)

    with pytest.raises(SystemExit) as excinfo:
        rag.build_chat_model()
    assert config.openai_compatible_api_key_env in str(excinfo.value)


def test_rag_chat_model_rejects_unknown_backend(monkeypatch):
    monkeypatch.setattr(config, "chat_backend", "nope", raising=False)
    with pytest.raises(SystemExit):
        rag.build_chat_model()


# --------------------------------------------------------------------------- 裁判侧
def test_judge_api_key_env_mapping():
    assert judge_api_key_env("dashscope") == "DASHSCOPE_API_KEY"
    assert judge_api_key_env("deepseek") == config.openai_compatible_api_key_env
    assert judge_api_key_env("openai-compatible") == config.openai_compatible_api_key_env
    assert judge_api_key_env("ollama") is None          # 本地服务不需要 key


def test_require_judge_key_points_at_the_right_variable(monkeypatch):
    monkeypatch.delenv(config.openai_compatible_api_key_env, raising=False)
    with pytest.raises(SystemExit) as excinfo:
        require_judge_key("deepseek")
    message = str(excinfo.value)
    assert config.openai_compatible_api_key_env in message
    assert "DASHSCOPE_API_KEY" not in message           # 不能误报到 DashScope 头上


def test_require_judge_key_silent_for_local_provider():
    require_judge_key("ollama")                          # 不抛异常即通过


def test_judge_llm_for_deepseek_uses_config(monkeypatch):
    _require_langchain_openai()
    monkeypatch.setenv(config.openai_compatible_api_key_env, "sk-dummy")
    judge = build_judge_llm("deepseek", config.openai_compatible_model)
    assert type(judge).__name__ == "ChatOpenAI"
    assert judge.model_name == config.openai_compatible_model
    assert config.openai_compatible_base_url in str(judge.openai_api_base)


def test_judge_llm_ignores_dashscope_default_model_on_openai_provider(monkeypatch):
    """`--judge-model` 的默认值是给 dashscope 的（qwen-plus）；
    换到 DeepSeek 时必须落到 config 里的模型名，否则会拿 qwen-plus 去打 DeepSeek（404）"""
    _require_langchain_openai()
    monkeypatch.setenv(config.openai_compatible_api_key_env, "sk-dummy")
    judge = build_judge_llm("deepseek", "qwen-plus")
    assert judge.model_name == config.openai_compatible_model


def test_judge_llm_rejects_unknown_provider():
    with pytest.raises(SystemExit):
        build_judge_llm("nope", "x")


# --------------------------------------------------------------------------- 生成侧的 key 检查（real 模式门槛）
def test_chat_api_key_env_follows_backend(monkeypatch):
    assert chat_api_key_env() == "DASHSCOPE_API_KEY"        # 默认后端
    monkeypatch.setattr(config, "chat_backend", "openai-compatible", raising=False)
    assert chat_api_key_env() == config.openai_compatible_api_key_env


def test_require_chat_key_names_the_backend_and_variable(monkeypatch):
    """★ 回归：real 模式的生成侧门槛曾写死 DASHSCOPE_API_KEY ——
    换到 DeepSeek 时会误报"缺 DASHSCOPE_API_KEY"（实际那次是 1 条冒烟跑抓到的）"""
    monkeypatch.setattr(config, "chat_backend", "openai-compatible", raising=False)
    monkeypatch.delenv(config.openai_compatible_api_key_env, raising=False)

    with pytest.raises(SystemExit) as excinfo:
        require_chat_key()
    message = str(excinfo.value)
    assert config.openai_compatible_api_key_env in message
    assert "DASHSCOPE_API_KEY" not in message


def test_require_chat_key_passes_when_key_present(monkeypatch):
    monkeypatch.setattr(config, "chat_backend", "openai-compatible", raising=False)
    monkeypatch.setenv(config.openai_compatible_api_key_env, "sk-dummy")
    require_chat_key()                                       # 不抛异常即通过


# --------------------------------------------------------------------------- 裁判兼容性（n=1）
def test_judge_compat_lowers_answer_relevancy_strictness():
    """★ 回归：DeepSeek 只支持 n=1，而 answer_relevancy 默认 strictness=3
    → 不降到 1 会 400，该指标整列变 N/A（其它 3 个照常出分，很容易被忽略）。
    这是 1 条冒烟跑抓到的。"""
    metrics_module = pytest.importorskip("ragas.metrics")     # CI 不装评测依赖 → 自动跳过
    from run_ragas import apply_judge_compat

    metric = metrics_module.AnswerRelevancy()
    assert metric.strictness == 3                              # ragas 的默认值
    assert apply_judge_compat("deepseek", metric) is True
    assert metric.strictness == 1


def test_judge_compat_is_noop_for_providers_supporting_n_gt_1():
    metrics_module = pytest.importorskip("ragas.metrics")
    from run_ragas import apply_judge_compat

    metric = metrics_module.AnswerRelevancy()
    assert apply_judge_compat("dashscope", metric) is False
    assert metric.strictness == 3                              # 不动它


# --------------------------------------------------------------------------- 报告里的两个报数字段
def test_resolve_judge_model_reports_the_model_actually_used():
    """★ 回归：报告曾把裁判记为 qwen-plus，而实际调用的是 deepseek-chat
    （`--judge-model` 的默认值是给 dashscope 的，换端点后必须落到 config 里的模型名）"""
    from run_ragas import resolve_judge_model

    assert resolve_judge_model("dashscope", "qwen-plus") == "qwen-plus"
    assert resolve_judge_model("deepseek", "qwen-plus") == config.openai_compatible_model
    assert resolve_judge_model("deepseek", "deepseek-reasoner") == "deepseek-reasoner"


def test_chunking_from_collection_parses_real_values():
    """★ 回归：real 模式的分块发生在入库时，报告必须记**集合里真实的**分块，
    而不是把 config 默认值（1000/100）冒充成实际值（第一次跑真实基线时报告就写错了）"""
    from run_ragas import chunking_from_collection

    assert chunking_from_collection("rag_local_corpus_c300_o30") == (300, 30)
    assert chunking_from_collection("rag_local_c1000_o100") == (1000, 100)
    assert chunking_from_collection("rag_local") is None          # 解析不出来就如实返回 None
    assert chunking_from_collection(None) is None
    assert chunking_from_collection("（复用明细：未知）") is None


def test_scored_counts_reports_per_metric_sample_counts():
    """★ 回归：指标是"可用样本上的均值"，分母必须显式记录（实测 faithfulness 曾为 35/36）"""
    from run_ragas import scored_counts

    rows = [
        {"faithfulness": 1.0, "answer_relevancy": 0.5},
        {"faithfulness": None, "answer_relevancy": 0.6},          # 打分失败 → None
        {"faithfulness": float("nan"), "answer_relevancy": 0.7},  # 打分失败 → NaN
    ]
    counts = scored_counts(rows, ("faithfulness", "answer_relevancy"))
    assert counts["faithfulness"] == 1
    assert counts["answer_relevancy"] == 3
