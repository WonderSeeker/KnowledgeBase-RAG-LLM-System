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
from run_ragas import build_judge_llm, judge_api_key_env, require_judge_key


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
    monkeypatch.setattr(config, "chat_backend", "openai-compatible", raising=False)
    monkeypatch.setenv(config.openai_compatible_api_key_env, "sk-dummy")

    model = rag.build_chat_model()
    assert type(model).__name__ == "ChatOpenAI"
    assert model.model_name == config.openai_compatible_model
    assert config.openai_compatible_base_url in str(model.openai_api_base)


def test_rag_chat_model_accepts_deepseek_alias(monkeypatch):
    monkeypatch.setattr(config, "chat_backend", "deepseek", raising=False)
    monkeypatch.setenv(config.openai_compatible_api_key_env, "sk-dummy")
    assert type(rag.build_chat_model()).__name__ == "ChatOpenAI"


def test_rag_chat_model_error_names_the_exact_env_var(monkeypatch):
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
    monkeypatch.setenv(config.openai_compatible_api_key_env, "sk-dummy")
    judge = build_judge_llm("deepseek", config.openai_compatible_model)
    assert type(judge).__name__ == "ChatOpenAI"
    assert judge.model_name == config.openai_compatible_model
    assert config.openai_compatible_base_url in str(judge.openai_api_base)


def test_judge_llm_ignores_dashscope_default_model_on_openai_provider(monkeypatch):
    """`--judge-model` 的默认值是给 dashscope 的（qwen-plus）；
    换到 DeepSeek 时必须落到 config 里的模型名，否则会拿 qwen-plus 去打 DeepSeek（404）"""
    monkeypatch.setenv(config.openai_compatible_api_key_env, "sk-dummy")
    judge = build_judge_llm("deepseek", "qwen-plus")
    assert judge.model_name == config.openai_compatible_model


def test_judge_llm_rejects_unknown_provider():
    with pytest.raises(SystemExit):
        build_judge_llm("nope", "x")
