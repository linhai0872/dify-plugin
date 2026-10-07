"""Dify parameter values -> ZenMux request fields."""

from dify_plugin.entities.model.llm import LLMResultChunk, LLMResultChunkDelta
from dify_plugin.entities.model.message import AssistantPromptMessage

from models._common import strip_reasoning
from models.llm.openai import ZenMuxOpenAICCLargeLanguageModel


def test_reasoning_params_mapping():
    params = {"enable_thinking": True, "reasoning_effort": "xhigh", "reasoning_budget": 2048, "temperature": 0.2}
    ZenMuxOpenAICCLargeLanguageModel._set_reasoning_params(params)
    assert params == {"temperature": 0.2, "reasoning": {"enabled": True, "effort": "xhigh", "max_tokens": 2048}}


def test_unknown_effort_is_dropped():
    params = {"reasoning_effort": "turbo"}
    ZenMuxOpenAICCLargeLanguageModel._set_reasoning_params(params)
    assert params == {}


def test_json_schema_is_wrapped():
    params = {"response_format": "json_schema", "json_schema": '{"type": "object"}'}
    ZenMuxOpenAICCLargeLanguageModel._set_json_schema_params(params)
    assert params["json_schema"] == '{"name": "output", "schema": {"type": "object"}}'


def _chunks(*contents):
    return (LLMResultChunk(model="m", delta=LLMResultChunkDelta(index=i, message=AssistantPromptMessage(content=c)))
            for i, c in enumerate(contents))


def test_strip_reasoning_removes_think_blocks():
    out = [c.delta.message.content for c in strip_reasoning(_chunks("<think>\nplan", " more", "\n</think>Answer", " done"))]
    assert "".join(out) == "Answer done"


def test_strip_reasoning_passthrough_without_thinking():
    out = [c.delta.message.content for c in strip_reasoning(_chunks("plain ", "text"))]
    assert out == ["plain ", "text"]


def test_anthropic_thinking_style_follows_schema(llm):
    budget_model = llm._get_model_class_for_model("anthropic/claude-haiku-4.5")
    adaptive_model = llm._get_model_class_for_model("anthropic/claude-opus-4.8")
    on = {"enable_thinking": True}
    assert budget_model._thinking_config("anthropic/claude-haiku-4.5", {}, on)["type"] == "enabled"
    assert adaptive_model._thinking_config("anthropic/claude-opus-4.8", {}, on) == {"type": "adaptive"}
    assert adaptive_model._thinking_config("anthropic/claude-opus-4.8", {}, {}) is None


def test_anthropic_json_schema_objects_are_closed(llm):
    model = llm._get_model_class_for_model("anthropic/claude-haiku-4.5")
    schema = '{"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"a": {"type": "string"}}}}}}'
    _, extra = model._build_request_params(
        "anthropic/claude-haiku-4.5", {"max_tokens": 10, "response_format": "json_schema", "json_schema": schema},
        None, [], None, None,
    )
    out = extra["output_config"]["format"]["schema"]
    assert out["additionalProperties"] is False
    assert out["properties"]["items"]["items"]["additionalProperties"] is False


def test_anthropic_budget_never_exceeds_max_tokens(llm):
    model = llm._get_model_class_for_model("anthropic/claude-haiku-4.5")
    params, _ = model._build_request_params(
        "anthropic/claude-haiku-4.5", {"max_tokens": 1000}, None, [], None, None,
        thinking={"type": "enabled", "budget_tokens": 4096},
    )
    assert params["max_tokens"] > 4096
    assert "temperature" not in params
