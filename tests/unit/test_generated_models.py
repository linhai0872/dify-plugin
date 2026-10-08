"""Regression guards for model metadata that once broke real requests."""

import yaml

from conftest import ROOT


def _rules(stem):
    data = yaml.safe_load((ROOT / "models" / "llm" / f"{stem}.yaml").read_text())
    return {r["name"]: r for r in data.get("parameter_rules") or []}


def test_gemini_25_pro_has_no_thinking_off_switch():
    # Turning thinking off sends thinking_budget=0, which gemini-2.5-pro rejects with 400.
    assert "thinking_mode" not in _rules("gemini-2.5-pro")
    assert "thinking_budget" in _rules("gemini-2.5-pro")


def test_claude_temperature_only_where_accepted():
    # Newer Claude models reject any non-default temperature; haiku-4.5 caps it at 1.
    assert "temperature" not in _rules("claude-opus-4.8")
    assert _rules("claude-haiku-4.5")["temperature"]["max"] == 1.0


def test_claude_thinking_style_matches_model():
    assert "reasoning_budget" in _rules("claude-haiku-4.5")
    assert "reasoning_budget" not in _rules("claude-opus-4.8")


def _model(folder, stem):
    return yaml.safe_load((ROOT / "models" / folder / f"{stem}.yaml").read_text())


def test_multimodal_retrieval_models_are_flagged_for_dify():
    # Dify's knowledge base only sends images to embedding/rerank models that list `vision`.
    assert "vision" in _model("text_embedding", "qwen3-vl-embedding")["features"]
    assert "vision" in _model("rerank", "qwen3-vl-rerank")["features"]
    # ZenMux embeds gemini-embedding-2 data URIs as text, so it must stay text-only.
    assert "features" not in _model("text_embedding", "gemini-embedding-2")
    assert "features" not in _model("rerank", "qwen3-rerank")


def test_qwen_vl_embedding_batch_limit():
    # ZenMux rejects more than 20 inputs per request for this model (400).
    assert _model("text_embedding", "qwen3-vl-embedding")["model_properties"]["max_chunks"] == 20


def test_free_and_retired_models_are_not_listed():
    names = {p.stem for p in (ROOT / "models" / "llm").glob("*.yaml")}
    assert not any(n.endswith("-free") for n in names)
    assert not {"claude-opus-4.1", "glm-5-turbo", "ernie-x1.1-preview"} & names
