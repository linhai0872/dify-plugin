"""Live smoke tests through the plugin's own classes (real ZenMux calls, a few cents in total).

Run: ZENMUX_LIVE_API_KEY=sk-... pytest tests/live -q
Optional: ZENMUX_LIVE_REGION=cn to run everything against zenmux.dev.
"""

import json
import os
from pathlib import Path

import pytest
import yaml
from dify_plugin.entities.model import ModelType
from dify_plugin.entities.model.message import (
    PromptMessageTool,
    SystemPromptMessage,
    UserPromptMessage,
)

KEY = os.environ.get("ZENMUX_LIVE_API_KEY")
pytestmark = pytest.mark.skipif(not KEY, reason="set ZENMUX_LIVE_API_KEY to run live tests")

WEATHER = PromptMessageTool(
    name="get_weather", description="Get the current weather for a city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
SCHEMA = json.dumps({"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]})


def creds(region=None):
    return {"api_key": KEY, "region": region or os.environ.get("ZENMUX_LIVE_REGION", "global")}


def ask(text):
    return [UserPromptMessage(content=text)]


def as_text(content):
    # Content is either a string or a list of content parts (the Gemini path streams parts).
    if isinstance(content, list):
        return "".join(getattr(part, "data", "") for part in content if getattr(part, "type", None) == "text")
    return content or ""


def run(llm, model, prompt, params=None, tools=None, stream=True, region=None):
    # The SDK always yields chunks; stream=False only changes the upstream request.
    result = llm.invoke(model, creds(region), prompt, params or {}, tools=tools, stream=stream)
    text, calls, usage = "", [], None
    for chunk in result:
        text += as_text(chunk.delta.message.content)
        calls += chunk.delta.message.tool_calls or []
        usage = chunk.delta.usage or usage
    return text, calls, usage


def assert_billed(usage):
    assert usage is not None and usage.prompt_tokens > 0 and usage.completion_tokens > 0
    assert usage.total_price > 0


@pytest.mark.parametrize("model", [
    "openai/gpt-4.1-nano",             # OpenAI-compatible
    "anthropic/claude-haiku-4.5",      # Anthropic native
    "google/gemini-2.5-flash-lite",    # Vertex
    "moonshotai/kimi-k2.6",            # generic OpenAI-compatible
])
@pytest.mark.parametrize("stream", [True, False])
def test_basic_chat_and_usage(llm, model, stream):
    params = {"max_output_tokens": 256} if model.startswith("google/") else {"max_tokens": 256}
    text, _, usage = run(llm, model, ask("What is 17*23? Reply with just the number."), params, stream=stream)
    assert "391" in text
    assert_billed(usage)


@pytest.mark.parametrize("model", ["openai/gpt-4.1-nano", "anthropic/claude-haiku-4.5", "google/gemini-2.5-flash-lite", "z-ai/glm-5.1"])
def test_tool_call(llm, model):
    params = {"max_output_tokens": 512} if model.startswith("google/") else {"max_tokens": 512}
    _, calls, _ = run(llm, model, ask("What's the weather in Paris right now? Use the tool."), params, tools=[WEATHER])
    assert calls and calls[0].function.name == "get_weather"
    assert "paris" in calls[0].function.arguments.lower()


@pytest.mark.parametrize("model", ["openai/gpt-4.1-nano", "anthropic/claude-haiku-4.5"])
def test_json_schema(llm, model):
    params = {"max_tokens": 128, "response_format": "json_schema", "json_schema": SCHEMA}
    text, _, _ = run(llm, model, [SystemPromptMessage(content="Answer in JSON."), *ask("What is 17*23?")], params)
    assert json.loads(text.strip().strip("`").removeprefix("json"))["answer"] == 391


def test_thinking_shown_and_hidden(llm):
    prompt = ask("What is 17*23? Reply with just the number.")
    shown, _, _ = run(llm, "moonshotai/kimi-k2.6", prompt, {"max_tokens": 1500, "enable_thinking": True})
    hidden, _, _ = run(llm, "moonshotai/kimi-k2.6", prompt,
                       {"max_tokens": 1500, "enable_thinking": True, "exclude_reasoning_tokens": True})
    assert "<think>" in shown and "391" in shown
    assert "<think>" not in hidden and "391" in hidden


@pytest.mark.parametrize("model,expect", [("anthropic/claude-haiku-4.5", "budget"), ("anthropic/claude-sonnet-5", "adaptive")])
def test_claude_thinking_styles(llm, model, expect):
    text, _, usage = run(llm, model, ask("What is 17*23? Reply with just the number."),
                         {"max_tokens": 4096, "enable_thinking": True})
    assert "391" in text
    assert_billed(usage)


THINKING_OFF = json.loads((Path(__file__).resolve().parents[2] / "models" / "llm" / "_anthropic_thinking.json").read_text())
TRICKY = "A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. How much is the ball? Answer with just the number."


@pytest.mark.parametrize("model", sorted(m for m, off in THINKING_OFF.items() if off != "always_on"))
def test_claude_thinking_switch_off_really_stops_thinking(llm, model):
    # Newer Claude models think by default; with the switch off the plugin must send the model's off value.
    text, _, usage = run(llm, model, ask(TRICKY), {"max_tokens": 1024, "enable_thinking": False})
    assert "<think>" not in text and "0.05" in text, (model, text[:200])
    assert_billed(usage)


@pytest.mark.parametrize("model", sorted(m for m, off in THINKING_OFF.items() if off == "always_on"))
def test_always_on_claude_can_hide_the_thought_process(llm, model):
    text, _, usage = run(llm, model, ask(TRICKY), {"max_tokens": 4096, "exclude_reasoning_tokens": True})
    assert "<think>" not in text and "0.05" in text, (model, text[:200])
    assert_billed(usage)


@pytest.mark.parametrize("model,params", [
    ("google/gemini-2.5-flash-lite", {"thinking_mode": False}),
    ("google/gemini-2.5-flash-lite", {"thinking_mode": True, "thinking_budget": 1024}),
    ("google/gemini-2.5-pro", {"thinking_budget": 1024}),
    ("google/gemini-3.5-flash", {"thinking_level": "Low"}),
])
def test_gemini_thinking_controls(llm, model, params):
    text, _, usage = run(llm, model, ask("What is 17*23? Reply with just the number."),
                         {"max_output_tokens": 2048, **params})
    assert "391" in text
    assert_billed(usage)


def _square_png(rgb: bytes = b"\xff\x00\x00") -> str:
    import base64
    import struct
    import zlib

    raw = b"".join(b"\x00" + rgb * 64 for _ in range(64))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return base64.b64encode(png).decode()


# openai/gpt-4.1* is left out on purpose: it misreads images even via the raw ZenMux API (upstream issue).
@pytest.mark.parametrize("model,params", [
    ("openai/gpt-4o-mini", {"max_tokens": 32}),
    ("anthropic/claude-haiku-4.5", {"max_tokens": 32}),
    ("google/gemini-2.5-flash-lite", {"max_output_tokens": 32}),
    ("moonshotai/kimi-k2.6", {"max_tokens": 512}),
])
def test_vision_input(llm, model, params):
    from dify_plugin.entities.model.message import ImagePromptMessageContent, TextPromptMessageContent

    prompt = [UserPromptMessage(content=[
        TextPromptMessageContent(data="What single color fills this image? Answer with one word."),
        ImagePromptMessageContent(format="png", base64_data=_square_png(), mime_type="image/png"),
    ])]
    text, _, _ = run(llm, model, prompt, params)
    assert "red" in text.lower().split("</think>")[-1]


def test_openai_reasoning_effort(llm):
    text, _, usage = run(llm, "openai/gpt-5-nano", ask("What is 17*23? Reply with just the number."),
                         {"max_tokens": 2000, "reasoning_effort": "low"})
    assert "391" in text
    assert_billed(usage)


def test_mainland_region_endpoint(llm):
    text, _, usage = run(llm, "openai/gpt-4.1-nano", ask("Reply with exactly: OK"), {"max_tokens": 16}, region="cn")
    assert "OK" in text
    assert_billed(usage)


def test_provider_credentials(registration):
    _config, provider, _factory = registration.models_mapping["zenmux"]
    provider.validate_provider_credentials(creds())


@pytest.mark.parametrize("model", ["openai/text-embedding-3-small", "google/gemini-embedding-2", "qwen/qwen3-vl-embedding"])
def test_embeddings_keep_one_vector_per_text(factory, model):
    embedder = factory.get_instance(ModelType.TEXT_EMBEDDING)
    result = embedder.invoke(model, creds(), ["alpha", "beta", "gamma"])
    assert len(result.embeddings) == 3
    assert len({len(v) for v in result.embeddings}) == 1
    assert result.usage.total_tokens > 0


def test_rerank(factory):
    reranker = factory.get_instance(ModelType.RERANK)
    docs = ["Paris is the capital of France.", "Bananas are yellow.", "Berlin is in Germany."]
    result = reranker.invoke("qwen/qwen3-rerank", creds(), "What is the capital of France?", docs, top_n=2)
    assert result.docs[0].index == 0 and 0 < result.docs[0].score <= 1


def test_qwen_vl_embedding_batches_over_the_20_input_limit(factory):
    embedder = factory.get_instance(ModelType.TEXT_EMBEDDING)
    result = embedder.invoke("qwen/qwen3-vl-embedding", creds(), [f"chunk {i}" for i in range(45)])
    assert len(result.embeddings) == 45


COLORS = {"red": b"\xe6\x1e\x1e", "green": b"\x1e\xaa\x3c", "blue": b"\x1e\x3c\xdc"}


def _vision_models(folder):
    root = Path(__file__).resolve().parents[2] / "models" / folder
    return sorted(d["model"] for p in root.glob("[!_]*.yaml")
                  if "vision" in ((d := yaml.safe_load(p.read_text())).get("features") or []))


def _mm(content, image=False):
    from dify_plugin.entities.model.text_embedding import MultiModalContent, MultiModalContentType

    return MultiModalContent(content=content, content_type=MultiModalContentType.IMAGE if image else MultiModalContentType.TEXT)


@pytest.mark.parametrize("model", _vision_models("text_embedding"))
def test_multimodal_embedding_matches_image_to_caption(factory, model):
    # Every embedding model shipped with `vision` must actually encode the image content.
    embedder = factory.get_instance(ModelType.TEXT_EMBEDDING)
    docs = [_mm(_square_png(rgb), image=True) for rgb in COLORS.values()] + [_mm(f"a solid {c} square") for c in COLORS]
    vecs = embedder.invoke_multimodal(model, creds(), docs).embeddings
    cos = lambda a, b: sum(x * y for x, y in zip(a, b)) / (sum(x * x for x in a) * sum(y * y for y in b)) ** 0.5
    for i in range(len(COLORS)):
        sims = [cos(vecs[i], vecs[len(COLORS) + j]) for j in range(len(COLORS))]
        assert sims.index(max(sims)) == i, (model, list(COLORS)[i], sims)


@pytest.mark.parametrize("model", _vision_models("rerank"))
def test_multimodal_rerank_matches_image_to_caption(factory, model):
    reranker = factory.get_instance(ModelType.RERANK)
    captions = [_mm(f"a solid {c} square") for c in COLORS]
    for i, rgb in enumerate(COLORS.values()):
        ranked = reranker.invoke_multimodal(model, creds(), _mm(_square_png(rgb), image=True), captions)
        assert ranked.docs[0].index == i, (model, list(COLORS)[i], ranked.docs)
    ranked = reranker.invoke_multimodal(model, creds(), _mm("a solid blue square"),
                                        [_mm(_square_png(rgb), image=True) for rgb in COLORS.values()])
    assert ranked.docs[0].index == 2


def test_multimodal_embedding_request_path(factory):
    # The image request path works mechanically even where image semantics are not verified yet.
    embedder = factory.get_instance(ModelType.TEXT_EMBEDDING)
    result = embedder.invoke_multimodal("qwen/qwen3-vl-embedding", creds(), [_mm(_square_png(), image=True), _mm("red")])
    assert len(result.embeddings) == 2 and result.usage.total_price > 0
