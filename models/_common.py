from collections.abc import Generator, Mapping
from decimal import Decimal, InvalidOperation

from dify_plugin.entities.model.llm import LLMResultChunk, LLMUsage
from dify_plugin.entities.model.text_embedding import EmbeddingUsage

# ZenMux serves the same API from two regions; `region` comes from the provider credentials.
REGION_HOSTS = {
    "global": "https://zenmux.ai",
    "cn": "https://zenmux.dev",
}
PRICE_UNIT = Decimal("0.000001")


def zenmux_host(credentials: Mapping | None) -> str:
    region = (credentials or {}).get("region") or "global"
    return REGION_HOSTS.get(region, REGION_HOSTS["global"])


def openai_base_url(credentials: Mapping | None) -> str:
    return f"{zenmux_host(credentials)}/api/v1"


def anthropic_base_url(credentials: Mapping | None) -> str:
    return f"{zenmux_host(credentials)}/api/anthropic"


def vertex_base_url(credentials: Mapping | None) -> str:
    return f"{zenmux_host(credentials)}/api/vertex-ai"


def _decimal(value) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _unit_price(amount: Decimal, tokens: int) -> Decimal:
    return (amount / (tokens * PRICE_UNIT)).quantize(Decimal("0.000001")) if tokens else Decimal(0)


def apply_reported_cost(usage, raw_usage: Mapping | None):
    """Replace the YAML-price estimate with the cost ZenMux reports for the request.

    ZenMux returns `usage.cost` (USD) and `usage.cost_details` synchronously on the
    OpenAI-compatible and Anthropic endpoints. It equals the bill's `originAmount`:
    tiered pricing, cache reads/writes and web search are applied, promotional
    discounts are not (verified 2026-10-07 against /management/generation). Dify's flat
    input/output price cannot express tiers or caching. Without it we keep the estimate.
    """
    cost = _decimal((raw_usage or {}).get("cost"))
    if usage is None or cost is None:
        return usage
    if isinstance(usage, EmbeddingUsage):
        return usage.model_copy(update={
            "total_price": cost, "unit_price": _unit_price(cost, usage.tokens),
            "price_unit": PRICE_UNIT, "currency": "USD",
        })
    if not isinstance(usage, LLMUsage):
        return usage
    completion = min(_decimal(((raw_usage or {}).get("cost_details") or {}).get("completion")) or Decimal(0), cost)
    prompt = cost - completion
    return usage.model_copy(update={
        "prompt_price": prompt, "prompt_unit_price": _unit_price(prompt, usage.prompt_tokens),
        "prompt_price_unit": PRICE_UNIT,
        "completion_price": completion, "completion_unit_price": _unit_price(completion, usage.completion_tokens),
        "completion_price_unit": PRICE_UNIT,
        "total_price": cost, "currency": "USD",
    })


def strip_reasoning(chunks: Generator[LLMResultChunk, None, None]) -> Generator[LLMResultChunk, None, None]:
    """Drop `<think>...</think>` segments from a stream of chunks.

    ZenMux ignores `reasoning.exclude`, so "hide the thought process" is enforced here.
    The SDK always emits the tags whole, so per-chunk scanning with carried state is enough.
    """
    inside = False
    for chunk in chunks:
        content = chunk.delta.message.content
        if not isinstance(content, str) or not content:
            if not inside or chunk.delta.message.tool_calls or chunk.delta.usage:
                yield chunk
            continue
        kept, rest = [], content
        while rest:
            if inside:
                end = rest.find("</think>")
                if end < 0:
                    rest = ""
                else:
                    rest, inside = rest[end + len("</think>"):].lstrip("\n"), False
            else:
                start = rest.find("<think>")
                if start < 0:
                    kept.append(rest)
                    rest = ""
                else:
                    kept.append(rest[:start])
                    rest, inside = rest[start + len("<think>"):], True
        text = "".join(kept)
        if text or chunk.delta.message.tool_calls or chunk.delta.usage or chunk.delta.finish_reason:
            chunk.delta.message.content = text
            yield chunk
