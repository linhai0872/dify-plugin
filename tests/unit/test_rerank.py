import pytest
from dify_plugin.entities.model import ModelType
from dify_plugin.errors.model import InvokeRateLimitError


class FakeResp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


@pytest.fixture
def rerank(factory):
    return factory.get_instance(ModelType.RERANK)


def test_rerank_request_and_raw_scores(rerank, monkeypatch):
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, body=json)
        return FakeResp(200, {"results": [{"index": 2, "relevance_score": 0.3}, {"index": 0, "relevance_score": 0.9}]})

    monkeypatch.setattr("models.rerank.rerank.requests.post", fake_post)
    result = rerank._invoke("qwen/qwen3-rerank", {"api_key": "k", "region": "cn"}, "q", ["a", "b", "c"],
                            score_threshold=0.5, top_n=2)
    assert seen["url"] == "https://zenmux.dev/api/v1/rerank"
    assert seen["body"]["input"] == {"query": "q", "documents": ["a", "b", "c"]}
    assert seen["body"]["parameters"]["top_n"] == 2
    assert [(d.index, d.text, d.score) for d in result.docs] == [(0, "a", 0.9)]


def test_rerank_http_errors_are_typed(rerank, monkeypatch):
    monkeypatch.setattr("models.rerank.rerank.requests.post", lambda *a, **k: FakeResp(429, {"error": "slow down"}))
    with pytest.raises(InvokeRateLimitError):
        rerank.invoke("qwen/qwen3-rerank", {"api_key": "k"}, "q", ["a"])
