import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import anyio
import httpx2
import openai
import pytest
from helpers.tagger_ai_review import make_metadata, make_review

from salmon import cfg
from salmon.tagger import ai_review


@pytest.mark.parametrize("outcome", ["success", "timeout", "fallback", "invalid_json", "cancelled"])
def test_review_closes_its_api_client_on_every_exit(monkeypatch, outcome):
    async def run():
        client = openai.AsyncOpenAI(api_key="offline-test-key", base_url="http://127.0.0.1:1/v1")

        async def create(**kwargs):
            if outcome == "timeout":
                raise TimeoutError
            if outcome == "cancelled":
                raise asyncio.CancelledError
            if outcome == "fallback":
                raise openai.NotFoundError("missing", response=MagicMock(status_code=404, headers={}), body={})
            return SimpleNamespace(status="completed", error=None, output=[], id="response")

        async def fallback(*args):
            assert not client.is_closed()
            return make_review(), None

        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: client)
        monkeypatch.setattr(client.responses, "create", create)
        monkeypatch.setattr(ai_review, "_request_ai_review_chat", fallback)
        monkeypatch.setattr(ai_review, "_should_use_background", lambda: False)
        monkeypatch.setattr(
            ai_review,
            "_extract_output_text",
            lambda response: "not json" if outcome == "invalid_json" else json.dumps(make_review()),
        )
        monkeypatch.setattr(cfg.upload.ai_review, "provider", "openai")
        monkeypatch.setattr(cfg.upload.ai_review, "background", False)
        try:
            args = (make_metadata(), make_metadata(), None, None, None)
            if outcome in {"timeout", "invalid_json"}:
                with pytest.raises(RuntimeError):
                    await ai_review._request_ai_review(*args)
            elif outcome == "cancelled":
                with pytest.raises(asyncio.CancelledError):
                    await ai_review._request_ai_review(*args)
            else:
                await ai_review._request_ai_review(*args)
            assert client.is_closed(), "Review leaked its HTTP client"
        finally:
            await client.close()

    anyio.run(run)


def test_cancelling_a_review_finishes_closing_its_transport(monkeypatch):
    class Transport(httpx2.AsyncBaseTransport):
        closed = False

        async def handle_async_request(self, request):
            raise AssertionError("No network request expected")

        async def aclose(self):
            await anyio.sleep(0)
            self.closed = True

    async def run():
        transport = Transport()
        client = openai.AsyncOpenAI(api_key="offline-test-key", http_client=httpx2.AsyncClient(transport=transport))
        with anyio.CancelScope() as scope:

            async def create(**kwargs):
                scope.cancel()
                await anyio.sleep(0)

            monkeypatch.setattr(client.responses, "create", create)
            monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: client)
            monkeypatch.setattr(ai_review, "_should_use_background", lambda: False)
            monkeypatch.setattr(cfg.upload.ai_review, "provider", "openai")
            await ai_review._request_ai_review(make_metadata(), make_metadata(), None, None, None)
        assert transport.closed, "Cancelled review interrupted HTTP transport cleanup"

    anyio.run(run)


def test_chat_fallback_discloses_that_it_cannot_browse(monkeypatch):
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(make_review())))])

    client = MagicMock()
    client.chat.completions.create = create
    monkeypatch.setattr(cfg.upload.ai_review, "use_web_search", True)

    anyio.run(
        ai_review._request_ai_review_chat,
        client,
        make_metadata(),
        make_metadata(),
        "https://example.com/release",
        None,
        10,
    )

    prompt = captured["messages"][1]["content"]
    assert "Web research is unavailable." in prompt
    assert "Open that exact page first" not in prompt
