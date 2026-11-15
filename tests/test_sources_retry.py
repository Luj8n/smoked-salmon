import ssl

import anyio
import pytest
from aiohttp import web

from salmon.errors import ScrapeError
from salmon.search.beatport import Searcher
from salmon.sources import retry
from salmon.sources.base import BaseScraper
from salmon.sources.retry import SourceHTTPError


@pytest.fixture
def waits(monkeypatch):
    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(retry.anyio, "sleep", sleep)
    monkeypatch.setattr(retry, "uniform", lambda _a, _b: 0)
    return delays


async def fetch(statuses, after=None, page=False):
    calls = []

    async def handler(request):
        calls.append(request.method)
        status = statuses[min(len(calls) - 1, len(statuses) - 1)]
        return web.json_response({"ok": True}, status=status, headers={"Retry-After": after} if after else {})

    app = web.Application()
    app.router.add_get("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        url = f"http://127.0.0.1:{runner.addresses[0][1]}/"
        searcher = Searcher()
        try:
            result = await searcher.fetch_page(url) if page else await searcher.get_json(url)
        except ScrapeError as error:
            result = error
        return result, calls
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("page", [False, True])
def test_beatport_get_recovers_with_bounded_backoff(waits, page):
    result, calls = anyio.run(fetch, [503, 502, 200], None, page)
    assert not isinstance(result, ScrapeError)
    assert calls == ["GET"] * 3
    assert waits == [3, 10]


def test_persistent_failure_stops_after_three_gets(waits):
    result, calls = anyio.run(fetch, [503])
    assert isinstance(result, ScrapeError)
    assert calls == ["GET"] * 3
    assert waits == [3, 10]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_permanent_failure_is_not_retried(waits, status):
    result, calls = anyio.run(fetch, [status])
    assert isinstance(result, ScrapeError)
    assert calls == ["GET"]
    assert waits == []


def test_retry_after_is_honoured(waits):
    result, calls = anyio.run(fetch, [429, 200], "15")
    assert not isinstance(result, ScrapeError)
    assert calls == ["GET"] * 2
    assert waits == [15]


def test_long_retry_after_is_not_shortened(waits):
    result, calls = anyio.run(fetch, [429], "120")
    assert isinstance(result, ScrapeError)
    assert calls == ["GET"]
    assert waits == []


def test_total_wait_budget_is_bounded(waits):
    result, calls = anyio.run(fetch, [503], "40")
    assert isinstance(result, ScrapeError)
    assert calls == ["GET"] * 2
    assert waits == [40]


def test_timeout_is_retried_but_other_scrapers_stay_opt_in(waits):
    class TimedOut:
        retry_gets = True
        store = "Beatport"
        calls = 0

        @retry.retry_get
        async def get(self):
            self.calls += 1
            if self.calls == 1:
                raise ScrapeError("timeout") from TimeoutError()
            return "ok"

    reader = TimedOut()
    assert anyio.run(reader.get) == "ok"
    assert waits == [3]
    assert not BaseScraper.retry_gets


def test_exhausted_api_request_does_not_fall_back_to_another_endpoint(monkeypatch):
    async def token(self):
        return object()

    async def fail(self, *args):
        raise SourceHTTPError("cooldown", 429, "120")

    async def scraper(self, *args):
        pytest.fail("A cooldown must not trigger HTML requests")

    monkeypatch.setattr(Searcher, "load_token_storage", token)
    monkeypatch.setattr(Searcher, "search_releases_api", fail)
    monkeypatch.setattr(Searcher, "search_releases_scraping", scraper)
    with pytest.raises(ScrapeError):
        anyio.run(Searcher().search_releases, "album", 5)


def test_certificate_errors_are_not_transient():
    error = ScrapeError("certificate")
    error.__cause__ = ssl.SSLCertVerificationError()
    assert not retry.transient(error)


@pytest.mark.parametrize("title", ["Verifying your browser", "Client Challenge", "Just a moment..."])
def test_browser_check_is_not_transient(title):
    error = SourceHTTPError("challenge", 503, None, f"<html><title>{title}</title></html>")
    assert not retry.transient(error)


@pytest.mark.parametrize(
    "status,after,payload", [(429, None, None), (503, "120", None), (403, None, "<title>Client Challenge</title>")]
)
def test_login_cooldown_or_challenge_does_not_start_scraping(monkeypatch, status, after, payload):
    from salmon.sources.beatport import TokenStorage

    async def login(self):
        raise SourceHTTPError("blocked", status, after, payload)

    async def scraper(self, *args):
        pytest.fail("Do not continue making requests after a server cooldown or challenge")

    monkeypatch.setattr(TokenStorage, "load", classmethod(lambda cls: None))
    monkeypatch.setattr(Searcher, "initial_setup", login)
    monkeypatch.setattr(Searcher, "search_releases_scraping", scraper)
    with pytest.raises(SourceHTTPError):
        anyio.run(Searcher().search_releases, "album", 5)
