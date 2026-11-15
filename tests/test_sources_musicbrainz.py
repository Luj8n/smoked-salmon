"""MusicBrainz requests are bounded, and their failures are ScrapeErrors (#786).

musicbrainzngs, not aiohttp: the requests go to a local threaded server through its own set_hostname globals.
"""

import asyncio
import inspect
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anyio
import pytest
from musicbrainzngs import caa
from musicbrainzngs import musicbrainz as mb_module

from salmon.errors import ScrapeError
from salmon.release_notification import get_version
from salmon.search import musicbrainz as musicbrainz_search
from salmon.sources import musicbrainz, retry
from salmon.tagger.sources import musicbrainz as musicbrainz_tagger

RELEASE_ID = "0a1b2c3d-0000-0000-0000-000000000000"
RELEASE_URL = f"https://musicbrainz.org/release/{RELEASE_ID}"
RELEASE_XML = (
    '<?xml version="1.0" encoding="UTF-8"?><metadata xmlns="http://musicbrainz.org/ns/mmd-2.0#">'
    f'<release id="{RELEASE_ID}"><title>Album</title></release></metadata>'
)


class _Server:
    """A local server answering every request with `status` and `body`, after waiting `delay` seconds."""

    def __init__(self) -> None:
        self.status = 200
        self.statuses: list[int] = []
        self.retry_after: str | None = None
        self.body = RELEASE_XML
        self.delay = 0.0
        self.requests: list[str] = []
        self.user_agents: list[str | None] = []
        self.released = threading.Event()  # Set at teardown, so no handler outlives the test
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                server.requests.append(self.path)
                server.user_agents.append(self.headers.get("User-Agent"))
                if server.delay and server.released.wait(server.delay):
                    return  # The client gave up long ago: nothing to answer
                payload = server.body.encode()
                self.send_response(server.statuses.pop(0) if server.statuses else server.status)
                if server.retry_after is not None:
                    self.send_header("Retry-After", server.retry_after)
                self.send_header("Content-Type", "application/xml")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.host = f"127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def close(self) -> None:
        self.released.set()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(monkeypatch) -> Iterator[_Server]:
    """MusicBrainz and the Cover Art Archive pointed at a local server, with no rate limit and no real sleeps."""
    local = _Server()
    # urllib honours an environment proxy: the local server must be reached directly.
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(name, "127.0.0.1")
    for module in (mb_module, caa):
        monkeypatch.setattr(module, "hostname", local.host)
        monkeypatch.setattr(module, "https", False)
    monkeypatch.setattr(mb_module, "do_rate_limit", False)
    yield local
    local.close()


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """Every time.sleep during the test, recorded instead of slept: here, musicbrainzngs's between retries."""
    monkeypatch.setattr(retry, "uniform", lambda _a, _b: 0)
    slept: list[float] = []
    monkeypatch.setattr(mb_module.time, "sleep", slept.append)
    return slept


def _fetch_release() -> dict:
    return anyio.run(musicbrainz_tagger.Scraper().fetch_data, RELEASE_URL)


def test_safe_read_keeps_the_signature_the_wrapper_calls() -> None:
    # The bound wraps a private function of musicbrainzngs: an upgrade that changes it must fail here, not quietly.
    parameters = list(inspect.signature(musicbrainz._safe_read).parameters)

    assert parameters[:3] == ["opener", "req", "body"]
    assert "max_retries" in parameters
    assert mb_module._safe_read is musicbrainz._bounded_safe_read


def test_a_release_is_fetched_through_the_bounded_read(server) -> None:
    release = _fetch_release()

    assert release["title"] == "Album"
    assert len(server.requests) == 1


def test_a_release_link_with_a_query_is_fetched_as_its_release(server) -> None:
    release = anyio.run(musicbrainz_tagger.Scraper().fetch_data, f"{RELEASE_URL}?tab=tracks")

    assert release["title"] == "Album"
    assert RELEASE_ID in server.requests[0]


def test_a_failing_server_gets_three_requests_with_backoff(server, sleeps) -> None:
    # 503 every time: musicbrainzngs made 8 requests and slept 56 s, then raised NetworkError past the scrape.
    server.status = 503

    with pytest.raises(ScrapeError) as excinfo:
        _fetch_release()

    assert str(excinfo.value) == "MusicBrainz: Request failed (HTTP 503)."
    assert len(server.requests) == 3
    assert sleeps == [3.0, 10.0]


def test_a_server_that_does_not_answer_times_out(server, sleeps, monkeypatch) -> None:
    # No timeout at all: a server that stopped answering held the scrape, and STORE_LOCK under salmon web.
    monkeypatch.setattr(musicbrainz, "REQUEST_TIMEOUT", 0.2, raising=False)
    server.delay = 3

    started = time.monotonic()
    with pytest.raises(ScrapeError):
        _fetch_release()

    assert time.monotonic() - started < 2
    assert len(server.requests) == 3


def test_a_release_not_on_musicbrainz_is_said_so(server) -> None:
    server.status = 404

    with pytest.raises(ScrapeError) as excinfo:
        _fetch_release()

    assert str(excinfo.value) == "MusicBrainz: Not found."
    assert len(server.requests) == 1


def test_a_failed_search_is_a_scrape_error(server, sleeps) -> None:
    server.status = 503

    with pytest.raises(ScrapeError) as excinfo:
        anyio.run(musicbrainz_search.Searcher().search_releases, "artist album", 5)

    assert str(excinfo.value) == "MusicBrainz: Request failed (HTTP 503)."
    assert len(server.requests) == 3


def test_a_failed_cover_art_archive_request_is_no_cover_and_says_so(server, sleeps, capsys) -> None:
    # The cover lookup caught a 404 only: a Cover Art Archive that failed failed the whole scrape.
    server.status = 503
    release = {"id": RELEASE_ID, "cover-art-archive": {"front": "true"}}

    cover = musicbrainz_tagger.Scraper().parse_cover_url(release)

    out = capsys.readouterr().out
    assert cover is None
    assert "Cover Art Archive request failed (NetworkError): no cover." in out
    assert len(server.requests) == 3


def test_a_cover_art_archive_answer_that_is_not_json_is_no_cover_and_says_so(server, capsys) -> None:
    # A 200 with a page in place of the image list raised JSONDecodeError past the cover lookup and failed the scrape.
    server.body = "<html>Not the image list</html>"
    release = {"id": RELEASE_ID, "cover-art-archive": {"front": "true"}}

    cover = musicbrainz_tagger.Scraper().parse_cover_url(release)

    out = capsys.readouterr().out
    assert cover is None
    assert "Cover Art Archive request failed (JSONDecodeError): no cover." in out
    assert len(server.requests) == 1


def test_cover_lookup_does_not_block_the_scrape_event_loop(monkeypatch) -> None:
    cover_url = "https://coverartarchive.org/cover.jpg"

    async def fetch_data(self, url, rls_id=None):
        return {
            "id": RELEASE_ID,
            "title": "Album",
            "cover-art-archive": {"front": "true"},
            "medium-list": [],
        }

    async def scrape():
        loop = asyncio.get_running_loop()
        released = threading.Event()

        def get_image_list(release_id):
            assert release_id == RELEASE_ID
            # Only the event loop can release the blocking lookup. If it runs on that
            # loop, the wait expires instead; no real network request is needed.
            loop.call_soon_threadsafe(released.set)
            assert released.wait(2), "Cover lookup blocked the scrape event loop"
            return {"images": [{"approved": True, "front": True, "image": cover_url}]}

        monkeypatch.setattr(musicbrainz_tagger.musicbrainzngs, "get_image_list", get_image_list)
        return await musicbrainz_tagger.Scraper().scrape_release(RELEASE_URL)

    monkeypatch.setattr(musicbrainz_tagger.Scraper, "fetch_data", fetch_data)

    metadata = anyio.run(scrape)

    assert metadata["title"] == "Album"
    assert metadata["cover"] == cover_url


@pytest.mark.parametrize("join", [" feat. ", " ft. ", " featuring ", " Feat. ", " ft "])
def test_a_featured_join_makes_the_artists_after_it_guests(join) -> None:
    # Only " feat. " did: B after " ft. " or " featuring " was a main artist (#910).
    credits = [{"artist": {"name": "A"}}, join, {"artist": {"name": "B"}}]

    assert musicbrainz_tagger.parse_artists(credits) == [("A", "main"), ("B", "guest")]


@pytest.mark.parametrize("join", [" & ", ", ", " x ", " with "])
def test_another_join_keeps_the_artists_after_it_main(join) -> None:
    credits = [{"artist": {"name": "A"}}, join, {"artist": {"name": "B"}}]

    assert musicbrainz_tagger.parse_artists(credits) == [("A", "main"), ("B", "main")]


def test_artist_credit_uses_the_name_on_the_release_not_a_later_rename() -> None:
    credits = [
        {"name": "Kanye West", "artist": {"name": "Ye"}},
        " feat. ",
        {"name": "JAY-Z", "artist": {"name": "JAY‐Z"}},
    ]
    assert musicbrainz_tagger.parse_artists(credits) == [("Kanye West", "main"), ("JAY-Z", "guest")]


def test_empty_credited_name_falls_back_to_the_artist_name() -> None:
    assert musicbrainz_tagger.parse_artists([{"name": "", "artist": {"name": "Artist"}}]) == [("Artist", "main")]


@pytest.mark.parametrize("release_title", ["Release Mix", None, ""])
def test_release_track_title_takes_precedence_over_the_recording_title(release_title) -> None:
    track = {
        "number": "1",
        "recording": {"title": "Recording Mix", "artist-credit": [{"artist": {"name": "Artist"}}]},
    }
    if release_title is not None:
        track["title"] = release_title
    soup = {"medium-list": [{"position": "1", "track-list": [track]}]}

    parsed = anyio.run(musicbrainz_tagger.Scraper().parse_tracks, soup)

    assert parsed["1"]["1"]["title"] == (release_title or "Recording Mix")


def test_requests_identify_the_app_version_and_real_project_contact(server):
    _fetch_release()
    assert len(server.user_agents) == 1
    agent = server.user_agents[0]
    assert agent is not None
    assert f"smoked-salmon/{get_version()}" in agent
    assert "https://github.com/smokin-salmon/smoked-salmon" in agent
    assert "noreply@salm.on" not in agent


@pytest.mark.parametrize("status", [403, 503])
def test_browser_check_is_named_without_exposing_response_details(server, sleeps, status):
    server.status = status
    server.body = "<html><title>Verifying your browser</title><body>private-token=secret</body></html>"
    with pytest.raises(ScrapeError) as caught:
        _fetch_release()
    assert "Browser verification page" in str(caught.value)
    assert f"HTTP {status}" in str(caught.value)
    assert "private-token" not in str(caught.value)
    assert "secret" not in str(caught.value)
    assert len(server.requests) == 1
    assert sleeps == []


def test_ordinary_html_503_is_not_called_a_browser_check(server, sleeps):
    server.status = 503
    server.body = "<html><title>503 Service Temporarily Unavailable</title><body>Try later</body></html>"
    with pytest.raises(ScrapeError, match=r"Request failed \(HTTP 503\)"):
        _fetch_release()
    assert len(server.requests) == 3


def test_transient_outage_recovers_before_reporting_failure(server, sleeps, capsys):
    server.statuses = [503, 503, 200]
    assert _fetch_release()["title"] == "Album"
    assert len(server.requests) == 3
    assert sleeps == [3.0, 10.0]
    assert "retrying" in capsys.readouterr().out


@pytest.mark.parametrize("status", [429, 503])
def test_server_retry_after_is_honoured(server, sleeps, status):
    server.statuses = [status, 200]
    server.retry_after = "20"
    assert _fetch_release()["title"] == "Album"
    assert sleeps == [20.0]
    assert len(server.requests) == 2


def test_long_cooldown_stops_without_retrying_early(server, sleeps):
    server.status = 429
    server.retry_after = "120"
    with pytest.raises(ScrapeError, match="HTTP 429"):
        _fetch_release()
    assert len(server.requests) == 1
    assert sleeps == []


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_permanent_http_errors_are_not_retried(server, sleeps, status):
    server.status = status
    with pytest.raises(ScrapeError):
        _fetch_release()
    assert len(server.requests) == 1
    assert sleeps == []
