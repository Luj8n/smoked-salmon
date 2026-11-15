"""Bandcamp HTML search can be challenged even while its public autocomplete API works."""

from typing import Any

import anyio
import asyncclick as click
import pytest
from aiohttp import web

from salmon.errors import ScrapeError
from salmon.search.bandcamp import Searcher


def item(kind="a", path: Any = "/album/circus", **overrides):
    return {
        "type": kind,
        "name": "Circus",
        "band_name": "Sandhouse",
        "item_url_root": "https://sandhouse.bandcamp.com",
        "item_url_path": path,
        **overrides,
    }


async def search_with_response(body, limit=5, status=200):
    calls = []

    async def handler(request):
        calls.append((request.method, await request.json()))
        if isinstance(body, str):
            return web.Response(text=body, status=status)
        return web.json_response(body, status=status)

    app = web.Application()
    app.router.add_post("/search", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        searcher = Searcher()
        searcher.search_api_url = f"http://127.0.0.1:{runner.addresses[0][1]}/search"
        result = await searcher.search_releases("sandhouse circus", limit)
        return result, calls
    finally:
        await runner.cleanup()


def test_search_uses_public_api_and_keeps_album_before_its_tracks():
    # The live response lists three tracks before the album. A small limit must not hide the album.
    tracks = [item("t", f"/track/{name}") for name in ("circus", "grown", "undefeated")]
    (store, releases), calls = anyio.run(search_with_response, {"auto": {"results": [*tracks, item()]}}, 1)

    assert store == "Bandcamp"
    assert calls == [
        ("POST", {"search_text": "sandhouse circus", "search_filter": "", "full_page": False, "fan_id": None})
    ]
    assert list(releases) == [("sandhouse.bandcamp.com", "album", "circus")]
    ident, shown = next(iter(releases.values()))
    assert (ident.artist, ident.album, ident.track_count, ident.year) == ("Sandhouse", "Circus", None, None)
    assert click.unstyle(shown) == "Sandhouse - Circus sandhouse"
    assert Searcher.format_url(next(iter(releases))) == "https://sandhouse.bandcamp.com/album/circus"


def test_search_retains_tracks_skips_non_releases_and_deduplicates_clean_urls():
    results = [
        None,
        item("b"),
        item(path=None),
        item(name=None),
        item(path="mailto:someone@example.org"),
        item(path="https://[broken/album/circus"),
        item(path="/track/wrong-type"),
        item(path="https://sandhouse.bandcamp.com/album/circus?from=search#top"),
        item(),
        item("t", "/track/circus"),
    ]
    (_, releases), calls = anyio.run(search_with_response, {"auto": {"results": results}})
    assert list(releases) == [
        ("sandhouse.bandcamp.com", "album", "circus"),
        ("sandhouse.bandcamp.com", "track", "circus"),
    ]
    assert list(releases.values())[1][0].track_count == 1
    assert len(calls) == 1


@pytest.mark.parametrize("body", [{}, {"auto": None}, {"auto": {"results": {}}}, [], "<title>Client Challenge</title>"])
def test_unexpected_response_is_an_error_not_an_empty_search(body):
    with pytest.raises(ScrapeError, match="Bandcamp.*(results|JSON)"):
        anyio.run(search_with_response, body)


def test_empty_search_results_are_valid():
    (store, releases), calls = anyio.run(search_with_response, {"auto": {"results": []}})
    assert (store, releases) == ("Bandcamp", {})
    assert len(calls) == 1


def test_api_failure_is_reported():
    with pytest.raises(ScrapeError, match="Status code 429"):
        anyio.run(search_with_response, {}, 5, 429)


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_limit_does_not_request(limit):
    (store, releases), calls = anyio.run(search_with_response, {}, limit)
    assert (store, releases, calls) == ("Bandcamp", {}, [])
