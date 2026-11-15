import json

import anyio
import pytest
from bs4 import BeautifulSoup

from salmon.errors import ScrapeError
from salmon.sources import beatport
from salmon.sources.beatport import BeatportBase


def page(*, tracks=None, token: str | None = "anonymous-page-token", next_page=None):
    release = {"id": 1139534, "name": "Raise!", "track_count": 9}
    queries = [
        {"queryKey": ["release-1139534-recommendations"], "state": {"data": {"results": []}}},
        {"queryKey": ["release-1139534"], "state": {"data": release}},
    ]
    if tracks is not None:
        queries.append({"queryKey": ["tracks"], "state": {"data": {"results": tracks, "next": next_page}}})
    data = {"props": {"pageProps": {"dehydratedState": {"queries": queries}, "anonSession": {"access_token": token}}}}
    return BeautifulSoup(f'<script id="__NEXT_DATA__">{json.dumps(data)}</script>', "lxml")


def test_missing_embedded_tracks_use_the_pages_anonymous_session(monkeypatch):
    requests = []

    async def fetch_page(self, url, params):
        return page()

    async def get_json(self, url, params, headers):
        requests.append((url, dict(params), headers))
        number = params["page"]
        return {"results": [{"id": number}], "next": "next-page" if number == 1 else None}

    monkeypatch.setattr(BeatportBase, "fetch_page", fetch_page)
    monkeypatch.setattr(BeatportBase, "get_json", get_json)

    result = anyio.run(BeatportBase().fetch_data_scraping, "https://beatport.com/release/raise/1139534", None)

    assert result["release"]["id"] == 1139534
    assert result["tracks"] == [{"id": 1}, {"id": 2}]
    assert len(requests) == 2
    assert requests[0] == (
        "/catalog/tracks/",
        {"release_id": 1139534, "per_page": 100, "page": 1},
        {"Authorization": "Bearer anonymous-page-token"},
    )


def test_embedded_tracks_need_no_api_request(monkeypatch):
    async def fetch_page(self, url, params):
        return page(tracks=[{"id": 1}], token=None)

    async def get_json(*args, **kwargs):
        pytest.fail("Embedded tracks must not make another request")

    monkeypatch.setattr(BeatportBase, "fetch_page", fetch_page)
    monkeypatch.setattr(BeatportBase, "get_json", get_json)
    result = anyio.run(BeatportBase().fetch_data_scraping, "https://beatport.com/release/raise/1139534", None)
    assert result["tracks"] == [{"id": 1}]
    assert result["release"]["id"] == 1139534


@pytest.mark.parametrize("token", ["anonymous-page-token", None])
def test_incomplete_embedded_tracks_are_fetched_in_full_or_refused(monkeypatch, token):
    async def fetch_page(self, url, params):
        return page(tracks=[{"id": 1}], token=token, next_page="page2")

    async def get_json(self, url, params, headers):
        assert headers == {"Authorization": "Bearer anonymous-page-token"}
        return {"results": [{"id": 1}, {"id": 2}], "next": None}

    monkeypatch.setattr(BeatportBase, "fetch_page", fetch_page)
    monkeypatch.setattr(BeatportBase, "get_json", get_json)
    if token:
        result = anyio.run(BeatportBase().fetch_data_scraping, "https://beatport.com/release/raise/1139534", None)
        assert result["tracks"] == [{"id": 1}, {"id": 2}]
    else:
        with pytest.raises(ScrapeError, match="anonymous session"):
            anyio.run(BeatportBase().fetch_data_scraping, "https://beatport.com/release/raise/1139534", None)


def test_missing_tracks_and_session_are_reported(monkeypatch):
    async def fetch_page(self, url, params):
        return page(token=None)

    monkeypatch.setattr(BeatportBase, "fetch_page", fetch_page)
    with pytest.raises(ScrapeError, match="track data.*anonymous session"):
        anyio.run(BeatportBase().fetch_data_scraping, "https://beatport.com/release/raise/1139534", None)


def test_track_paging_is_bounded_and_does_not_return_a_partial_release(monkeypatch):
    requests = []

    async def fetch_page(self, url, params):
        return page()

    async def get_json(self, url, params, headers):
        requests.append(params["page"])
        return {"results": [{"id": params["page"]}], "next": "more"}

    monkeypatch.setattr(BeatportBase, "fetch_page", fetch_page)
    monkeypatch.setattr(BeatportBase, "get_json", get_json)
    monkeypatch.setattr(beatport, "MAX_TRACK_PAGES", 2)
    with pytest.raises(ScrapeError, match="more than 2 pages"):
        anyio.run(BeatportBase().fetch_data_scraping, "https://beatport.com/release/raise/1139534", None)
    assert requests == [1, 2]
