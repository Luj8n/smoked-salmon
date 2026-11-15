import anyio
import pytest

from salmon.errors import ScrapeError
from salmon.sources import apple_music
from salmon.sources.apple_music import AppleMusicBase
from salmon.sources.qobuz import QobuzBase
from salmon.tagger.sources.qobuz import Scraper as QobuzScraper


@pytest.mark.parametrize("mode", ["normal", "empty", "endless"])
def test_apple_album_pagination_is_complete_or_fails_with_bounded_requests(monkeypatch, mode):
    requests = []

    async def token():
        return "test-token"

    async def get_json(self, url, params, headers):
        requests.append(dict(params))
        if len(requests) == 1:
            return {
                "data": [
                    {
                        "attributes": {"name": "Album"},
                        "relationships": {"tracks": {"data": [{"id": "1"}], "next": "page2"}},
                    }
                ]
            }
        # Stop an unfixed infinite loop safely, without hitting a real service.
        assert len(requests) <= 3, "Album pagination did not stop at its bound"
        if mode == "empty":
            return {"data": [], "next": "page2"}
        return {"data": [{"id": str(len(requests))}], "next": None if mode == "normal" else "more"}

    monkeypatch.setattr(AppleMusicBase, "_get_token", staticmethod(token))
    monkeypatch.setattr(AppleMusicBase, "get_json", get_json)
    monkeypatch.setattr(apple_music, "MAX_TRACK_PAGES", 3, raising=False)
    if mode == "normal":
        data = anyio.run(AppleMusicBase().fetch_data, "https://music.apple.com/us/album/-/123")
        assert data["tracks"] == [{"id": "1"}, {"id": "2"}]
        assert requests[-1]["offset"] == 1
    else:
        with pytest.raises(ScrapeError, match="empty|pages"):
            anyio.run(AppleMusicBase().fetch_data, "https://music.apple.com/us/album/-/123")
        assert len(requests) <= 3


@pytest.mark.parametrize("total", [1, 2, None])
@pytest.mark.parametrize("scraper", [QobuzBase, QobuzScraper])
def test_qobuz_does_not_return_an_advertised_partial_album(monkeypatch, total, scraper):
    async def get_json(self, *args, **kwargs):
        return {
            "id": "album",
            "title": "Album",
            "tracks": {"items": [{"id": 1}], "offset": 0, "limit": 1, "total": total},
        }

    monkeypatch.setattr(QobuzBase, "require_configured", lambda self: None)
    monkeypatch.setattr(QobuzBase, "configured", staticmethod(lambda: True))
    monkeypatch.setattr(QobuzBase, "get_json", get_json)
    if total == 2:
        with pytest.raises(ScrapeError, match="incomplete tracklist"):
            anyio.run(scraper().fetch_data, "https://qobuz.com/album/album")
    else:
        assert anyio.run(scraper().fetch_data, "https://qobuz.com/album/album")["tracks"]["items"] == [{"id": 1}]
