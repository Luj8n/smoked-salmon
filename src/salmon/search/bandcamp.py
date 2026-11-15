from typing import Any
from urllib.parse import urljoin, urlsplit

from salmon.errors import ScrapeError
from salmon.search.base import IdentData, SearchMixin
from salmon.sources import BandcampBase


class Searcher(BandcampBase, SearchMixin):
    search_api_url = "https://bandcamp.com/api/bcsearch_public_api/1/autocomplete_elastic"

    async def fetch_search_results(self, searchstr: str) -> list[dict[str, Any]]:
        # The public autocomplete endpoint works where the HTML search returns a Client Challenge.
        payload = {"search_text": searchstr, "search_filter": "", "full_page": False, "fan_id": None}
        async with self._session() as session, session.post(self.search_api_url, json=payload) as response:
            data = await self.handle_json_response(response)
        auto = data.get("auto") if isinstance(data, dict) else None
        results = auto.get("results") if isinstance(auto, dict) else None
        if not isinstance(results, list):
            raise ScrapeError("Bandcamp: Search response is missing its results.")
        return [item for item in results if isinstance(item, dict) and item.get("type") in ("a", "t")]

    async def search_releases(self, searchstr: str, limit: int) -> tuple[str, dict]:
        releases: dict[Any, Any] = {}
        if limit <= 0:
            return "Bandcamp", releases
        results = await self.fetch_search_results(searchstr)
        # Album tracks can rank above their parent album; keep that album within the result limit.
        for meta in sorted(results, key=lambda item: item["type"] != "a"):
            root, path = meta.get("item_url_root"), meta.get("item_url_path")
            title, artists = meta.get("name"), meta.get("band_name")
            if not isinstance(root, str) or not isinstance(path, str):
                continue
            if not isinstance(title, str) or not isinstance(artists, str):
                continue
            if not all(value.strip() for value in (root, path, title, artists)):
                continue
            try:
                url = urlsplit(urljoin(root, path))
                match = self.regex.fullmatch(url._replace(query="", fragment="").geturl())
            except ValueError:
                continue
            if not match or match[2] != {"a": "album", "t": "track"}[meta["type"]]:
                continue
            domain, release_type, slug = match.groups()
            title, artists = title.strip(), artists.strip()
            # Autocomplete supplies neither release dates nor album track counts. Fetch them only on selection.
            track_count = 1 if release_type == "track" else None
            shown_title = f"{title[:98]}.." if len(title) > 100 else title
            releases[(domain, release_type, slug)] = (
                IdentData(artists, title, None, track_count, "WEB"),
                self.format_result(artists, shown_title, domain.removesuffix(".bandcamp.com"), track_count=track_count),
            )
            if len(releases) == limit:
                break
        return "Bandcamp", releases
