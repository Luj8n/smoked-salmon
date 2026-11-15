import re
from typing import Any

import musicbrainzngs

from salmon import cfg
from salmon.common.strings import names_no_artist
from salmon.errors import ScrapeError
from salmon.search.base import IdentData, SearchMixin
from salmon.sources import MusicBrainzBase


def album_query(artists: list[str] | None, album: str) -> str:
    """An album search across the right Lucene fields; bare text only searches release titles."""

    def quoted(value: str) -> str:
        escaped = re.sub(r'([+\-&|!(){}\[\]^"~*?:\\/])', r"\\\1", value)
        return f'"{escaped}"'

    query = f"release:{quoted(album)}"
    names = dict.fromkeys(name for name in artists or [] if name.strip() and not names_no_artist(name))
    if names:
        artist_query = " OR ".join(f"artist:{quoted(name)}" for name in names)
        if len(names) > 1:
            artist_query = f"({artist_query})"
        query = f"{artist_query} AND {query}"
    return query


class Searcher(MusicBrainzBase, SearchMixin):
    async def search_releases(self, searchstr: str, limit: int) -> tuple[str, dict[str, Any]]:
        """Search for releases on MusicBrainz.

        Args:
            searchstr: Search string.
            limit: Maximum number of results.

        Returns:
            Tuple of (source name, releases dict).
        """
        releases = {}
        result = await self._call(musicbrainzngs.search_releases, searchstr, limit)
        for rls in result["release-list"]:
            try:
                artists = rls["artist-credit-phrase"]
                try:
                    track_count = rls["medium-track-count"]
                except KeyError:
                    track_count = None
                label = catno = ""
                if (
                    "label-info-list" in rls
                    and rls["label-info-list"]
                    and "label" in rls["label-info-list"][0]
                    and "name" in rls["label-info-list"][0]["label"]
                ):
                    label = rls["label-info-list"][0]["label"]["name"]
                    if "catalog_number" in rls["label-info-list"][0]:
                        catno = rls["label-info-list"][0]["catalog_number"]

                try:
                    source = rls["medium-list"][0]["format"]
                except KeyError:
                    source = None

                edition = self.edition(label, catno)

                if label.lower() not in cfg.upload.search.excluded_labels:
                    releases[rls["id"]] = (
                        IdentData(
                            artists,
                            rls["title"],
                            None,
                            track_count,
                            source or "",
                        ),
                        self.format_result(
                            artists,
                            rls["title"],
                            edition,
                            ed_title=source,
                            track_count=track_count,
                        ),
                    )
            except (TypeError, IndexError) as e:
                raise ScrapeError("Failed to parse scraped search results.") from e
            if len(releases) == limit:
                break
        return "MusicBrainz", releases
