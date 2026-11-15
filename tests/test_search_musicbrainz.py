import anyio
import pytest

from salmon.search import run_metasearch
from salmon.sources.musicbrainz import MusicBrainzBase


@pytest.mark.parametrize(
    ("artists", "album", "expected"),
    [
        (["Earth, Wind & Fire"], "Raise!", 'artist:"Earth, Wind \\& Fire" AND release:"Raise\\!"'),
        (["One", "Two"], "Album", '(artist:"One" OR artist:"Two") AND release:"Album"'),
        (["Various Artists"], "Album", 'release:"Album"'),
        (None, "Album", 'release:"Album"'),
        (["Artist"], None, "free text"),
    ],
)
def test_metadata_search_uses_musicbrainz_release_and_artist_fields(monkeypatch, artists, album, expected):
    queries = []

    async def call(self, func, query, limit):
        queries.append((query, limit))
        return {"release-list": []}

    monkeypatch.setattr(MusicBrainzBase, "_call", call)
    anyio.run(
        lambda: run_metasearch(
            ["free text"],
            sources={"MusicBrainz": None},
            artists=artists,
            album=album,
            limit=5,
        )
    )
    assert queries == [(expected, 5)]


def test_structured_musicbrainz_search_is_not_repeated_for_each_artist(monkeypatch):
    queries = []

    async def call(self, func, query, limit):
        queries.append(query)
        return {"release-list": []}

    monkeypatch.setattr(MusicBrainzBase, "_call", call)
    anyio.run(
        lambda: run_metasearch(
            ["one album", "two album"],
            sources={"MusicBrainz": None},
            artists=["One", "Two"],
            album="Album",
        )
    )
    assert queries == ['(artist:"One" OR artist:"Two") AND release:"Album"']
