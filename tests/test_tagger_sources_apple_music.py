from copy import deepcopy

import anyio

from salmon.tagger.combine import combine_tracks
from salmon.tagger.sources.apple_music import Scraper


def test_apple_track_keys_match_track_numbers_and_do_not_duplicate_during_merge():
    parsed = anyio.run(
        Scraper().parse_tracks,
        {
            "tracks": [
                {
                    "type": "songs",
                    "attributes": {"name": "Song", "artistName": "Artist", "trackNumber": 1, "discNumber": 1},
                }
            ]
        },
    )
    source_track = deepcopy(next(iter(parsed["1"].values())))

    combined = combine_tracks(parsed, {"1": {"1": source_track}}, False)

    assert list(combined["1"]) == ["1"]
    assert len(combined["1"]) == 1
