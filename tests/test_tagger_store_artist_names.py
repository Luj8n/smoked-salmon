"""Store artist lists already separate artists; punctuation inside one name must survive."""

import pytest

from salmon.tagger.sources.deezer import Scraper as Deezer
from salmon.tagger.sources.tidal import Scraper as Tidal


@pytest.mark.parametrize("name", ["Earth, Wind & Fire", "Simon & Garfunkel", "Tyler, The Creator"])
def test_tidal_keeps_a_band_as_one_main_artist(name):
    artists, _ = Tidal().parse_artists([{"name": name}], "Let's Groove", None, [{"name": name}])
    assert artists == [(name, "main")]


def test_tidal_keeps_distinct_artist_entries_separate():
    artists, _ = Tidal().parse_artists(
        [{"name": "Earth, Wind & Fire"}, {"name": "Guest"}], "Song", None, [{"name": "Earth, Wind & Fire"}]
    )
    assert artists == [("Earth, Wind & Fire", "main"), ("Guest", "guest")]


@pytest.mark.parametrize("field", ["mainartist", "main_artist", "featuredartist", "featuring"])
def test_deezer_keeps_each_credited_artist_name_whole(field):
    role = "main" if field.startswith("main") else "guest"
    assert Deezer().parse_artists({field: ["Earth, Wind & Fire", "Other"]}, [], "Song") == [
        ("Earth, Wind & Fire", role),
        ("Other", role),
    ]


def test_deezer_fallback_artist_objects_keep_their_names_whole():
    assert Deezer().parse_artists({}, [{"ART_NAME": "Earth, Wind &amp; Fire"}], "Song") == [
        ("Earth, Wind & Fire", "main")
    ]


@pytest.mark.parametrize("contributors", [{"composer": ["Composer"]}, {"mainartist": []}, {"producer": ["Producer"]}])
def test_deezer_uses_fallback_performers_when_contributors_have_no_artist_roles(contributors):
    assert Deezer().parse_artists(contributors, [{"ART_NAME": "Performer"}], "Song") == [("Performer", "main")]


def test_deezer_does_not_replace_an_explicit_guest_role_with_fallback_main():
    assert Deezer().parse_artists({"featuredartist": ["Guest"]}, [{"ART_NAME": "Guest"}], "Song") == [
        ("Guest", "guest")
    ]
