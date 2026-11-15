import anyio
import pytest

from salmon.tagger.pairing import pair_by_content
from salmon.tagger.sources.qobuz import Scraper


def credits(album_artist, album_artists, *tracks):
    soup = {
        "artist": {"name": album_artist},
        "artists": album_artists,
        "tracks": {"items": [{"title": "Song", "track_number": i, **t} for i, t in enumerate(tracks, 1)]},
    }
    return [t["artists"] for t in anyio.run(Scraper().parse_tracks, soup)["1"].values()]


def test_qobuz_credits_each_explicit_album_main_as_main_when_on_the_track():
    mains = [{"name": n, "roles": ["main-artist"]} for n in ("Ella Fitzgerald", "Louis Armstrong")]
    assert credits(
        "Ella Fitzgerald",
        mains,
        {"performers": "Ella Fitzgerald, MainArtist, Vocalist - Louis Armstrong, MainArtist, Vocalist"},
    ) == [[("Ella Fitzgerald", "main"), ("Louis Armstrong", "main")]]


def test_qobuz_does_not_add_an_album_collaborator_to_every_track():
    mains = [{"name": n, "roles": ["main-artist"]} for n in ("Lead", "Other")]
    assert credits(
        "Lead",
        mains,
        {"performers": "Lead, MainArtist - Other, MainArtist"},
        {"performers": "Lead, MainArtist - Guest, FeaturedArtist"},
    ) == [[("Lead", "main"), ("Other", "main")], [("Lead", "main"), ("Guest", "guest")]]


def test_qobuz_name_variations_do_not_make_a_band_its_own_guest():
    assert credits("Sigur Rós", [], {"performer": {"name": "Sigur Ros"}, "performers": "Sigur Ros , MainArtist"}) == [
        [("Sigur Rós", "main")]
    ]


def test_qobuz_known_names_with_commas_are_not_split_by_the_roles():
    assert credits("Earth, Wind & Fire", [], {"performers": "Earth, Wind & Fire, MainArtist, Performer"}) == [
        [("Earth, Wind & Fire", "main")]
    ]


def test_qobuz_missing_album_artist_uses_the_track_performer():
    assert credits(None, [], {"performer": {"name": "Artist"}}) == [[("Artist", "main")]]


@pytest.mark.parametrize("track", [{"performer": {"name": "Other"}}, {"performers": "Other, MainArtist"}])
def test_qobuz_does_not_add_the_primary_album_artist_to_another_mains_solo_track(track):
    mains = [{"name": n, "roles": ["main-artist"]} for n in ("Lead", "Other")]
    assert credits("Lead", mains, track) == [[("Other", "main")]]


@pytest.mark.parametrize("track", [{}, {"performers": "Guest, FeaturedArtist"}])
def test_qobuz_keeps_album_primary_as_fallback_without_a_known_track_main(track):
    got = credits("Lead", [], track)[0]
    assert got[0] == ("Lead", "main")
    if track:
        assert got[1:] == [("Guest", "guest")]


@pytest.mark.parametrize("placeholder", ["Various", "Various Artists", " various artists "])
def test_qobuz_compilations_keep_track_performers_main_and_featured_artists_guest(placeholder):
    assert credits(
        placeholder,
        [{"name": placeholder, "roles": ["main-artist"]}],
        {"performer": {"name": "Singer"}, "performers": "Singer, MainArtist - Guest, FeaturedArtist"},
    ) == [[("Singer", "main"), ("Guest", "guest")]]


def test_qobuz_various_in_a_real_band_name_is_not_a_placeholder():
    assert credits("Various Cruelties", [], {}) == [[("Various Cruelties", "main")]]


@pytest.mark.parametrize("duration", [128, None])
def test_qobuz_preserves_track_duration_for_pairing(duration):
    soup = {"artist": {"name": "Artist"}, "tracks": {"items": [{"title": "Song", "duration": duration}]}}
    parsed = anyio.run(Scraper().parse_tracks, soup)
    assert parsed["1"]["1"]["duration"] == duration


def test_qobuz_repeated_titles_pair_with_the_correct_audio_by_duration():
    soup = {
        "artist": {"name": "Artist"},
        "tracks": {
            "items": [
                {"title": "Song", "track_number": 1, "duration": 128},
                {"title": "Song", "track_number": 2, "duration": 256},
            ]
        },
    }
    source = list(anyio.run(Scraper().parse_tracks, soup)["1"].values())
    local = [{"title": "Song", "duration": 256}, {"title": "Song", "duration": 128}]

    pairing, _ = pair_by_content(local, source)

    assert pairing == [1, 0]
