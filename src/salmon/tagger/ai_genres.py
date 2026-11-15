"""Small checks on AI genre edits; unfamiliar musical genres are not a closed-list error."""

from difflib import get_close_matches

from salmon.common.strings import comparable
from salmon.constants import GENRE_LIST
from salmon.tagger.sources.base import standardize_genres

_KNOWN = {comparable(name): name for names in GENRE_LIST.values() for name in sorted(names)}
_NON_GENRES = {
    "flac",
    "mp3",
    "lossless",
    "web",
    "cd",
    "vinyl",
    "tidal",
    "qobuz",
    "deezer",
    "bandcamp",
    "applemusic",
    "spotify",
    "beatport",
    "discogs",
    "musicbrainz",
}


def _genres(values):
    if not isinstance(values, list):
        return []
    normalized = standardize_genres([g.strip() for g in values if isinstance(g, str) and g.strip()])
    unique = {}
    for genre in normalized:
        if key := comparable(genre):
            unique.setdefault(key, genre)
    return list(unique.values())


def _cited_genres(citations):
    names = []
    for citation in citations:
        for support in citation.get("supports", []):
            if isinstance(support, str) and support.casefold().startswith("genres:"):
                # Keep an appended explanation out of the name, e.g. "Shoegaze — tagged shoegazer".
                names.extend(_genres([support.split(":", 1)[1].partition(" — ")[0]]))
    return {comparable(name) for name in names}


def guard_genres(metadata, review, opened_citations):
    """Require named genre evidence for additions and some accepted genre evidence before removing old tags.

    Citations are claims by the model, restricted by the caller to pages it actually opened. This does not
    independently establish that a genre describes the music. Possible typos are flagged, never guessed away.
    """
    proposed = review.get("metadata")
    if not isinstance(proposed, dict) or "genres" not in proposed:
        return []
    before = _genres(metadata.get("genres"))
    previous = {comparable(g) for g in before}
    evidence = _cited_genres(opened_citations)
    non_genres = (
        _NON_GENRES
        | {comparable(name) for name, _ in metadata.get("artists", [])}
        | {comparable(metadata.get("label") or "")}
    )
    accepted, warnings = [], []
    for genre in _genres(proposed["genres"]):
        key = comparable(genre)
        if key in _NON_GENRES or (key in non_genres and key not in _KNOWN):
            warnings.append(f"Ignored non-genre AI tag: {genre}.")
        elif key not in previous and key not in evidence:
            warnings.append(f"Ignored AI genre {genre}: no named genre citation from an opened page.")
        else:
            accepted.append(genre)
    # An empty answer or unrelated citation must not erase the current genres.
    if not evidence.intersection(comparable(g) for g in accepted):
        missing = [g for g in before if comparable(g) not in {comparable(a) for a in accepted}]
        if missing:
            warnings.append("Kept existing genres: removals need genre evidence from an opened page.")
            accepted.extend(missing)
    for genre in accepted:
        key = comparable(genre)
        if key not in _KNOWN and (matches := get_close_matches(key, _KNOWN, n=1, cutoff=0.88)):
            warnings.append(f"Check genre spelling: {genre!r}; perhaps {_KNOWN[matches[0]]!r}.")
    proposed["genres"] = accepted
    return warnings
