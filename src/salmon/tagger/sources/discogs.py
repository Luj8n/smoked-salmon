import re
from collections import defaultdict

from salmon.common import RE_FEAT_JOIN
from salmon.common.strings import strip_numbered_suffix
from salmon.sources import DiscogsBase
from salmon.tagger.sources.base import MetadataMixin

VALID_EDITION_TITLES = {
    "Remastered",
    "Reissue",
    "Repress",
    "Club Edition",
    "Deluxe Edition",
    "Enhanced",
    "Limited Edition",
    "Mixed",
    "Partially Mixed",
    "Promo",
    "Special Edition",
    "Mono",
    "Quadraphonic",
    "Ambisonic",
    "Unofficial Release",
}

ROLES = {
    "Composed By": "composer",
    "Producer": "producer",
    "Featuring": "guest",
    "Vocals": "guest",
    "Featuring [Vocals]": "guest",
    "Remix": "remixer",
    "Arranged By": "arranger",
    "Conductor": "conductor",
    "Orchestra": "main",
}

# Explicit performance credits commonly used on classical releases. Do not infer performance
# from unrelated credits such as production, liner notes, or orchestration.
_PERFORMANCE_ROLES = {
    "Piano",
    "Violin",
    "Viola",
    "Cello",
    "Double Bass",
    "Flute",
    "Oboe",
    "Clarinet",
    "Bassoon",
    "French Horn",
    "Trumpet",
    "Trombone",
    "Tuba",
    "Harp",
    "Guitar",
    "Organ",
    "Harpsichord",
    "Percussion",
    "Vocals",
    "Voice",
    "Soloist",
}


def _performs(credit):
    return any(role.split("[", 1)[0].strip() in _PERFORMANCE_ROLES for role in credit["role"].split(","))


def _performer_names(credits):
    return {sanitize_artist_name(a["name"]) for a in credits if _performs(a)}


RELEASE_TYPES = {
    "Album": "Album",
    "Mini-Album": "Album",
    "EP": "EP",
    "Sampler": "EP",
    "Single": "Single",
    "Maxi-Single": "Single",
    "Compilation": "Compilation",
    "Mixtape": "Mixtape",
}


def _clean_discogs_position(position: str, default_disc: int) -> tuple[str, str]:
    """Parse raw Discogs position string into clean disc and track numbers.

    Examples:
        '1-01' -> ('1', '1')
        '2-15' -> ('2', '15')
        '01'   -> ('1', '1')
    """
    pos = str(position).strip().upper()

    # Handles "1-01", "2-15" multi-disc format
    match = re.match(r"^(\d+)-(\d+)([A-Z]*)$", pos)
    if match:
        return str(int(match[1])), str(int(match[2])) + match[3]

    # Handles "01", "02", "1"
    if pos.isdecimal():
        return str(default_disc), str(int(pos))

    return str(default_disc), pos


def _track_entries(entries):
    """Flatten index headings, carrying the work's title and credits into its playable children."""
    for entry in entries:
        if entry["type_"] != "index":
            yield entry
            continue
        for child in _track_entries(entry.get("sub_tracks", [])):
            child = {**child, "title": f"{entry['title'].strip()}: {child['title'].strip()}"}
            if "artists" not in child and "artists" in entry:
                child["artists"] = entry["artists"]
            child["extraartists"] = [*entry.get("extraartists", []), *child.get("extraartists", [])]
            yield child


def _credit_applies(scope: str, position: str, positions: list[str]) -> bool:
    """Discogs credits either every track, named positions, or inclusive '1 to 4' ranges."""
    if not scope.strip():
        return True
    for part in scope.upper().split(","):
        part = part.strip()
        if part == position:
            return True
        ends = part.split(" TO ")
        if (
            len(ends) == 2
            and all(end in positions for end in ends)
            and position in positions
            and positions.index(ends[0]) <= positions.index(position) <= positions.index(ends[1])
        ):
            return True
    return False


def _classical_track(soup, track, positions, composers):
    """Apply explicit classical roles, including a work's composer, without turning composers into performers."""
    position = track.get("position", "").strip().upper()
    extra = [
        art
        for art in soup.get("extraartists", [])
        if (
            any(role.strip() in {"Composed By", "Conductor", "Orchestra"} for role in art["role"].split(","))
            or _performs(art)
        )
        and _credit_applies(art.get("tracks", ""), position, positions)
    ]
    extra.extend(track.get("extraartists", []))
    composers_only = composers - _performer_names(extra)
    album_artists = [a for a in soup["artists"] if sanitize_artist_name(a["name"]) not in composers_only]
    credits = track.get("artists", album_artists)
    performers = [a for a in credits if sanitize_artist_name(a["name"]) not in composers_only]
    # An index can credit the work's composer alone; its movements still have the release's performers.
    if credits and not performers:
        performers = album_artists
    return {**track, "artists": performers, "extraartists": extra}


class Scraper(DiscogsBase, MetadataMixin):
    def parse_release_title(self, soup):
        return soup["title"]

    def parse_cover_url(self, soup):
        try:
            return soup["images"][0]["resource_url"]
        except (KeyError, IndexError):
            return None

    def parse_genres(self, soup):
        return set(soup["genres"])

    def parse_release_year(self, soup):
        return int(soup["year"])

    def parse_release_date(self, soup):
        if "released" in soup and re.match(r"\d{4}-\d{2}-\d{2}", soup["released"]):
            return soup["released"]

    def parse_edition_title(self, soup):
        if soup["formats"] and "descriptions" in soup["formats"][0]:
            return (
                " / ".join([w for w in soup["formats"][0]["descriptions"] if any(v in w for v in VALID_EDITION_TITLES)])
                or None
            )

    def parse_release_label(self, soup):
        if soup["labels"]:
            return sanitize_artist_name(soup["labels"][0]["name"])
        return "Not On Label"

    def parse_release_catno(self, soup):
        if soup["labels"] and soup["labels"][0]["catno"] != "none":
            return soup["labels"][0]["catno"]

    def parse_release_type(self, soup):
        if "formats" in soup and soup["formats"] and "descriptions" in soup["formats"][0]:
            try:
                return next(iter(RELEASE_TYPES[f] for f in soup["formats"][0]["descriptions"] if f in RELEASE_TYPES))
            except StopIteration:
                return

    async def parse_tracks(self, soup):
        tracks = defaultdict(dict)
        cur_disc = 1
        entries = list(_track_entries(soup["tracklist"]))
        classical = "Classical" in soup.get("genres", [])
        positions = [t.get("position", "").strip().upper() for t in entries if t["type_"] == "track"]
        composers = {
            sanitize_artist_name(a["name"])
            for a in [*soup.get("extraartists", []), *(a for t in entries for a in t.get("extraartists", []))]
            if "Composed By" in [role.strip() for role in a["role"].split(",")]
        }
        for track in entries:
            if track["type_"] == "heading" and tracks:
                cur_disc += 1
            elif track["type_"] == "track":
                raw_pos = track.get("position", "")
                disc_str, track_num = _clean_discogs_position(raw_pos, cur_disc)
                disc_num = int(disc_str)

                credited = _classical_track(soup, track, positions, composers) if classical else track
                artists, variations = parse_credits(soup["artists"], credited)
                tracks[disc_str][track_num] = self.generate_track(
                    trackno=track_num,
                    discno=disc_num,
                    artists=artists,
                    title=track["title"],
                    duration=parse_duration(track.get("duration")),
                    # Each under the name the release credits it too (the ANV), which the files may use instead.
                    credited_remixers=[
                        *(name for name, role in artists if role == "remixer"),
                        *(v["anv"] for v in variations if v["role"] == "remixer"),
                    ],
                )
                # Only the merge reads this, to match tag artists spelled the way the release credits them.
                tracks[disc_str][track_num]["artist_variations"] = variations
        return dict(tracks)


def parse_duration(duration):
    """Convert Discogs' "m:ss" / "h:mm:ss" track duration to seconds."""
    if not duration:
        return None
    parts = duration.strip().split(":")
    # isdecimal(), not isdigit(): isdigit() accepts some Unicode digits (superscript "2") that int() rejects.
    if not all(p.isdecimal() for p in parts) or not 2 <= len(parts) <= 3:
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return seconds


def parse_artists(artist_soup, track):
    """
    Generate the artists list from the artist dictionary provided with
    each track.
    """
    return parse_credits(artist_soup, track)[0]


def parse_credits(artist_soup, track):
    """
    Generate the artists list of a track and, beside it, the name each artist is
    credited under on this release when that differs from the main Discogs name
    (the ANV): `[{"name": ..., "role": ..., "anv": ...}]`, plain data that survives JSON.
    """
    credits = track["artists"] if "artists" in track else [art for art in artist_soup if art["name"] != "Various"]
    artists, variations = _credited_artists(credits)
    if "extraartists" in track:
        performers = _performer_names(track["extraartists"])
        for art in track["extraartists"]:
            for role in art["role"].split(","):
                role = role.strip()
                credit = (sanitize_artist_name(art["name"]), ROLES.get(role))
                if role in ROLES and credit not in artists:
                    artists.append(credit)
                    _note_variation(variations, credit, art)
        for name, role in artists:
            # Production and arranging credits do not replace a performing credit.
            if (
                role not in {"main", "producer", "arranger"}
                and (name, "main") in artists
                and (name, "arranger") not in artists
                and not (name in performers and (name, "composer") in artists)
            ):
                artists.remove((name, "main"))
    return artists, variations


def _credited_artists(credits):
    """Name the artists of a credit list, making those after a feat. join guests."""
    artists = []
    variations = []
    role = "main"
    for art in credits:
        if art["name"] == "No Artist":
            continue
        credit = (sanitize_artist_name(art["name"]), role)
        artists.append(credit)
        _note_variation(variations, credit, art)
        if RE_FEAT_JOIN.fullmatch((art.get("join") or "").strip()):
            role = "guest"
    return artists, variations


def _note_variation(variations, credit, art):
    """Add the name `art` is credited under on the release to `variations` when it is not the main name."""
    anv = sanitize_artist_name((art.get("anv") or "").strip())
    if anv and anv != credit[0]:
        variations.append({"name": credit[0], "role": credit[1], "anv": anv})


def sanitize_artist_name(name):
    """
    Remove parenthentical number disambiguation bullshit from artist names,
    as well as the asterisk stuff.
    """
    return strip_numbered_suffix(re.sub(r"\*+$", "", name))
