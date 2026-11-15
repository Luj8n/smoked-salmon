"""Keep the selected source's explicit artist roles when merging tags and other sources."""

from salmon.common.strings import comparable

_PERFORMING_ROLES = {"main", "guest", "remixer", "conductor", "composer"}


def source_roles(track):
    roles = {}
    for name, role in track["artists"]:
        if role in _PERFORMING_ROLES:
            roles.setdefault(comparable(name), []).append(role)
    for variation in track.get("artist_variations", []):
        if variation["role"] in roles.get(comparable(variation["name"]), []):
            roles.setdefault(comparable(variation["anv"]), []).append(variation["role"])
    return roles


def _apply(artists, roles, aliases):
    corrected = []
    seen = set()
    for name, role in artists:
        preferred = roles.get(comparable(name)) if role in {"main", "guest"} else None
        for selected in preferred or [role]:
            key = (aliases.get(comparable(name), comparable(name)), selected)
            if key not in seen:
                corrected.append((name, selected))
                seen.add(key)
    return corrected


def merge_roles(base, incoming, selected, preferences):
    """Only resolve known names' main/guest roles; keep unlisted artists and additional production credits.

    Preferences belong to one album merge, keyed by the local track object so pairing/renumbering cannot move
    them to a different song. None preserves combine_tracks' standalone behaviour.
    """
    if preferences is None:
        return list(base["artists"]), incoming["artists"]
    if selected:
        preferences[id(base)] = {}
    roles = preferences.setdefault(id(base), {})
    additions = source_roles(incoming)
    for variation in incoming.get("artist_variations", []):
        name, alias = comparable(variation["name"]), comparable(variation["anv"])
        # A later source can establish an alias, but cannot change a role already chosen
        # for either spelling by an earlier source.
        if preferred := roles.get(name) or roles.get(alias):
            additions[name] = additions[alias] = preferred
    for name, known_roles in additions.items():
        roles.setdefault(name, known_roles)
    aliases = {comparable(v["anv"]): comparable(v["name"]) for v in incoming.get("artist_variations", [])}
    return _apply(base["artists"], roles, aliases), _apply(incoming["artists"], roles, aliases)
