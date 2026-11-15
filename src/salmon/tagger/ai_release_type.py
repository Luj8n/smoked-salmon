"""AI review uses RED's release categories for both RED and OPS.

Guidance paraphrased from user-supplied wiki snapshots (2026-11-15):
RED Category guidelines - Release types, r15 (wiki alias: releasetype);
OPS Uploading ~ Titles of Releases & Groups, r37 (article 12), and Multiple Artists, r10 (article 140).
Naming: RED Capitalization Guidelines r9 (article 27), Editing Standards r12 (article 391),
and the supplied RED Torrent Naming Irregularities and OPS Edition Information Guidelines r28 (article 18).
The supplied naming/edition guides govern presentation, not automatic upload eligibility.
"""

from salmon.constants import RELEASE_TYPES

RELEASE_TYPE_GUIDANCE = """Release type (rls_type), using RED's categories for both RED and OPS:
Choose a schema value and briefly explain the classification, citing release evidence when browsing.
Prefer artist/label intent and release history; track count and duration are clues, not fixed cutoffs.
- Album: a normal original-material release; reissues with bonus tracks/discs remain albums.
- EP: artist-designated EP/mini-album; neither a short runtime nor 'EP' in a store title proves it.
- Single: one main song with possible B-sides/remixes; maxi-singles can be long and have many tracks.
- Anthology: one artist's retrospective, archival collection, or set of previously released albums.
- Compilation: a collection by different artists; prefer a more specific type such as Soundtrack or DJ Mix.
  Collaborations, a single artist's covers, and new classical recordings are not automatically compilations.
- DJ Mix: the release is mixed for continuous listening, even if split into tracks; an album by a DJ is not enough.
- Soundtrack: music for film, TV, games or stage productions.
- Live album: a produced, marketed official live album. Concert Recording: an official individual-show recording,
  including instant concert downloads. Bootleg: an unofficial performance release, not just missing online evidence.
- Remix: primarily reworked material; distinguish a remix collection from a single with alternate mixes.
- Demo: explicitly identified preliminary recordings. Interview: an interview release.
- Mixtape: a deliberately sequenced mixtape; verify the designation rather than inferring it from the genre.
- Unknown: no defensible classification. If uncertain, retain the existing type and explain the uncertainty.
Classification does not establish upload eligibility; flag evidence of prohibited service-only compilations
for manual checking rather than declaring them allowed.
"""


def resolve_release_type(before, value):
    """Keep the existing type if a provider ignores the enum or omits its answer."""
    return value if isinstance(value, str) and value in RELEASE_TYPES else before


def guard_release_type(metadata, review):
    proposed = review.get("metadata")
    if not isinstance(proposed, dict) or "rls_type" not in proposed:
        return None
    value = proposed["rls_type"]
    if isinstance(value, str) and value in RELEASE_TYPES:
        return None
    proposed["rls_type"] = metadata.get("rls_type")
    return f"Ignored unsupported AI release type {value!r}; kept the current type."
