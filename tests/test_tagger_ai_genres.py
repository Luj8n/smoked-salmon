import anyio
import pytest
from helpers.tagger_ai_review import make_metadata, make_review

from salmon.tagger.ai_review import _apply_ai_review_guardrails, apply_ai_metadata_result


def check(before, proposed, *, supports=(), opened=True, **metadata_fields):
    metadata = {**make_metadata(), "genres": before, **metadata_fields}
    review = make_review(genres=proposed)
    review["citations"] = [{"title": "Release page", "url": "https://example.com/release", "supports": list(supports)}]
    review["_opened_page_urls"] = ["https://example.com/release"] if opened else []
    sanitized, warnings = anyio.run(_apply_ai_review_guardrails, metadata, review, None)
    return apply_ai_metadata_result(metadata, sanitized)["genres"], warnings


def test_only_genres_named_in_opened_citations_are_added():
    genres, warnings = check(["Jazz"], ["Jazz", "House", "Downtempo"], supports=["genres: House"])
    assert genres == ["Jazz", "House"]
    assert any("Downtempo" in warning for warning in warnings)


@pytest.mark.parametrize(
    ("supports", "opened"), [(["genres: House"], False), (["genres"], True), (["label: House"], True)]
)
def test_unsourced_changes_keep_existing_genres(supports, opened):
    genres, warnings = check(["Jazz"], ["House"], supports=supports, opened=opened)
    assert genres == ["Jazz"]
    assert warnings


@pytest.mark.parametrize("proposed", [[], None, [" "], "Jazz", {"genre": "Jazz"}])
def test_empty_or_malformed_answers_cannot_erase_the_existing_genres(proposed):
    genres, _ = check(["Jazz"], proposed, supports=["genres: House"])
    assert genres == ["Jazz"]


def test_evidenced_replacement_can_correct_a_wrong_tag():
    genres, warnings = check(["Jazz"], ["Downtempo", "House"], supports=["genres: Downtempo", "genres: House"])
    assert genres == ["Downtempo", "House"]
    assert warnings == []


def test_formatting_aliases_and_duplicates_do_not_require_new_research():
    genres, warnings = check(["deep.house", "dnb", "uk.bass"], ["Deep House", "Drum & Bass", "UK Bass", "uk.bass"])
    assert genres == ["Deep House", "Drum & Bass", "UK Bass"]
    assert warnings == []


def test_a_broad_genre_citation_does_not_support_an_invented_subgenre():
    genres, warnings = check(["Jazz"], ["Jazz", "Deep House"], supports=["genres: House"])
    assert genres == ["Jazz"]
    assert any("Deep House" in warning for warning in warnings)


def test_unfamiliar_genres_are_allowed_when_cited():
    genres, warnings = check(
        ["Electronic"], ["UK Bass", "Deconstructed Club"], supports=["genres: UK Bass, Deconstructed Club"]
    )
    assert genres == ["UK Bass", "Deconstructed Club"]
    assert warnings == []


def test_a_likely_typo_is_flagged_without_guessing_a_replacement():
    genres, warnings = check(["Downtemop"], ["Downtemop"])
    assert genres == ["Downtemop"]
    assert any("Downtempo" in warning and "spelling" in warning for warning in warnings)


def test_the_ai_can_correct_a_typo_with_genre_evidence():
    genres, warnings = check(["Downtemop"], ["Downtempo"], supports=["genres: Downtempo"])
    assert genres == ["Downtempo"]
    assert warnings == []


@pytest.mark.parametrize("tag", ["FLAC", "Qobuz", "Example Artist", "Old Label"])
def test_formats_stores_and_release_entities_are_not_added_as_genres(tag):
    genres, warnings = check(["Jazz"], ["Jazz", tag], supports=[f"genres: {tag}"])
    assert genres == ["Jazz"]
    assert any("non-genre" in warning for warning in warnings)


def test_translating_a_store_category_does_not_need_new_genre_evidence():
    genres, warnings = check(["Alternative en Indie"], ["Alternative"])
    assert genres == ["Alternative"]
    assert warnings == []


def test_dutch_world_translation_does_not_need_an_opened_citation():
    genres, warnings = check(["Wereldmuziek"], ["World"], supports=["genres: World"], opened=False)
    assert genres == ["World Music"]
    assert warnings == []


@pytest.mark.parametrize("support", ["genres: Shoegaze — tagged shoegazer", "genres: shoegazer"])
def test_shoegaze_citation_accepts_artist_tag_alias_and_explanation(support):
    genres, warnings = check(["Dream Pop"], ["Dream Pop", "Shoegaze"], supports=[support])
    assert genres == ["Dream Pop", "Shoegaze"]
    assert warnings == []


def test_genre_mentioned_only_in_citation_explanation_is_not_evidence():
    genres, warnings = check(
        ["Dream Pop"], ["Dream Pop", "Shoegaze"], supports=["genres: Dream Pop — not tagged Shoegaze"]
    )
    assert genres == ["Dream Pop"]
    assert any("Ignored AI genre Shoegaze" in warning for warning in warnings)


def test_annotated_genre_citation_still_requires_an_opened_page():
    genres, warnings = check(
        ["Dream Pop"], ["Dream Pop", "Shoegaze"], supports=["genres: Shoegaze — tagged shoegazer"], opened=False
    )
    assert genres == ["Dream Pop"]
    assert any("Ignored AI genre Shoegaze" in warning for warning in warnings)
