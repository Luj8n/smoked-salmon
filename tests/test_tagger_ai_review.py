import ssl
import subprocess
import sys
import textwrap
from copy import deepcopy
from pathlib import Path

import anyio
import openai
import pytest
from aiohttp import web
from openai.types.responses import Response, ResponseFunctionWebSearch, ResponseReasoningItem
from openai.types.responses.response_function_web_search import ActionFind, ActionOpenPage, ActionSearch
from openai.types.responses.response_reasoning_item import Summary

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helpers.tagger_ai_review import make_metadata, make_review

from salmon import cfg
from salmon.errors import InvalidMetadataError
from salmon.tagger import ai_review
from salmon.tagger.ai_review import (
    _ai_review_schema,
    _apply_ai_review_guardrails,
    _build_album_metadata_snapshot,
    _build_release_reference,
    _build_request_payload,
    _choose_ai_anchor_url,
    _describe_web_search_action,
    _emit_ai_status_heartbeat,
    _extract_opened_page_urls,
    _extract_progress_updates,
    _format_ai_progress,
    _page_explicitly_names_label,
    _style_ai_progress,
    apply_ai_metadata_result,
    build_ai_review_diff,
    format_ai_review_citations,
)

_RESPONSE_DEFAULTS: dict = {
    "id": "resp_test",
    "created_at": 0.0,
    "model": "test",
    "object": "response",
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}


def make_response(output: list) -> Response:
    return Response(**{**_RESPONSE_DEFAULTS, "output": output})


async def _names_label(_url: str, _label: str) -> bool:
    return True


def test_apply_ai_metadata_result_replaces_album_level_fields() -> None:
    metadata = make_metadata()
    review = make_review(
        label="New Label",
        year="2003",
        group_year="2003",
        genres=["Electronic", "Deep House", "Electronic"],
        urls=["https://official.example/release", "https://official.example/release"],
    )

    updated = apply_ai_metadata_result(metadata, review)

    assert updated["label"] == "New Label"
    assert updated["year"] == "2003"
    assert updated["group_year"] == "2003"
    assert updated["genres"] == ["Electronic", "Deep House"]
    assert updated["urls"] == ["https://official.example/release"]
    assert updated["tracks"]["1"]["2"]["title"] == "Track Two"
    assert metadata["label"] == "Old Label"


def test_apply_ai_metadata_result_can_replace_release_level_artists_only() -> None:
    metadata = make_metadata()
    original_track_artists = deepcopy(metadata["tracks"]["1"]["1"]["artists"])
    review = make_review(
        artists=[
            {"name": "Riddim Research Lab", "role": "main"},
            {"name": "Ant To Be", "role": "guest"},
        ]
    )

    updated = apply_ai_metadata_result(metadata, review)

    assert updated["artists"] == [
        ("Riddim Research Lab", "main"),
        ("Ant To Be", "guest"),
    ]
    assert updated["tracks"]["1"]["1"]["artists"] == original_track_artists
    assert updated["tracks"]["1"]["2"]["artists"] == metadata["tracks"]["1"]["2"]["artists"]


def test_apply_ai_metadata_result_can_clear_fields() -> None:
    metadata = make_metadata()
    review = make_review(label=None, catno=None, genres=[], urls=[])

    updated = apply_ai_metadata_result(metadata, review)

    assert updated["label"] is None
    assert updated["catno"] is None
    assert updated["genres"] == []
    assert updated["urls"] == []


def test_build_ai_review_diff_reports_only_actual_changes() -> None:
    metadata = make_metadata()
    review = make_review(
        label="New Label",
        year="2003",
        genres=["Electronic", "Deep House"],
        urls=metadata["urls"],
    )

    diff_lines = build_ai_review_diff(metadata, review)

    assert "label: Old Label -> New Label" in diff_lines
    assert "year: 2004 -> 2003" in diff_lines
    assert "genres: + Deep House" in diff_lines
    assert not any(line.startswith("urls:") for line in diff_lines)
    assert not any("track" in line for line in diff_lines)


def test_build_ai_review_diff_formats_release_level_artist_changes() -> None:
    metadata = make_metadata()
    review = make_review(
        artists=[
            {"name": "Riddim Research Lab", "role": "main"},
            {"name": "Ant To Be", "role": "guest"},
        ]
    )

    diff_lines = build_ai_review_diff(metadata, review)

    assert "artists: Example Artist [main] -> Riddim Research Lab [main], Ant To Be [guest]" in diff_lines


def test_ai_review_schema_requires_every_metadata_key() -> None:
    metadata_schema = _ai_review_schema()["properties"]["metadata"]
    assert metadata_schema["required"] == [
        "artists",
        "title",
        "rls_type",
        "group_year",
        "year",
        "edition_title",
        "label",
        "catno",
        "upc",
        "genres",
        "urls",
    ]


def test_build_request_payload_requests_reasoning_summary_and_does_not_store() -> None:
    payload = _build_request_payload(make_metadata(), make_metadata(), None)

    assert payload["reasoning"]["summary"] == "auto"
    assert payload["store"] is False


def test_choose_ai_anchor_url_prefers_explicit_source_then_first_metadata_url() -> None:
    metadata = make_metadata()

    assert _choose_ai_anchor_url(metadata, "https://explicit.example/release") == "https://explicit.example/release"
    assert _choose_ai_anchor_url(metadata, None) == "https://old.example/release"


def test_build_release_reference_keeps_only_identifying_fields() -> None:
    reference = _build_release_reference(make_metadata(), "https://example.com/release")

    assert reference == {
        "artists": [{"name": "Example Artist", "role": "main"}],
        "release_title_hint": "Original Title",
        "release_type_hint": "Album",
        "source": "WEB",
        "format": "FLAC",
        "encoding": "Lossless",
        "selected_source_url": "https://example.com/release",
    }


def test_build_album_metadata_snapshot_keeps_only_patchable_album_fields() -> None:
    snapshot = _build_album_metadata_snapshot(make_metadata())

    assert snapshot == {
        "artists": [{"name": "Example Artist", "role": "main"}],
        "title": "Original Title",
        "rls_type": "Album",
        "group_year": "2004",
        "year": "2004",
        "edition_title": None,
        "label": "Old Label",
        "catno": "OLD-001",
        "upc": None,
        "genres": ["Electronic"],
        "urls": ["https://old.example/release"],
    }


def test_build_request_payload_includes_local_album_metadata_context_but_not_tracks() -> None:
    metadata = make_metadata()
    tag_baseline = make_metadata()
    tag_baseline["title"] = "Tag Title"
    tag_baseline["label"] = None
    tag_baseline["urls"] = ["https://tag.example/release"]

    payload = _build_request_payload(metadata, tag_baseline, None)
    prompt = payload["input"]

    assert "Release reference JSON:" in prompt
    assert "Current editable album metadata JSON:" in prompt
    assert "Tag-derived album metadata baseline JSON:" in prompt
    assert '"release_title_hint": "Original Title"' in prompt
    assert '"artists": [' in prompt
    assert '"name": "Example Artist"' in prompt
    assert '"label": "Old Label"' in prompt
    assert '"title": "Tag Title"' in prompt
    assert "Track One" not in prompt


def test_build_request_payload_surfaces_selected_source_url_as_first_step() -> None:
    payload = _build_request_payload(
        make_metadata(),
        make_metadata(),
        "https://example.com/release",
    )
    prompt = payload["input"]

    assert "Selected source URL:" in prompt
    assert "https://example.com/release" in prompt
    assert "Open that exact page first before doing any search." in prompt


def test_disabled_web_review_does_not_request_online_research(monkeypatch):
    monkeypatch.setattr(cfg.upload.ai_review, "use_web_search", False)
    payload = _build_request_payload(make_metadata(), make_metadata(), "https://example.com/release")

    assert "tools" not in payload
    assert "Web research is unavailable." in payload["input"]
    assert "Open that exact page first" not in payload["input"]
    assert "https://example.com/release" in payload["input"]


def test_format_ai_progress_compacts_without_clamping_text() -> None:
    formatted = _format_ai_progress("line one\nline two   " + ("x" * 220))

    assert "\n" not in formatted
    assert "line one line two" in formatted
    assert formatted.endswith("x" * 220)


def test_style_ai_progress_renders_markdown_bold_without_literal_asterisks() -> None:
    styled = _style_ai_progress("reasoning: **Verifying release-level evidence** right now")

    assert "**" not in styled
    assert "Verifying release-level evidence" in styled
    assert "\x1b[" in styled


def test_extract_progress_updates_reports_reasoning_and_web_search() -> None:
    response = make_response(
        [
            ResponseReasoningItem(
                id="r_1",
                type="reasoning",
                summary=[
                    Summary(type="summary_text", text="Comparing Bandcamp and MusicBrainz."),
                    Summary(type="summary_text", text="Keeping the anchor page as primary evidence."),
                ],
            ),
            ResponseFunctionWebSearch(
                id="ws_123",
                type="web_search_call",
                status="searching",
                action=ActionSearch(type="search", query="Mouse and Banjo Dawn Dust label"),
            ),
            ResponseFunctionWebSearch(
                id="ws_456",
                type="web_search_call",
                status="completed",
                action=ActionSearch(type="search", query='"Mouse and Banjo" "Dawn//Dust"'),
            ),
            ResponseFunctionWebSearch(
                id="ws_789",
                type="web_search_call",
                status="completed",
                action=ActionFind(type="find_in_page", pattern="", url="https://example.com/release"),
            ),
            ResponseFunctionWebSearch(
                id="ws_999",
                type="web_search_call",
                status="completed",
                action=ActionOpenPage(type="open_page"),
            ),
        ]
    )

    lines, last_summary = _extract_progress_updates(response, set(), None)

    assert any(line == "reasoning: Keeping the anchor page as primary evidence." for line in lines)
    assert any(line == 'web_search: completed | search | "Mouse and Banjo" "Dawn//Dust"' for line in lines)
    assert not any("web_search: searching" in line for line in lines)
    assert not any("find in page" in line for line in lines)
    assert not any(line.endswith("open page") for line in lines)
    assert last_summary == "Keeping the anchor page as primary evidence."


def test_emit_ai_status_heartbeat_reports_initial_and_periodic_waits(monkeypatch) -> None:
    emitted: list[str] = []
    monotonic_values = iter([100.0, 104.0, 116.0, 117.0])

    monkeypatch.setattr(ai_review.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(ai_review, "_emit_ai_progress", emitted.append)

    last_status, last_emit_at = _emit_ai_status_heartbeat(
        "queued",
        started_at=100.0,
        last_status=None,
        last_emit_at=None,
    )
    last_status, last_emit_at = _emit_ai_status_heartbeat(
        "queued",
        started_at=100.0,
        last_status=last_status,
        last_emit_at=last_emit_at,
    )
    last_status, last_emit_at = _emit_ai_status_heartbeat(
        "queued",
        started_at=100.0,
        last_status=last_status,
        last_emit_at=last_emit_at,
    )
    last_status, last_emit_at = _emit_ai_status_heartbeat(
        "in_progress",
        started_at=100.0,
        last_status=last_status,
        last_emit_at=last_emit_at,
    )

    assert emitted == [
        "status: queued (0s elapsed)",
        "status: queued (16s elapsed)",
        "status: in_progress (17s elapsed)",
    ]
    assert last_status == "in_progress"
    assert last_emit_at == 117.0


def test_extract_opened_page_urls_keeps_only_completed_open_page_actions() -> None:
    response = make_response(
        [
            ResponseFunctionWebSearch(
                id="ws_1",
                type="web_search_call",
                status="completed",
                action=ActionOpenPage(type="open_page", url="https://example.com/release?utm=1"),
            ),
            ResponseFunctionWebSearch(
                id="ws_2",
                type="web_search_call",
                status="searching",
                action=ActionOpenPage(type="open_page", url="https://example.com/ignored"),
            ),
            ResponseFunctionWebSearch(
                id="ws_3",
                type="web_search_call",
                status="completed",
                action=ActionSearch(type="search", query="ignored"),
            ),
        ]
    )

    assert _extract_opened_page_urls(response) == {"https://example.com/release"}


def test_page_explicitly_names_label_rejects_bare_rights_lines() -> None:
    assert _page_explicitly_names_label(
        "Label: Crash Blossoms Money World Suicide Mob",
        "Crash Blossoms Money World Suicide Mob",
    )
    assert not _page_explicitly_names_label(
        "℗ 2024 Crash Blossoms Money World Suicide Mob",
        "Crash Blossoms Money World Suicide Mob",
    )


def test_apply_ai_review_guardrails_ignores_unsupported_label_changes() -> None:
    metadata = make_metadata()
    metadata["label"] = "doin' fine"
    review = make_review(label="Crash Blossoms Money World Suicide Mob")

    sanitized_review, warnings = anyio.run(_apply_ai_review_guardrails, metadata, review, None)

    assert sanitized_review["metadata"]["label"] == "doin' fine"
    assert warnings == [
        'Ignored AI label change to "Crash Blossoms Money World Suicide Mob" because no opened citation '
        "explicitly supported the label field."
    ]


def test_apply_ai_review_guardrails_rejects_self_released_without_opened_label_citation() -> None:
    metadata = make_metadata()
    metadata["label"] = "doin' fine"
    review = make_review(label="Self-Released")

    sanitized_review, warnings = anyio.run(_apply_ai_review_guardrails, metadata, review, None)

    assert sanitized_review["metadata"]["label"] == "doin' fine"
    assert len(warnings) == 1
    assert "no opened citation" in warnings[0]


@pytest.mark.parametrize("supported", [True, False])
def test_self_released_requires_explicit_page_evidence(monkeypatch, supported):
    metadata = make_metadata()
    review = make_review(label="Self-Released")
    review["citations"] = [{"title": "Release", "url": "https://example.com/release", "supports": ["label"]}]
    review["_opened_page_urls"] = ["https://example.com/release"]

    async def page_text(url):
        return "Label: Self-Released" if supported else "Artist: Example Artist. Copyright Example Artist."

    monkeypatch.setattr(ai_review, "_fetch_release_page_text", page_text)
    sanitized, warnings = anyio.run(_apply_ai_review_guardrails, metadata, review, None)

    assert sanitized["metadata"]["label"] == ("Self-Released" if supported else metadata["label"])
    assert bool(warnings) is not supported


def test_apply_ai_review_guardrails_accepts_explicit_opened_label_evidence(monkeypatch) -> None:
    metadata = make_metadata()
    review = make_review(label="New Label")
    review["citations"] = [
        {
            "title": "Official release page",
            "url": "https://example.com/release",
            "supports": ["label"],
        }
    ]
    review["_opened_page_urls"] = ["https://example.com/release"]
    monkeypatch.setattr(ai_review, "_url_explicitly_names_label", _names_label)

    sanitized_review, warnings = anyio.run(_apply_ai_review_guardrails, metadata, review, None)

    assert sanitized_review["metadata"]["label"] == "New Label"
    assert warnings == []


def test_apply_ai_review_guardrails_accepts_annotated_label_support(monkeypatch) -> None:
    metadata = make_metadata()
    review = make_review(label="ITModels")
    review["citations"] = [
        {
            "title": "Qobuz release page",
            "url": "https://example.com/release",
            "supports": ["label (compound source credit)", "title", "urls"],
        }
    ]
    review["_opened_page_urls"] = ["https://example.com/release"]
    monkeypatch.setattr(ai_review, "_url_explicitly_names_label", _names_label)

    sanitized_review, warnings = anyio.run(_apply_ai_review_guardrails, metadata, review, None)

    assert sanitized_review["metadata"]["label"] == "ITModels"
    assert warnings == []


def test_format_ai_review_citations_marks_opened_status() -> None:
    review = make_review()
    review["citations"] = [
        {
            "title": "Opened source",
            "url": "https://example.com/release",
            "supports": ["title"],
        },
        {
            "title": "Not opened source",
            "url": "https://example.com/other",
            "supports": ["label"],
        },
    ]
    review["_opened_page_urls"] = ["https://example.com/release?utm=1"]

    assert format_ai_review_citations(review) == [
        "- Opened source (title) [opened]: https://example.com/release",
        "- Not opened source (label) [not opened]: https://example.com/other",
    ]


def test_apply_ai_review_guardrails_preserves_existing_guest_artists() -> None:
    metadata = make_metadata()
    metadata["artists"] = [
        ("doin' fine + ilysm", "main"),
        ("MSPAINTBOI", "guest"),
        ("STARRI", "guest"),
        ("SOR UH", "guest"),
    ]
    review = make_review(
        artists=[
            {"name": "doin' fine", "role": "main"},
            {"name": "ilysm", "role": "main"},
        ]
    )

    sanitized_review, warnings = anyio.run(_apply_ai_review_guardrails, metadata, review, None)

    assert sanitized_review["metadata"]["artists"] == [
        {"name": "doin' fine", "role": "main"},
        {"name": "ilysm", "role": "main"},
        {"name": "MSPAINTBOI", "role": "guest"},
        {"name": "STARRI", "role": "guest"},
        {"name": "SOR UH", "role": "guest"},
    ]
    assert warnings == ["Preserved existing guest artists that the AI tried to remove: MSPAINTBOI, STARRI, SOR UH."]


@pytest.mark.parametrize(("opened", "field"), [(True, "artists"), (False, "artists"), (True, "title")])
def test_guest_correction_requires_an_opened_artist_citation(opened, field):
    metadata = make_metadata()
    metadata["artists"].append(("Wrong Guest", "guest"))
    review = make_review()
    review["summary"] = "Removed Wrong Guest: the edition credits identify a different performer."
    review["citations"] = [{"title": "Release", "url": "https://example.com/release", "supports": [field]}]
    review["_opened_page_urls"] = ["https://example.com/release"] if opened else []

    sanitized, warnings = anyio.run(_apply_ai_review_guardrails, metadata, review, None)

    names = {artist["name"] for artist in sanitized["metadata"]["artists"]}
    allowed = opened and field == "artists"
    assert ("Wrong Guest" not in names) is allowed
    assert bool(warnings) is not allowed


def test_apply_ai_review_guardrails_allows_guest_to_main_promotion() -> None:
    metadata = make_metadata()
    metadata["artists"] = [
        ("Anna Zak", "main"),
        ("Itay Galo", "guest"),
    ]
    metadata["tracks"]["1"]["1"]["artists"] = [
        ("Anna Zak", "main"),
        ("Itay Galo", "guest"),
    ]
    metadata["tracks"]["1"]["2"]["artists"] = [
        ("Anna Zak", "main"),
        ("Itay Galo", "guest"),
    ]
    review = make_review(
        artists=[
            {"name": "Anna Zak", "role": "main"},
            {"name": "Itay Galo", "role": "main"},
        ]
    )

    sanitized_review, warnings = anyio.run(_apply_ai_review_guardrails, metadata, review, None)

    assert sanitized_review["metadata"]["artists"] == [
        {"name": "Anna Zak", "role": "main"},
        {"name": "Itay Galo", "role": "main"},
    ]
    assert warnings == []


def test_apply_ai_review_guardrails_drops_unopened_url_additions() -> None:
    metadata = make_metadata()
    review = make_review(
        urls=[
            "https://old.example/release",
            "https://music.apple.com/us/album/example/123",
            "https://music.amazon.in/albums/example",
        ]
    )
    review["_opened_page_urls"] = ["https://old.example/release"]

    sanitized_review, warnings = anyio.run(_apply_ai_review_guardrails, metadata, review, None)

    assert sanitized_review["metadata"]["urls"] == ["https://old.example/release"]
    assert warnings == [
        "Ignored AI URL additions that were not opened during review: "
        "https://music.apple.com/us/album/example/123, https://music.amazon.in/albums/example."
    ]


def test_review_metadata_with_ai_runs_after_manual_review_and_reopens_after_apply(monkeypatch) -> None:
    metadata = make_metadata()
    review = make_review(year="2003")
    sequence: list[str] = []
    original_enabled = cfg.upload.ai_review.enabled

    async def fake_manual_review(current_metadata, _validator):
        sequence.append(f"manual:{current_metadata['label']}")
        return current_metadata

    async def fake_request_ai_review(*_args, **_kwargs):
        sequence.append("ai")
        return review, "response-1"

    async def fake_prompt(*_args, **_kwargs):
        return "a"

    def fake_confirm(*_args, **_kwargs):
        sequence.append("confirm")
        return True

    try:
        cfg.upload.ai_review.enabled = True
        monkeypatch.setattr(ai_review, "_request_ai_review", fake_request_ai_review)
        monkeypatch.setattr(ai_review.click, "prompt", fake_prompt)
        monkeypatch.setattr(ai_review.click, "confirm", fake_confirm)

        result = anyio.run(
            ai_review.review_metadata_with_ai,
            metadata,
            metadata,
            None,
            lambda current_metadata: current_metadata,
            fake_manual_review,
        )
    finally:
        cfg.upload.ai_review.enabled = original_enabled

    assert result["year"] == "2003"
    assert sequence == ["manual:Old Label", "confirm", "ai", "manual:Old Label"]


def test_a_failed_ai_review_after_a_skipped_review_still_validates(monkeypatch) -> None:
    # With the first review skipped, nothing has validated the metadata yet: a track left with no artist (#831) must
    # go back to the manual review, not on to the tagging.
    reviewed: list[dict] = []

    async def fake_manual_review(current_metadata, _validator):
        reviewed.append(current_metadata)
        return current_metadata

    async def failing_request(*_args, **_kwargs):
        raise RuntimeError("no connection")

    def refuse(_metadata):
        raise InvalidMetadataError("You must have at least one main artist per track.")

    monkeypatch.setattr(cfg.upload.ai_review, "enabled", True)
    monkeypatch.setattr(ai_review, "_request_ai_review", failing_request)

    anyio.run(
        lambda: ai_review.review_metadata_with_ai(
            make_metadata(),
            make_metadata(),
            None,
            refuse,
            fake_manual_review,
            skip_initial_review=True,
            apply_suggestions=True,
        )
    )

    assert len(reviewed) == 1


def test_review_metadata_with_ai_skips_ai_when_user_declines(monkeypatch) -> None:
    metadata = make_metadata()
    sequence: list[str] = []
    confirm_kwargs: dict[str, object] = {}
    original_enabled = cfg.upload.ai_review.enabled

    async def fake_manual_review(current_metadata, _validator):
        sequence.append("manual")
        return current_metadata

    def fake_confirm(*_args, **_kwargs):
        confirm_kwargs.update(_kwargs)
        sequence.append("confirm")
        return False

    try:
        cfg.upload.ai_review.enabled = True
        monkeypatch.setattr(ai_review.click, "confirm", fake_confirm)

        result = anyio.run(
            ai_review.review_metadata_with_ai,
            metadata,
            metadata,
            None,
            lambda current_metadata: current_metadata,
            fake_manual_review,
        )
    finally:
        cfg.upload.ai_review.enabled = original_enabled

    assert result["label"] == "Old Label"
    assert sequence == ["manual", "confirm"]
    assert confirm_kwargs["default"] is None


def test_review_metadata_with_ai_auto_keeps_when_no_suggestions(monkeypatch) -> None:
    metadata = make_metadata()
    review = make_review()
    sequence: list[str] = []
    original_enabled = cfg.upload.ai_review.enabled

    async def fake_manual_review(current_metadata, _validator):
        sequence.append("manual")
        return current_metadata

    async def fake_request_ai_review(*_args, **_kwargs):
        sequence.append("ai")
        return review, "response-1"

    def fake_confirm(*_args, **_kwargs):
        sequence.append("confirm")
        return True

    async def fake_prompt(*_args, **_kwargs):
        raise AssertionError("Prompt should not be shown when AI has no suggestions")

    try:
        cfg.upload.ai_review.enabled = True
        monkeypatch.setattr(ai_review, "_request_ai_review", fake_request_ai_review)
        monkeypatch.setattr(ai_review.click, "confirm", fake_confirm)
        monkeypatch.setattr(ai_review.click, "prompt", fake_prompt)

        result = anyio.run(
            ai_review.review_metadata_with_ai,
            metadata,
            metadata,
            None,
            lambda current_metadata: current_metadata,
            fake_manual_review,
        )
    finally:
        cfg.upload.ai_review.enabled = original_enabled

    assert result["label"] == "Old Label"
    assert sequence == ["manual", "confirm", "ai"]


def test_review_metadata_with_ai_uses_first_metadata_url_as_anchor_when_source_url_missing(monkeypatch) -> None:
    metadata = make_metadata()
    review = make_review()
    seen_source_urls: list[str | None] = []
    original_enabled = cfg.upload.ai_review.enabled

    async def fake_manual_review(current_metadata, _validator):
        return current_metadata

    async def fake_request_ai_review(current_metadata, _tag_baseline, source_url, _user_instruction, _previous_id):
        seen_source_urls.append(source_url)
        return review, "response-1"

    def fake_confirm(*_args, **_kwargs):
        return True

    try:
        cfg.upload.ai_review.enabled = True
        monkeypatch.setattr(ai_review, "_request_ai_review", fake_request_ai_review)
        monkeypatch.setattr(ai_review.click, "confirm", fake_confirm)

        result = anyio.run(
            ai_review.review_metadata_with_ai,
            metadata,
            metadata,
            None,
            lambda current_metadata: current_metadata,
            fake_manual_review,
        )
    finally:
        cfg.upload.ai_review.enabled = original_enabled

    assert result["label"] == "Old Label"
    assert seen_source_urls == ["https://old.example/release"]


def test_review_metadata_with_ai_auto_applies_when_flag_enabled(monkeypatch) -> None:
    metadata = make_metadata()
    review = make_review(year="2003")
    sequence: list[str] = []
    original_enabled = cfg.upload.ai_review.enabled
    original_yes_all = cfg.upload.yes_all

    async def fake_manual_review(current_metadata, _validator):
        sequence.append(f"manual:{current_metadata['label']}")
        return current_metadata

    async def fake_request_ai_review(*_args, **_kwargs):
        sequence.append("ai")
        return review, "response-1"

    async def fail_prompt(*_args, **_kwargs):
        raise AssertionError("click.prompt should not run when AI suggestions are auto-applied")

    def fail_confirm(*_args, **_kwargs):
        raise AssertionError("click.confirm should not run when AI suggestions auto-run and auto-apply")

    async def run_review():
        return await ai_review.review_metadata_with_ai(
            metadata,
            metadata,
            None,
            lambda current_metadata: current_metadata,
            fake_manual_review,
            apply_suggestions=True,
        )

    try:
        cfg.upload.ai_review.enabled = True
        cfg.upload.yes_all = False
        monkeypatch.setattr(ai_review, "_request_ai_review", fake_request_ai_review)
        monkeypatch.setattr(ai_review.click, "prompt", fail_prompt)
        monkeypatch.setattr(ai_review.click, "confirm", fail_confirm)

        result = anyio.run(run_review)
    finally:
        cfg.upload.ai_review.enabled = original_enabled
        cfg.upload.yes_all = original_yes_all

    assert result["year"] == "2003"
    assert sequence == ["manual:Old Label", "ai"]


def test_review_metadata_with_ai_skips_initial_review_when_flag_enabled(monkeypatch) -> None:
    metadata = make_metadata()
    review = make_review(year="2003")
    sequence: list[str] = []
    original_enabled = cfg.upload.ai_review.enabled

    async def fake_manual_review(current_metadata, _validator):
        sequence.append(f"manual:{current_metadata['label']}")
        return current_metadata

    async def fake_request_ai_review(*_args, **_kwargs):
        sequence.append("ai")
        return review, "response-1"

    async def fake_prompt(*_args, **_kwargs):
        return "a"

    def fake_confirm(*_args, **_kwargs):
        sequence.append("confirm")
        return True

    async def run_review():
        return await ai_review.review_metadata_with_ai(
            metadata,
            metadata,
            None,
            lambda current_metadata: current_metadata,
            fake_manual_review,
            skip_initial_review=True,
        )

    try:
        cfg.upload.ai_review.enabled = True
        monkeypatch.setattr(ai_review, "_request_ai_review", fake_request_ai_review)
        monkeypatch.setattr(ai_review.click, "prompt", fake_prompt)
        monkeypatch.setattr(ai_review.click, "confirm", fake_confirm)

        result = anyio.run(run_review)
    finally:
        cfg.upload.ai_review.enabled = original_enabled

    assert result["year"] == "2003"
    assert sequence == ["confirm", "ai", "manual:Old Label"]


def test_review_metadata_with_ai_yes_all_auto_applies(monkeypatch) -> None:
    metadata = make_metadata()
    review = make_review(year="2003")
    sequence: list[str] = []
    original_enabled = cfg.upload.ai_review.enabled
    original_yes_all = cfg.upload.yes_all

    async def fake_manual_review(current_metadata, _validator):
        sequence.append(f"manual:{current_metadata['label']}")
        return current_metadata

    async def fake_request_ai_review(*_args, **_kwargs):
        sequence.append("ai")
        return review, "response-1"

    async def fail_prompt(*_args, **_kwargs):
        raise AssertionError("click.prompt should not run when -yyy auto-applies AI suggestions")

    def fail_confirm(*_args, **_kwargs):
        raise AssertionError("click.confirm should not run when -yyy is enabled")

    try:
        cfg.upload.ai_review.enabled = True
        cfg.upload.yes_all = True
        monkeypatch.setattr(ai_review, "_request_ai_review", fake_request_ai_review)
        monkeypatch.setattr(ai_review.click, "prompt", fail_prompt)
        monkeypatch.setattr(ai_review.click, "confirm", fail_confirm)

        result = anyio.run(
            ai_review.review_metadata_with_ai,
            metadata,
            metadata,
            None,
            lambda current_metadata: current_metadata,
            fake_manual_review,
        )
    finally:
        cfg.upload.ai_review.enabled = original_enabled
        cfg.upload.yes_all = original_yes_all

    assert result["year"] == "2003"
    assert sequence == ["ai"]


def _web_search(action) -> ResponseFunctionWebSearch:
    return ResponseFunctionWebSearch(id="ws_1", type="web_search_call", status="completed", action=action)


def test_describe_web_search_action_tolerates_missing_query_and_url() -> None:
    # openai 3 types these fields as Optional; a search with no query or a page open with no URL
    # must describe as nothing instead of calling .strip() on None.
    assert _describe_web_search_action(_web_search(ActionSearch(type="search", query=None))) == ""
    assert _describe_web_search_action(_web_search(ActionSearch(type="search", query="  "))) == ""
    assert _describe_web_search_action(_web_search(ActionOpenPage(type="open_page", url=None))) == ""
    assert _describe_web_search_action(_web_search(ActionOpenPage(type="open_page", url="  "))) == ""


def test_describe_web_search_action_describes_known_actions() -> None:
    assert _describe_web_search_action(_web_search(ActionSearch(type="search", query=" a b "))) == "search | a b"
    assert (
        _describe_web_search_action(_web_search(ActionOpenPage(type="open_page", url=" https://example.com/x ")))
        == "open page | https://example.com/x"
    )
    find = ActionFind(type="find_in_page", pattern="p", url="https://example.com/x")
    assert _describe_web_search_action(_web_search(find)) == ""


def test_importing_ai_review_and_building_a_client_leaves_ssl_unpatched() -> None:
    # openai 3 pulls in httpx2 and truststore. truststore must only back the openai client's own
    # SSL context, never replace ssl.SSLContext process-wide, or aiohttp's tracker traffic changes.
    # Regression guard: it holds today; this fails if openai or httpx2 ever injects truststore globally.
    script = textwrap.dedent(
        """
        import ssl

        original = (ssl.SSLContext, ssl.create_default_context)

        import openai
        import salmon.tagger.ai_review  # noqa: F401

        openai.AsyncOpenAI(api_key="k", base_url="https://192.0.2.1:8443/v1")

        assert (ssl.SSLContext, ssl.create_default_context) == original, "ssl was patched"
        assert type(ssl.create_default_context()) is original[0]
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    # The test process itself must not be patched either.
    assert type(ssl.create_default_context()) is ssl.SSLContext


def test_client_with_custom_base_url_reaches_a_local_server_and_maps_404() -> None:
    # Regression guard: openai-compatible servers (local LLMs) keep working through base_url alone.
    seen: list[str] = []

    async def responses(request: web.Request) -> web.Response:
        seen.append(request.path)
        return web.json_response({"error": {"message": "not found"}}, status=404)

    async def run() -> None:
        app = web.Application()
        app.router.add_post("/v1/responses", responses)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        try:
            client = openai.AsyncOpenAI(api_key="k", base_url=f"http://127.0.0.1:{port}/v1", max_retries=0)
            try:
                await client.responses.create(model="m", input="hi")
            except openai.NotFoundError:
                pass
            else:
                raise AssertionError("expected NotFoundError")
            finally:
                await client.close()
        finally:
            await runner.cleanup()

    anyio.run(run)
    assert seen == ["/v1/responses"]


@pytest.mark.parametrize("field", ["genres", "urls"])
def test_list_diff_shows_only_added_and_removed_items(field) -> None:
    old, kept, new = (
        ("Jazz", "Electronic", "House")
        if field == "genres"
        else ("https://example.com/old", "https://example.com/kept", "https://example.com/new")
    )
    metadata = {**make_metadata(), field: [old, kept]}
    review = make_review(**{field: [kept, new]})
    lines = build_ai_review_diff(metadata, review)
    assert [line for line in lines if line.startswith(f"{field}:")] == [f"{field}: - {old}", f"{field}: + {new}"]


@pytest.mark.parametrize("field", ["genres", "urls"])
def test_list_diff_ignores_reordering_and_shows_explicit_clear(field) -> None:
    values = ["Jazz", "Electronic"] if field == "genres" else ["https://example.com/a", "https://example.com/b"]
    metadata = {**make_metadata(), field: values}
    reordered = build_ai_review_diff(metadata, make_review(**{field: list(reversed(values))}))
    assert not any(line.startswith(f"{field}:") for line in reordered)
    cleared = build_ai_review_diff(metadata, make_review(**{field: []}))
    assert [line for line in cleared if line.startswith(f"{field}:")] == [f"{field}: - {value}" for value in values]


def test_url_diff_respects_preserved_source_urls() -> None:
    old, source, added = "https://example.com/old", "https://example.com/source", "https://example.com/new"
    metadata = {**make_metadata(), "urls": [old, source]}
    lines = build_ai_review_diff(metadata, make_review(urls=[added]), source_url=source)
    assert [line for line in lines if line.startswith("urls:")] == [f"urls: + {added}"]


def test_qobuz_localized_url_does_not_duplicate_existing_source():
    source = "https://www.qobuz.com/album/-/rn74e65h2n71a"
    localized = "https://www.qobuz.com/us-en/album/planet-ladji-ladji-mouflet/rn74e65h2n71a"
    metadata = {**make_metadata(), "urls": [source]}
    review = make_review(urls=[source, localized])
    review["_opened_page_urls"] = [localized]
    review["summary"] = "Added a Qobuz page."
    sanitized, warnings = anyio.run(_apply_ai_review_guardrails, metadata, review, source)
    assert sanitized["metadata"]["urls"] == [source]
    # Citation provenance still records the exact page opened; this is display-URL deduplication only.
    assert sanitized["_opened_page_urls"] == [localized]
    assert not any(line.startswith("urls:") for line in build_ai_review_diff(metadata, sanitized, source))
    assert ai_review._review_summary(sanitized, [], warnings) == "No metadata changes remain after checks."


@pytest.mark.parametrize("opened", [True, False])
def test_different_qobuz_release_is_not_a_duplicate_and_still_needs_opening(opened):
    source = "https://www.qobuz.com/album/-/rn74e65h2n71a"
    different = "https://www.qobuz.com/us-en/album/other/anotherid"
    metadata = {**make_metadata(), "urls": [source]}
    review = make_review(urls=[source, different])
    review["_opened_page_urls"] = [different] if opened else []
    sanitized, warnings = anyio.run(_apply_ai_review_guardrails, metadata, review, source)
    assert sanitized["metadata"]["urls"] == ([source, different] if opened else [source])
    assert bool(warnings) is not opened


def test_unavailable_label_page_does_not_claim_evidence_is_absent(monkeypatch):
    metadata = {**make_metadata(), "label": "Self-Released"}
    review = make_review(label="Ladji Mouflet")
    review["summary"] = "Changed the label to Ladji Mouflet."
    review["citations"] = [{"url": "https://example.com/release", "supports": ["label: Ladji Mouflet"]}]
    review["_opened_page_urls"] = ["https://example.com/release"]

    async def unavailable(url):
        raise TimeoutError()

    monkeypatch.setattr(ai_review, "_fetch_release_page_text", unavailable)
    sanitized, warnings = anyio.run(_apply_ai_review_guardrails, metadata, review, None)
    assert sanitized["metadata"]["label"] == "Self-Released"
    assert "could not independently fetch" in warnings[0]
    assert "none of the opened cited pages" not in warnings[0]
    assert ai_review._review_summary(sanitized, [], warnings) == "No metadata changes remain after checks."


def test_filtered_summary_only_names_changes_still_offered():
    review = make_review(label="Ladji Mouflet")
    review["summary"] = "Changed label and genre."
    summary = ai_review._review_summary(review, ["genres: - Wereldmuziek", "genres: + World"], ["Label blocked"])
    assert summary == "Suggested changes after checks: genres."


def test_unfiltered_summary_keeps_the_models_reasoning():
    review = make_review()
    review["summary"] = "Retained the original release date."
    assert ai_review._review_summary(review, [], []) == review["summary"]
