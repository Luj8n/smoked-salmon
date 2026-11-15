from copy import deepcopy
from unittest.mock import AsyncMock

import anyio
import pytest
from helpers.tagger_ai_review import make_metadata, make_review

from salmon import cfg
from salmon.constants import RELEASE_TYPES
from salmon.tagger import ai_review, metadata_validator_base


def test_both_review_backends_receive_a_required_supported_release_type_enum():
    schema = ai_review._ai_review_schema()["properties"]["metadata"]
    assert "rls_type" in schema["required"]
    assert schema["properties"]["rls_type"] == {"type": "string", "enum": list(RELEASE_TYPES)}
    assert "Split" not in schema["properties"]["rls_type"]["enum"]
    payload = ai_review._build_request_payload(make_metadata(), make_metadata(), None)
    assert '"rls_type": "Album"' in payload["input"]


@pytest.mark.parametrize("release_type", list(RELEASE_TYPES))
def test_supported_release_types_are_shown_applied_and_validated(release_type):
    original = make_metadata()
    before = deepcopy(original)
    review = make_review(rls_type=release_type)

    updated = ai_review.apply_ai_metadata_result(original, review)
    diff = ai_review.build_ai_review_diff(original, review)

    assert updated["rls_type"] == release_type
    assert original == before
    assert updated["tracks"] == original["tracks"]
    assert (f"rls_type: Album -> {release_type}" in diff) is (release_type != "Album")
    metadata_validator_base(updated)


@pytest.mark.parametrize("value", [None, "", "Extended Play", "Split", 5, ["EP"]])
def test_invalid_or_empty_model_release_types_preserve_current_metadata(value):
    metadata = make_metadata()
    review = make_review(rls_type=value)

    sanitized, warnings = anyio.run(ai_review._apply_ai_review_guardrails, metadata, review, None)

    assert sanitized["metadata"]["rls_type"] == "Album"
    assert any("unsupported AI release type" in warning for warning in warnings)
    assert ai_review.apply_ai_metadata_result(metadata, review)["rls_type"] == "Album"
    assert not any(line.startswith("rls_type:") for line in ai_review.build_ai_review_diff(metadata, review))


def test_older_reviews_without_a_release_type_preserve_the_existing_value():
    review = make_review()
    del review["metadata"]["rls_type"]
    assert ai_review.apply_ai_metadata_result(make_metadata(), review)["rls_type"] == "Album"


@pytest.mark.parametrize("automatic", [True, False])
def test_review_flow_applies_the_type_with_automatic_or_manual_review(monkeypatch, automatic):
    seen = []

    async def request(*args):
        return make_review(rls_type="DJ Mix"), None

    async def manual(metadata, validator):
        seen.append(metadata["rls_type"])
        validator(metadata)
        return metadata

    monkeypatch.setattr(cfg.upload.ai_review, "enabled", True)
    monkeypatch.setattr(ai_review, "_request_ai_review", request)
    monkeypatch.setattr(ai_review.interaction, "confirm", AsyncMock(return_value=True))
    monkeypatch.setattr(ai_review.interaction, "prompt", AsyncMock(return_value="a"))

    async def run():
        return await ai_review.review_metadata_with_ai(
            make_metadata(),
            make_metadata(),
            None,
            metadata_validator_base,
            manual,
            skip_initial_review=True,
            apply_suggestions=automatic,
        )

    result = anyio.run(run)

    assert result["rls_type"] == "DJ Mix"
    assert seen == ([] if automatic else ["DJ Mix"])
