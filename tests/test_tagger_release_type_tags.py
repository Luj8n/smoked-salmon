"""An accepted release-type edit must reach the files, not just the tracker metadata."""

import pytest
from helpers.tagger_ai_review import make_review
from helpers.tagger_multi_value_tags import _flac_album, _metadata, _mp3_album, _retag, _trackmeta
from mutagen.flac import FLAC
from mutagen.id3 import TXXX
from mutagen.mp3 import MP3

from salmon.tagger.ai_review import apply_ai_metadata_result
from salmon.tagger.retagger import collect_album_data, print_changes
from salmon.tagger.tagfile import TagFile


@pytest.mark.parametrize("kind", ["flac", "mp3"])
def test_accepted_ep_correction_replaces_single_in_preview_and_saved_tags(tmp_path, capsys, kind):
    path = (
        _flac_album(tmp_path, RELEASETYPE="Single")
        if kind == "flac"
        else _mp3_album(tmp_path, TXXX(encoding=3, desc="releasetype", text="Single"))
    )
    before = _metadata(_trackmeta([("Sandhouse", "main")]))
    before["rls_type"] = "Single"
    updated = apply_ai_metadata_result(before, make_review(rls_type="EP"))
    print_changes(collect_album_data(updated), {}, TagFile(str(path)))
    output = capsys.readouterr().out
    assert "releasetype" in output
    assert "Single >>> EP" in output

    _retag(tmp_path, updated)

    assert TagFile(str(path)).releasetype == "EP"
    if kind == "flac":
        assert FLAC(path)["releasetype"] == ["EP"]
    else:
        tags = MP3(path).tags
        assert tags is not None
        frames = [frame for frame in tags.getall("TXXX") if frame.desc.lower() == "releasetype"]
        assert len(frames) == 1
        assert frames[0].text == ["EP"]


@pytest.mark.parametrize("release_type", [None, ""])
def test_unknown_release_type_does_not_erase_existing_tag(tmp_path, release_type):
    path = _flac_album(tmp_path, RELEASETYPE="EP")
    metadata = _metadata(_trackmeta([("Sandhouse", "main")]))
    metadata["rls_type"] = release_type
    _retag(tmp_path, metadata)
    assert FLAC(path)["releasetype"] == ["EP"]
