from types import SimpleNamespace

import msgspec
import pytest

from salmon import cfg
from salmon.tagger.retagger import generate_file_name


@pytest.mark.parametrize("as_dict", [True, False])
def test_single_artist_filename_uses_the_keys_from_its_own_template(monkeypatch, as_dict):
    formatting = msgspec.structs.replace(
        cfg.upload.formatting,
        file_template="{artist} - {title}",
        one_album_artist_file_template="{tracknumber}. {title}",
        no_artist_in_filename_if_only_one_album_artist=True,
    )
    monkeypatch.setattr(cfg.upload, "formatting", formatting)
    tags = (
        {"artist": ["Artist"], "tracknumber": ["1"], "title": ["Song"]}
        if as_dict
        else SimpleNamespace(artist=["Artist"], tracknumber="1", title="Song")
    )

    assert generate_file_name(tags, ".flac", False) == "01. Song.flac"
