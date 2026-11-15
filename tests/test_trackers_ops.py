import sys
from pathlib import Path

import anyio
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from salmon.common import UploadFiles
from salmon.errors import RequestError
from salmon.trackers.base import BaseGazelleApi, HttpResponse
from salmon.trackers.ops import OpsApi


@pytest.mark.parametrize("release_type", ["Album", "EP", "Single", "Compilation", "DJ Mix"])
@pytest.mark.parametrize("group_id", [None, 5])
def test_ops_keeps_the_selected_type_without_a_split_question(monkeypatch, release_type, group_id):
    tracker = OpsApi()
    sent = []

    async def unexpected_question(*args, **kwargs):
        pytest.fail("OPS must use the normal release type without a Split question")

    async def fake_upload(self, data, files):
        sent.append(data.copy())
        return 123, 456

    monkeypatch.setattr("salmon.interaction.confirm", unexpected_question)
    monkeypatch.setattr(BaseGazelleApi, "upload", fake_upload)
    data = {
        "releasetype": tracker.release_types[release_type],
        "artists[]": ["A", "B"],
        "importance[]": [1, 1],
    }
    if group_id is not None:
        data["groupid"] = group_id

    result = anyio.run(tracker.upload, data, UploadFiles(torrent_data=b"torrent"))

    assert result == (123, 456)
    assert sent == [data]
    assert data["releasetype"] == tracker.release_types[release_type]


def test_a_failed_lossy_master_report_shows_none_of_its_credentials(monkeypatch) -> None:
    tracker = OpsApi()
    tracker.authkey = "ops-authkey-0123456789"

    async def redirected(*_args, **_kwargs) -> HttpResponse:
        url = f"https://orpheus.network/reportsv2.php?id=5&key={tracker.authkey}"
        return HttpResponse(text="", url=url, status=200)

    monkeypatch.setattr(tracker, "_request", redirected)

    with pytest.raises(RequestError, match="reportsv2.php") as caught:
        anyio.run(tracker.report_lossy_master, 555, "a comment", "CD")
    assert tracker.authkey not in str(caught.value)
