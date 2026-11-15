"""Exercise the Codex subprocess boundary with a local fake CLI, never an authenticated model or tracker."""

import json
import os
import sys
import textwrap
from pathlib import Path

import anyio
import msgspec
import pytest
from helpers.tagger_ai_review import make_metadata, make_review

from salmon import cfg
from salmon.config.validations import UploadAiReview
from salmon.tagger import ai_codex, ai_review

_FAKE_CODEX = """\
import json
import os
import subprocess
import sys
import time
from pathlib import Path

state = Path(__STATE__)
args = sys.argv[1:]
schema = Path(args[args.index("--output-schema") + 1])
output = Path(args[args.index("--output-last-message") + 1])
prompt = sys.stdin.read()
(state / "request.json").write_text(json.dumps({
    "args": args, "prompt": prompt, "schema": json.loads(schema.read_text()),
    "cwd": os.getcwd(), "env": dict(os.environ), "pid": os.getpid(),
}))
mode = (state / "mode").read_text()
if mode == "hang":
    time.sleep(600)
if mode == "launcher":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    (state / "child.pid").write_text(str(child.pid))
    child.wait()
if mode == "failed":
    print("secret diagnostic must not be displayed", file=sys.stderr)
    sys.exit(1)
if mode != "missing":
    output.write_text((state / "response.json").read_text())
print((state / "events.jsonl").read_text())
"""


@pytest.fixture
def cli(tmp_path, monkeypatch):
    script = tmp_path / "codex"
    script.write_text(f"#!{sys.executable}\n" + textwrap.dedent(_FAKE_CODEX).replace("__STATE__", repr(str(tmp_path))))
    script.chmod(0o700)
    (tmp_path / "mode").write_text("success")
    (tmp_path / "response.json").write_text(json.dumps(make_review()))
    (tmp_path / "events.jsonl").write_text("")
    monkeypatch.setattr(ai_codex.shutil, "which", lambda name: str(script) if name == "codex" else None)
    return tmp_path


def _config(**overrides):
    return UploadAiReview(provider="codex", enabled=True, **overrides)


def _request(config):
    return ai_codex.request_review(
        "Review the supplied release only.", "Album metadata", ai_review._ai_review_schema(), config
    )


def _event(action, event_type="item.completed"):
    return {"type": event_type, "item": {"type": "web_search", "action": action}}


def test_api_provider_still_requires_its_key_but_codex_uses_saved_login():
    with pytest.raises(ValueError, match="api_key must be set"):
        UploadAiReview(enabled=True)
    assert _config().api_key is None
    assert UploadAiReview().provider == "openai"
    assert UploadAiReview().model == "gpt-5.4-2026-03-05"
    with pytest.raises(ValueError, match="codex_model must name"):
        _config(codex_model=" ")
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert({"provider": "typo"}, type=UploadAiReview)


def test_codex_uses_isolated_working_directory_and_passes_only_review_inputs(cli, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-use-api-billing")
    monkeypatch.setenv("TRACKER_SESSION", "must-not-share")
    monkeypatch.setenv("CODEX_HOME", str(cli / "saved-login"))
    config = _config(codex_model="gpt-6-astra", reasoning_effort="high")

    review = anyio.run(_request, config)

    assert review["metadata"] == make_review()["metadata"]
    request = json.loads((cli / "request.json").read_text())
    args = request["args"]
    assert args[0] == "exec" and args[-1] == "-"
    assert args[args.index("--model") + 1] == "gpt-6-astra"
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in args and "--ignore-user-config" in args and "--skip-git-repo-check" in args
    assert 'model_reasoning_effort="high"' in args and 'web_search="live"' in args
    assert "features.shell_tool=false" in args and "features.plugins=false" in args and "features.apps=false" in args
    assert "features.multi_agent=false" in args and "project_doc_max_bytes=0" in args
    assert request["prompt"] == "Album metadata"
    assert "Album metadata" not in " ".join(args)
    assert request["schema"] == ai_review._ai_review_schema()
    assert request["env"]["CODEX_HOME"] == str(cli / "saved-login")
    assert "OPENAI_API_KEY" not in request["env"] and "TRACKER_SESSION" not in request["env"]
    assert not Path(request["cwd"]).exists()  # Temporary inputs, output and stderr were removed.


def test_codex_only_trusts_completed_page_actions_for_evidence(cli):
    events = [
        _event({"type": "open_page", "url": "https://label.example/release"}),
        _event({"type": "find_in_page", "url": "https://store.example/album"}),
        _event({"type": "open_page", "url": "https://label.example/release"}),
        _event({"type": "open_page", "url": "https://unfinished.example"}, "item.started"),
        _event({"type": "search", "url": "https://snippet.example"}),
        _event({"type": "open_page", "url": "turn0search1"}),
        _event({"type": "open_page", "url": "file:///secret"}),
        _event({"type": "open_page", "url": None}),
    ]
    (cli / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events))
    response = make_review()
    response["_opened_page_urls"] = ["https://invented.example"]
    (cli / "response.json").write_text(json.dumps(response))

    review = anyio.run(_request, _config())

    assert review["_opened_page_urls"] == ["https://label.example/release", "https://store.example/album"]


def test_codex_without_web_search_does_not_claim_citations_were_opened(cli):
    review = anyio.run(_request, _config(use_web_search=False))

    assert review["_opened_page_urls"] == []
    assert 'web_search="disabled"' in json.loads((cli / "request.json").read_text())["args"]


@pytest.mark.parametrize(
    ("mode", "response", "events", "message"),
    [
        ("failed", "{}", "", "exited with code 1"),
        ("missing", "{}", "", "missing or invalid JSON"),
        ("success", "not JSON", "", "missing or invalid JSON"),
        ("success", "[]", "", "unexpected response shape"),
        ("success", "{}", "", "unexpected response shape"),
        ("success", "{}", "not JSON", "missing or invalid JSON"),
        ("success", "{}", '{"type":"turn.failed"}', "review failed"),
    ],
)
def test_codex_failures_are_actionable_and_cleanup_outputs(cli, mode, response, events, message):
    (cli / "mode").write_text(mode)
    (cli / "response.json").write_text(response)
    (cli / "events.jsonl").write_text(events)

    with pytest.raises(RuntimeError, match=message) as error:
        anyio.run(_request, _config())

    assert "secret diagnostic" not in str(error.value)
    assert not Path(json.loads((cli / "request.json").read_text())["cwd"]).exists()


def test_codex_missing_executable_explains_installation(monkeypatch):
    monkeypatch.setattr(ai_codex.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="install Codex.*codex login"):
        anyio.run(_request, _config())


def _assert_process_cleaned(cli):
    request = json.loads((cli / "request.json").read_text())
    assert not Path(request["cwd"]).exists()
    with pytest.raises(ProcessLookupError):
        os.kill(request["pid"], 0)


@pytest.mark.skipif(os.name != "posix", reason="checks the exited process with POSIX kill(pid, 0)")
def test_codex_timeout_kills_and_reaps_the_process(cli):
    (cli / "mode").write_text("hang")
    with pytest.raises(RuntimeError, match="timed out after 5s"):
        anyio.run(_request, _config(timeout_seconds=5))
    _assert_process_cleaned(cli)


@pytest.mark.skipif(os.name != "posix", reason="checks the exited process with POSIX kill(pid, 0)")
def test_cancelling_a_review_kills_and_reaps_the_process(cli):
    (cli / "mode").write_text("hang")

    async def run():
        with anyio.fail_after(5):
            async with anyio.create_task_group() as group:
                group.start_soon(_request, _config())
                while not (cli / "request.json").exists():
                    await anyio.sleep(0.01)
                group.cancel_scope.cancel()

    anyio.run(run)
    _assert_process_cleaned(cli)


@pytest.mark.skipif(sys.platform != "linux", reason="checks a launcher's child state through /proc")
def test_cancelling_a_launcher_also_kills_its_native_child(cli):
    (cli / "mode").write_text("launcher")

    async def run():
        with anyio.fail_after(5):
            async with anyio.create_task_group() as group:
                group.start_soon(_request, _config())
                while not (cli / "child.pid").exists():
                    await anyio.sleep(0.01)
                group.cancel_scope.cancel()
            stat = Path(f"/proc/{(cli / 'child.pid').read_text()}/stat")
            # A dead orphan can briefly be a zombie until the OS reaps it.
            while stat.exists() and stat.read_text().split()[2] != "Z":
                await anyio.sleep(0.01)

    anyio.run(run)
    _assert_process_cleaned(cli)


def test_codex_dispatch_includes_followup_instruction_and_not_an_api_response_id(cli, monkeypatch):
    monkeypatch.setattr(cfg.upload, "ai_review", _config())

    result, response_id = anyio.run(
        ai_review._request_ai_review,
        make_metadata(),
        make_metadata(),
        "https://store.example/album",
        "Double-check the edition year",
        "previous-api-response",
    )

    request = json.loads((cli / "request.json").read_text())
    assert "Double-check the edition year" in request["prompt"]
    assert "https://store.example/album" in request["prompt"]
    assert "previous-api-response" not in request["prompt"]
    assert response_id is None and result["metadata"] == make_review()["metadata"]
