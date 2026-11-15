"""Album metadata review through `codex exec`, using the CLI's saved authentication.

Only the supplied metadata is sent: no album directory, salmon config or credentials. Each run has an empty
working directory, read-only sandbox, no shell/plugins, and no user/project instructions. Completed web actions,
not the model's citations, supply the opened-page evidence used by ai_review's existing guardrails.
"""

import json
import os
import shutil
import signal
import subprocess
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlsplit

import anyio
import msgspec

from salmon.config.validations import UploadAiReview

# Keep authentication discovery, certificates and proxies, but not service keys or the album's environment.
_ENV_KEYS = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "CODEX_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "XDG_RUNTIME_DIR",
    "DBUS_SESSION_BUS_ADDRESS",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SYSTEMROOT",
    "SystemRoot",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
}
_MAX_OUTPUT_BYTES = 8 * 1024 * 1024


def _command(executable: str, work: Path, instructions: str, config: UploadAiReview) -> list[str]:
    overrides = {
        "approval_policy": "never",
        "model_provider": "openai",
        "model_reasoning_effort": config.reasoning_effort,
        "web_search": "live" if config.use_web_search else "disabled",
        "project_doc_max_bytes": 0,
        "skills.include_instructions": False,
        "features.shell_tool": False,
        "features.apps": False,
        "features.plugins": False,
        "features.remote_plugin": False,
        "features.multi_agent": False,
        "features.memories": False,
        "developer_instructions": instructions,
    }
    command = [
        executable,
        "exec",
        "--ignore-user-config",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "--json",
        "--model",
        config.codex_model,
        "--output-schema",
        str(work / "schema.json"),
        "--output-last-message",
        str(work / "review.json"),
    ]
    for key, value in overrides.items():
        command += ["-c", f"{key}={json.dumps(value)}"]
    return [*command, "-"]


def _read_output(path: Path) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(_MAX_OUTPUT_BYTES + 1)
    if len(data) > _MAX_OUTPUT_BYTES:
        raise RuntimeError("Codex metadata review returned too much output")
    return data


def _opened_pages(events: bytes) -> list[str]:
    """Only completed page opens/finds with actual URLs count; search snippets and citations never do."""
    opened: set[str] = set()
    for line in events.splitlines():
        if not line.strip():
            continue
        event = msgspec.json.decode(line)
        if not isinstance(event, dict):
            raise RuntimeError("Codex metadata review returned an invalid event")
        if event.get("type") == "turn.failed":
            raise RuntimeError("Codex metadata review failed; check `codex login status` and model availability")
        item = event.get("item")
        if event.get("type") != "item.completed" or not isinstance(item, dict) or item.get("type") != "web_search":
            continue
        action = item.get("action")
        if not isinstance(action, dict) or action.get("type") not in {"open_page", "find_in_page"}:
            continue
        url = action.get("url")
        if isinstance(url, str) and urlsplit(url).scheme in {"http", "https"} and urlsplit(url).netloc:
            opened.add(url)
    return sorted(opened)


def _result(work: Path) -> dict[str, Any]:
    try:
        review = msgspec.json.decode(_read_output(work / "review.json"))
        opened = _opened_pages(_read_output(work / "events.jsonl"))
    except (OSError, msgspec.DecodeError, ValueError) as exc:
        raise RuntimeError("Codex metadata review returned missing or invalid JSON output") from exc
    if (
        not isinstance(review, dict)
        or not isinstance(review.get("summary"), str)
        or not isinstance(review.get("metadata"), dict)
        or not isinstance(review.get("citations"), list)
    ):
        raise RuntimeError("Codex metadata review returned an unexpected response shape")
    review["_opened_page_urls"] = opened
    return review


async def _run(command: list[str], prompt: str, work: Path, environment: dict[str, str]) -> int:
    with (work / "events.jsonl").open("wb") as events, (work / "stderr.txt").open("wb") as errors:
        async with await anyio.open_process(
            command,
            stdin=subprocess.PIPE,
            stdout=events,
            stderr=errors,
            cwd=work,
            env=environment,
            start_new_session=os.name == "posix",
        ) as process:
            try:
                assert process.stdin is not None
                await process.stdin.send(prompt.encode())
                await process.stdin.aclose()
                return await process.wait()
            except BaseException:
                # The npm launcher starts a native child. Killing only the launcher leaves that child reviewing.
                # On other platforms AnyIO still kills and reaps the direct process when the context closes.
                if os.name == "posix":
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                raise


async def request_review(
    instructions: str, prompt: str, schema: dict[str, Any], config: UploadAiReview
) -> dict[str, Any]:
    """Run one review, with a deadline; AnyIO kills and reaps the process on timeout or job cancellation.

    A prompted rerun starts fresh with the current metadata and user instruction, not a persisted Codex session.
    Raw stderr is deliberately not echoed: it can contain authentication details or supplied metadata.
    """
    executable = shutil.which("codex")
    if executable is None:
        raise RuntimeError("Codex CLI not found: install Codex, run `codex login`, and put codex on PATH")
    environment = {key: value for key, value in os.environ.items() if key in _ENV_KEYS}
    with TemporaryDirectory(prefix="salmon-ai-review-") as directory:
        work = Path(directory)
        (work / "schema.json").write_bytes(msgspec.json.encode(schema))
        try:
            with anyio.fail_after(config.timeout_seconds):
                returncode = await _run(_command(executable, work, instructions, config), prompt, work, environment)
        except TimeoutError as exc:
            raise RuntimeError(
                f"Codex metadata review timed out after {config.timeout_seconds}s; "
                "increase upload.ai_review.timeout_seconds or lower reasoning_effort"
            ) from exc
        except OSError as exc:
            raise RuntimeError("Could not start Codex CLI; check its installation and permissions") from exc
        if returncode:
            raise RuntimeError(
                f"Codex metadata review exited with code {returncode}; check `codex login status`, "
                "codex_model and your usage allowance. Update Codex if it does not support --ignore-user-config."
            )
        return _result(work)
