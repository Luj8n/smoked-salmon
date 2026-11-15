import re
import ssl
import time
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError

import anyio.to_thread
import musicbrainzngs
from musicbrainzngs import musicbrainz as mb_module

from salmon.errors import ScrapeError
from salmon.release_notification import get_version
from salmon.sources.retry import RETRY_DELAYS, TRANSIENT_STATUSES, browser_check, retry_delay, say_retry

from .base import BaseScraper

musicbrainzngs.set_useragent("smoked-salmon", get_version(), "https://github.com/smokin-salmon/smoked-salmon")

# musicbrainzngs has no timeout and tries a failing request up to 8 times (7 retries, sleeping 56 s in all). Every
# request it makes, the Cover Art Archive's too, goes through its private _safe_read, so that is where it is bounded.
# A test pins the parameters the wrapper passes.
MAX_REQUESTS = len(RETRY_DELAYS) + 1
REQUEST_TIMEOUT = 10  # Seconds per socket operation

_safe_read = mb_module._safe_read


class _TimeoutOpener:
    """A urllib opener whose requests time out after REQUEST_TIMEOUT seconds."""

    def __init__(self, opener: Any) -> None:
        self._opener = opener

    def open(self, req: Any, body: Any = None) -> Any:
        return self._opener.open(req, body, timeout=REQUEST_TIMEOUT)


def _bounded_safe_read(opener, req, body=None):
    waited = 0.0
    for attempt in range(MAX_REQUESTS):
        try:
            # Own the whole retry budget: do not multiply it by the library's internal retries.
            return _safe_read(_TimeoutOpener(opener), req, body, max_retries=1)
        except mb_module.WebServiceError as error:
            if body is not None or req.get_method() != "GET" or not _transient(error.cause):
                raise
            after = error.cause.headers.get("Retry-After") if isinstance(error.cause, HTTPError) else None
            delay = retry_delay(attempt, after, waited)
            if delay is None:
                raise
            say_retry("MusicBrainz / Cover Art Archive", delay)
            time.sleep(delay)
            waited += delay
    raise AssertionError("unreachable")


mb_module._safe_read = _bounded_safe_read


def _is_browser_check(error: HTTPError) -> bool:
    """Inspect only a small error body; never expose its URLs, cookies or other contents in the log."""
    cached = getattr(error, "_salmon_browser_check", None)
    if cached is not None:
        return cached
    try:
        body = error.read(16384).decode("utf-8", errors="replace")
    except (OSError, ValueError, AttributeError):
        return False
    challenged = browser_check(body)
    error.__dict__["_salmon_browser_check"] = challenged
    return challenged


def _transient(cause: Any) -> bool:
    if isinstance(cause, HTTPError):
        return cause.code in TRANSIENT_STATUSES and not _is_browser_check(cause)
    if isinstance(cause, URLError):
        cause = cause.reason
    return not isinstance(cause, ssl.SSLError) and isinstance(cause, (TimeoutError, ConnectionError))


class MusicBrainzBase(BaseScraper):
    store = "MusicBrainz"
    url = site_url = "https://musicbrainz.org"
    release_format = "/release/{rls_id}"
    regex = re.compile(r"^https?://(?:www\.)?musicbrainz\.org/release/([a-f0-9\-]+)/?(?=$|[?#])")

    async def _call(self, func: Callable[..., Any], *args: Any) -> Any:
        """Run a musicbrainzngs call in a thread, its errors as ScrapeError.

        The message names the error type, never the request URL, as get_json's do.

        Raises:
            ScrapeError: If the request failed, or MusicBrainz has no such release.
        """
        try:
            return await anyio.to_thread.run_sync(func, *args)
        except mb_module.WebServiceError as e:
            if isinstance(e.cause, HTTPError) and e.cause.code == 404:
                raise ScrapeError(f"{self.store or self.__class__.__name__}: Not found.") from e
            if isinstance(e.cause, HTTPError):
                if _is_browser_check(e.cause):
                    raise ScrapeError(
                        f"MusicBrainz: Browser verification page received (HTTP {e.cause.code}); "
                        "the API request was blocked. Try again later."
                    ) from e
                raise ScrapeError(
                    f"{self.store or self.__class__.__name__}: Request failed (HTTP {e.cause.code})."
                ) from e
            raise ScrapeError(f"{self.store or self.__class__.__name__}: Request failed ({type(e).__name__}).") from e

    async def fetch_data(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
        follow_redirects: bool = True,
        rls_id: Any = None,
    ) -> dict[str, Any]:
        """Fetch release data from MusicBrainz API.

        Args:
            url: The MusicBrainz release URL.
            params: Unused, kept for API compatibility.
            headers: Unused, kept for API compatibility.
            follow_redirects: Unused, kept for API compatibility.

        Returns:
            Release data dict from MusicBrainz.

        Raises:
            ValueError: If URL is invalid.
            ScrapeError: If the request failed, or MusicBrainz has no such release.
        """
        match = self.regex.match(url)
        if not match:
            raise ValueError("Invalid MusicBrainz URL.")
        rls_id = match[1]
        return (
            await self._call(
                musicbrainzngs.get_release_by_id,
                rls_id,
                [
                    "artists",
                    "labels",
                    "recordings",
                    "release-groups",
                    "media",
                    "artist-credits",
                    "artist-rels",
                    "recording-level-rels",
                ],
            )
        )["release"]
