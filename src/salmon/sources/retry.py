"""Bounded retries for source GETs, never tracker requests or authentication POSTs."""

import re
from collections.abc import Awaitable, Callable
from functools import wraps
from random import uniform
from typing import Any

import aiohttp
import anyio
import asyncclick as click

from salmon.common.urls import parse_retry_after
from salmon.errors import ScrapeError

RETRY_DELAYS = (3.0, 10.0)
MAX_RETRY_WAIT = 60.0
TRANSIENT_STATUSES = {429, 500, 502, 503, 504}


def browser_check(body: Any) -> bool:
    if not isinstance(body, str):
        return False
    title = re.search(r"<title\b[^>]*>(.*?)</title>", body[:16384], re.IGNORECASE | re.DOTALL)
    return bool(
        title
        and any(
            marker in title[1].casefold() for marker in ("verifying your browser", "client challenge", "just a moment")
        )
    )


class SourceHTTPError(ScrapeError):
    def __init__(self, message: str, status: int, retry_after: str | None, payload=None):
        super().__init__(message, payload)
        self.status = status
        self.retry_after = retry_after


def blocks_fallback(error: ScrapeError) -> bool:
    """A cooldown or browser check must not be followed by requests to another endpoint."""
    return isinstance(error, SourceHTTPError) and (
        error.status == 429 or parse_retry_after(error.retry_after) is not None or browser_check(error.payload)
    )


def retry_delay(attempt: int, retry_after: str | None, waited: float) -> float | None:
    if attempt >= len(RETRY_DELAYS):
        return None
    delay = max(RETRY_DELAYS[attempt], parse_retry_after(retry_after) or 0) + uniform(0, 1)
    # Never shorten the server's requested wait just to squeeze in another attempt.
    return delay if waited + delay <= MAX_RETRY_WAIT else None


def transient(error: ScrapeError) -> bool:
    if isinstance(error, SourceHTTPError):
        return error.status in TRANSIENT_STATUSES and not browser_check(error.payload)
    cause = error.__cause__
    if isinstance(cause, (aiohttp.ClientSSLError, aiohttp.ClientConnectorCertificateError)):
        return False
    return isinstance(cause, (TimeoutError, aiohttp.ClientConnectionError, aiohttp.ClientPayloadError))


def say_retry(store: str, delay: float) -> None:
    click.secho(f"{store}: temporary request problem; retrying in {delay:.0f}s…", fg="yellow")


def retry_get(operation: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    @wraps(operation)
    async def run(self, *args, **kwargs):
        waited = 0.0
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                return await operation(self, *args, **kwargs)
            except ScrapeError as error:
                if not self.retry_gets or not transient(error):
                    raise
                after = error.retry_after if isinstance(error, SourceHTTPError) else None
                delay = retry_delay(attempt, after, waited)
                if delay is None:
                    raise
                say_retry(self.store or type(self).__name__, delay)
                await anyio.sleep(delay)
                waited += delay
        raise AssertionError("unreachable")

    return run
