"""
API rate limiter with token bucket algorithm for judge API calls.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Prevents exceeding API rate limits for OpenAI and Anthropic judge calls.
    Supports both synchronous and async usage patterns.
    """

    requests_per_minute: int = 60
    tokens_per_minute: int = 150_000
    _request_timestamps: list = field(default_factory=list)
    _token_counts: list = field(default_factory=list)

    def wait_if_needed(self, estimated_tokens: int = 0) -> float:
        """
        Block until a request can proceed without exceeding rate limits.

        Args:
            estimated_tokens: Estimated token count for the request.

        Returns:
            Time waited in seconds (0 if no wait needed).
        """
        now = time.time()
        cutoff = now - 60.0

        # Prune old timestamps
        self._request_timestamps = [t for t in self._request_timestamps if t > cutoff]
        self._token_counts = [
            (t, c) for t, c in self._token_counts if t > cutoff
        ]

        waited = 0.0

        # Check request rate
        if len(self._request_timestamps) >= self.requests_per_minute:
            sleep_time = self._request_timestamps[0] - cutoff
            if sleep_time > 0:
                time.sleep(sleep_time)
                waited += sleep_time

        # Check token rate
        if estimated_tokens > 0:
            total_tokens = sum(c for _, c in self._token_counts) + estimated_tokens
            if total_tokens > self.tokens_per_minute:
                oldest = min(t for t, _ in self._token_counts) if self._token_counts else now
                sleep_time = oldest - cutoff
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    waited += sleep_time

        # Record this request
        self._request_timestamps.append(time.time())
        if estimated_tokens > 0:
            self._token_counts.append((time.time(), estimated_tokens))

        return waited

    async def async_wait_if_needed(self, estimated_tokens: int = 0) -> float:
        """Async version of wait_if_needed."""
        now = time.time()
        cutoff = now - 60.0

        self._request_timestamps = [t for t in self._request_timestamps if t > cutoff]
        self._token_counts = [
            (t, c) for t, c in self._token_counts if t > cutoff
        ]

        waited = 0.0

        if len(self._request_timestamps) >= self.requests_per_minute:
            sleep_time = self._request_timestamps[0] - cutoff
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                waited += sleep_time

        if estimated_tokens > 0:
            total_tokens = sum(c for _, c in self._token_counts) + estimated_tokens
            if total_tokens > self.tokens_per_minute:
                oldest = min(t for t, _ in self._token_counts) if self._token_counts else now
                sleep_time = oldest - cutoff
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    waited += sleep_time

        self._request_timestamps.append(time.time())
        if estimated_tokens > 0:
            self._token_counts.append((time.time(), estimated_tokens))

        return waited

    @property
    def current_rpm(self) -> int:
        """Current requests in the last minute."""
        cutoff = time.time() - 60.0
        return sum(1 for t in self._request_timestamps if t > cutoff)

    @property
    def current_tpm(self) -> int:
        """Current tokens used in the last minute."""
        cutoff = time.time() - 60.0
        return sum(c for t, c in self._token_counts if t > cutoff)
