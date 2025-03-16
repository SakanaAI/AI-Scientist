"""Rate limit handling for AI-Scientist API calls."""
import time
import logging
from typing import Optional, Callable, Any
from functools import wraps
import backoff
from queue import Queue, Empty
from threading import Lock

import openai
import anthropic
import google.api_core.exceptions
import requests

class RateLimitHandler:
    """Handles rate limiting across different API providers."""

    def __init__(self):
        self._request_queues = {}  # Per-provider request queues
        self._locks = {}  # Per-provider locks
        self._last_request_time = {}  # Per-provider last request timestamps
        self._min_request_interval = {
            'openai': 1.0,  # 1 request per second
            'anthropic': 0.5,  # 2 requests per second
            'google': 1.0,  # 1 request per second
            'xai': 1.0,  # 1 request per second
            'semantic_scholar': 1.0,  # 1 request per second
            'default': 1.0  # Default fallback
        }
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('rate_limit_handler')

    def _get_provider_key(self, model: str) -> str:
        """Map model name to provider key."""
        if 'gpt' in model or model.startswith('o1-'):
            return 'openai'
        elif 'claude' in model:
            return 'anthropic'
        elif 'gemini' in model:
            return 'google'
        elif 'grok' in model:
            return 'xai'
        return 'default'

    def _ensure_provider_initialized(self, provider: str):
        """Initialize provider-specific resources if not already done."""
        if provider not in self._request_queues:
            self._request_queues[provider] = Queue()
        if provider not in self._locks:
            self._locks[provider] = Lock()
        if provider not in self._last_request_time:
            self._last_request_time[provider] = 0

    def handle_rate_limit(self, model: str) -> Callable:
        """Decorator for handling rate limits for specific models."""
        provider = self._get_provider_key(model)
        self._ensure_provider_initialized(provider)

        def on_backoff(details):
            """Callback for backoff events."""
            wait_time = details['wait']
            tries = details['tries']
            func_name = details['target'].__name__
            logging.warning(
                f"Rate limit hit for {model} ({provider}). "
                f"Backing off {wait_time:.1f}s after {tries} tries "
                f"calling {func_name} at {time.strftime('%X')}"
            )

        def on_success(details):
            """Callback for successful requests."""
            if details['tries'] > 1:
                logging.info(
                    f"Successfully completed request for {model} after "
                    f"{details['tries']} attempts"
                )

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self._locks[provider]:
                    # Enforce minimum interval between requests
                    current_time = time.time()
                    time_since_last = current_time - self._last_request_time[provider]
                    if time_since_last < self._min_request_interval[provider]:
                        sleep_time = self._min_request_interval[provider] - time_since_last
                        time.sleep(sleep_time)

                    try:
                        # Use exponential backoff for rate limits
                        @backoff.on_exception(
                            backoff.expo,
                            (
                                Exception,  # Catch all exceptions to check if rate limit
                            ),
                            max_tries=8,  # Maximum number of retries
                            on_backoff=on_backoff,
                            on_success=on_success,
                            giveup=lambda e: not self._is_rate_limit_error(e)
                        )
                        def _execute_with_backoff():
                            return func(*args, **kwargs)

                        result = _execute_with_backoff()
                        self._last_request_time[provider] = time.time()
                        return result

                    except Exception as e:
                        if self._is_rate_limit_error(e):
                            logging.error(
                                f"Rate limit exceeded for {model} ({provider}) "
                                f"after maximum retries: {str(e)}"
                            )
                        raise

            return wrapper

        return decorator

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an error is related to rate limiting."""
        error_str = str(error).lower()
        rate_limit_indicators = [
            'rate limit',
            'too many requests',
            '429',
            'quota exceeded',
            'capacity',
            'throttle'
        ]
        return any(indicator in error_str for indicator in rate_limit_indicators)

# Global rate limit handler instance
rate_limiter = RateLimitHandler()
