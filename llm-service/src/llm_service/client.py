import os
import time


class LLMUnavailableError(Exception):
    """Raised when the LLM cannot be reached or the circuit breaker is open."""


class CircuitBreaker:
    """Stop calling a failing service after N consecutive failures.

    Closed  = calls allowed. Open = calls rejected immediately.
    After `reset_after` seconds the breaker allows one trial call; success
    closes it again, failure re-opens it.
    """

    def __init__(self, threshold=5, reset_after=60.0):
        self.threshold = threshold
        self.reset_after = reset_after
        self.consecutive_failures = 0
        self.opened_at = None

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.monotonic() - self.opened_at >= self.reset_after:
            self.opened_at = None
            self.consecutive_failures = 0
            return False
        return True

    def record_success(self):
        self.consecutive_failures = 0
        self.opened_at = None

    def record_failure(self):
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.threshold:
            self.opened_at = time.monotonic()


class LLMClient:
    """Anthropic API client with timeout, retries and a circuit breaker."""

    def __init__(self, api_key=None, model=None, timeout=10.0,
                 max_retries=2, max_tokens=1200, temperature=0.2):
        self.api_key = api_key or os.getenv('LLM_API_KEY', '')
        self.model = model or os.getenv('LLM_MODEL', 'claude-sonnet-4-5')
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.breaker = CircuitBreaker()
        self._client = None

    @property
    def enabled(self) -> bool:
        return True

    def _get_client(self):
        if self._client is None:
            import anthropic
            if not self.api_key:
                raise LLMUnavailableError(
                    "LLM_API_KEY is not set. Set it in .env, or set "
                    "LLM_ENABLED=false to run without narrative generation."
                )
            self._client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout)
        return self._client

    def generate(self, system: str, user: str) -> str:
        if self.breaker.is_open:
            raise LLMUnavailableError(
                f"Circuit breaker open after {self.breaker.consecutive_failures} "
                f"consecutive failures. Not calling the API."
            )

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                client = self._get_client()
                response = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system,
                    messages=[{'role': 'user', 'content': user}],
                )
                self.breaker.record_success()
                return response.content[0].text
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)      # 1s, then 2s

        self.breaker.record_failure()
        raise LLMUnavailableError(
            f"LLM call failed after {self.max_retries + 1} attempts: {last_error}"
        )


class NullLLMClient(LLMClient):
    """Offline stand-in. Never makes a network call.

    Used when LLM_ENABLED=false and in every unit test.
    """

    def __init__(self):
        super().__init__(api_key='', model='null')

    @property
    def enabled(self) -> bool:
        return False

    def generate(self, system: str, user: str) -> str:
        raise LLMUnavailableError("LLM is disabled (NullLLMClient).")