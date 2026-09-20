import time
from pathlib import Path

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, sync_playwright
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import Settings


class ClinicalReviewApp:
    """Page-object wrapper for the deployed Streamlit chatbot."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pw = None
        self.browser = None
        self.context = None
        self.page: Page | None = None

    def __enter__(self):
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=self.settings.headless)
        self.context = self.browser.new_context(viewport={"width": 1440, "height": 1000})
        self.page = self.context.new_page()
        self.page.set_default_timeout(self.settings.question_timeout_seconds * 1000)
        self.page.goto(self.settings.app_url, wait_until="domcontentloaded", timeout=self.settings.page_load_timeout_seconds * 1000)
        self._wait_until_awake()
        return self

    def __exit__(self, *_):
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self._pw:
            self._pw.stop()

    def _wait_until_awake(self) -> None:
        assert self.page
        self.page.get_by_placeholder(self.settings.question_placeholder).wait_for(
            state="visible", timeout=self.settings.page_load_timeout_seconds * 1000
        )

    def upload_knowledge_base(self, path: Path) -> None:
        assert self.page
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Knowledge base not found: {resolved}")
        file_input = self.page.locator('input[type="file"]')
        file_input.set_input_files(str(resolved))
        self.page.wait_for_timeout(2500)
        if self.page.get_by_text(resolved.name, exact=False).count() == 0:
            # Streamlit may hide the filename, so textbox readiness is the final contract.
            self.page.get_by_placeholder(self.settings.question_placeholder).wait_for(state="visible")

    def _assistant_messages(self):
        assert self.page
        # Current Streamlit chat markup plus a role-based fallback.
        return self.page.locator('[data-testid="stChatMessage"]').filter(has=self.page.locator('[data-testid="stMarkdownContainer"]'))

    def ask(self, question: str) -> tuple[str, float]:
        return self._ask_with_retry(question)

    @retry(
        retry=retry_if_exception_type((PlaywrightTimeoutError, RuntimeError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _ask_with_retry(self, question: str) -> tuple[str, float]:
        assert self.page
        box = self.page.get_by_placeholder(self.settings.question_placeholder)
        before = self._assistant_messages().count()
        started = time.perf_counter()
        box.fill(question)
        box.press("Enter")

        self.page.wait_for_function(
            "expected => document.querySelectorAll('[data-testid=stChatMessage]').length > expected",
            arg=before,
            timeout=self.settings.question_timeout_seconds * 1000,
        )
        self.page.locator('[data-testid="stChatMessage"]').last.wait_for(state="visible")
        # Wait until Streamlit has finished streaming and the input becomes usable again.
        self.page.wait_for_function(
            "placeholder => { const e=[...document.querySelectorAll('textarea')].find(x=>x.placeholder===placeholder); return e && !e.disabled; }",
            arg=self.settings.question_placeholder,
            timeout=self.settings.question_timeout_seconds * 1000,
        )
        answer = self.page.locator('[data-testid="stChatMessage"]').last.inner_text().strip()
        if not answer:
            raise RuntimeError("The application returned an empty answer")
        return answer, round(time.perf_counter() - started, 3)

