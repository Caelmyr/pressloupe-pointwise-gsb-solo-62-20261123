"""Browser-level smoke test: page renders and dialogs open without JS errors."""
import os
import sys
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")

from playwright.sync_api import sync_playwright

BASE = os.environ.get("PRESSLOUPE_BASE", "http://127.0.0.1:19981")


def main():
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on(
            "console",
            lambda msg: errors.append(msg.text)
            if msg.type == "error" and "422" not in msg.text and "404" not in msg.text
            else None,
        )

        page.goto(BASE, wait_until="networkidle")
        page.click('[data-do="training"]')
        page.wait_for_function("location.hash.startsWith('#tile:')", timeout=15000)
        page.wait_for_selector(".tabs", timeout=15000)

        # Open the swatch planner dialog and check it renders.
        page.click('[data-do="swatch-plan"]')
        page.wait_for_selector("#dialog-form")
        assert page.locator("#field-target_tile_id").count() == 1
        assert "same physical rectangle" in page.locator("#dialog").inner_text()
        print("swatch planner dialog opened")

        # Running the plan without two lamp roles on the training tile must show
        # an in-dialog error, not a broken page.
        page.click('[data-do="swatch-preview"]')
        page.wait_for_selector(".dialog-error:not(:empty)", timeout=15000)
        error_text = page.locator(".dialog-error").inner_text()
        print("expected planner error:", error_text)
        assert "lamp" in error_text.lower()
        page.click('[data-do="close"]')

        # Reference library page still renders.
        page.goto(BASE + "/#references", wait_until="networkidle")
        page.wait_for_selector("h1")
        assert "Reference library" in page.locator("h1").inner_text()

        # Open a specimen and the reference approve dialog.
        page.goto(BASE, wait_until="networkidle")
        page.wait_for_selector(".tile-card", timeout=15000)
        page.click(".tile-card")
        page.wait_for_selector(".tabs")
        page.click('[data-do="reference-add"]')
        page.wait_for_selector("#field-title")
        print("legacy reference dialog opened")
        page.click('[data-do="close"]')

        assert not errors, f"JS errors: {errors}"
        browser.close()
    print("UI SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
