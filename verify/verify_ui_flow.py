"""Full browser flow for plan -> approve -> dual-lamp match -> missing lamp."""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

from pressloupe.geometry import DEFAULT_CORNERS, canvas_dimensions
from pressloupe.model import LAMP_ROLES
from pressloupe.photos import png_bytes, store_capture
from pressloupe.web import create_app
from pressloupe.store import make_id

PLANE = (460, 360)
BASE = os.environ.get("PRESSLOUPE_BASE", "http://127.0.0.1:19982")
PORT = int(os.environ.get("PRESSLOUPE_FLOW_PORT", "19982"))


def make_photo(store, role, rgb, alpha, name):
    rgba = np.dstack([np.rint(np.clip(rgb, 0, 1) * 255), np.rint(alpha * 255)]).astype(np.uint8)
    return store_capture(store.blobs, png_bytes(Image.fromarray(rgba)), role,
                         f"{name}-{role}.png", "t", make_id(), synthetic=True)


def rectangle(name, l, t, r, b):
    return {"id": make_id(), "name": name, "purpose": "inspection",
            "vertices": [[l, t], [r, t], [r, b], [l, b]]}


def seed(app, name, gradient=False):
    store = app.extensions["pressloupe.store"]
    h, w = PLANE[1], PLANE[0]
    gx = np.tile(np.linspace(0, 1, w), (h, 1))
    base = np.zeros((h, w, 3), dtype=np.float32)
    if gradient:
        base[..., 0] = 0.32 + 0.30 * gx
        base[..., 1] = 0.40 - 0.10 * gx
        base[..., 2] = 0.45 - 0.18 * gx
    else:
        base[:] = (0.33, 0.39, 0.44)
    std = np.clip(base * (1.00, 1.00, 0.98), 0, 1)
    warm = np.clip(base * (1.10, 0.96, 0.80), 0, 1)
    alpha = np.ones((h, w), dtype=np.float32)
    photos = [make_photo(store, LAMP_ROLES[0], std, alpha, name),
              make_photo(store, LAMP_ROLES[1], warm, alpha, name)]
    region = rectangle("Strip", 0.08, 0.15, 0.92, 0.85)
    tile = store.create_tile({
        "name": name, "body": "e", "glaze": "g", "firing": "f", "batch": "b", "notes": "",
        "width_mm": 100, "height_mm": 78, "photos": photos,
        "plane": {"photo_id": photos[0]["id"], "corners": DEFAULT_CORNERS,
                  "size": list(canvas_dimensions(photos[0], DEFAULT_CORNERS, 100, 78))},
        "regions": [region],
        "registrations": {photos[1]["id"]: {"base_photo_id": photos[0]["id"],
                                             "corners": DEFAULT_CORNERS, "support_percent": 100.0}},
    })
    return tile, region


def main():
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(tmp)
        signing, region = seed(app, "Signing 9", gradient=True)
        target, target_region = seed(app, "Target 9")

        from waitress import serve
        threading.Thread(
            target=serve, args=(app,),
            kwargs={"host": "127.0.0.1", "port": PORT, "threads": 2},
            daemon=True,
        ).start()
        time.sleep(1.5)

        errors = []
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.on("pageerror", lambda exc: errors.append("PAGEERROR: " + str(exc)))
            page.on("console", lambda m: errors.append("CONSOLE: " + m.text)
                    if m.type == "error" and "422" not in m.text else None)

            page.goto(f"{BASE}/#tile:{signing['id']}", wait_until="networkidle")
            page.wait_for_selector(".tabs")

            page.click('[data-do="swatch-plan"]')
            page.wait_for_selector("#field-target_tile_id")
            page.select_option("#field-target_tile_id", target["id"])
            page.select_option("#field-target_region_id", target_region["id"])
            page.select_option("#field-region_id", region["id"])
            page.click('[data-do="swatch-preview"]')
            page.wait_for_selector("#swatch-report .stage img", timeout=20000)
            report = page.locator("#swatch-report")
            assert "Standard lamp ΔE to target" in report.inner_text()
            assert "Warm lamp ΔE to target" in report.inner_text()
            assert report.locator("table tbody tr").count() >= 1
            print("plan rendered with ranked candidates")

            page.click("#dialog-form button[type=submit]")
            page.wait_for_function("location.hash.endsWith('references')", timeout=15000)
            page.wait_for_selector(".collection")
            lib_text = page.locator(".collection").inner_text()
            assert "Dual-lamp physical swatch" in lib_text
            assert "cut 6 × 8 mm" in lib_text
            print("library card shows dual-lamp swatch")

            page.goto(f"{BASE}/#tile:{signing['id']}", wait_until="networkidle")
            page.wait_for_selector(".tabs")
            page.click('[data-do="tile-tab"][data-value="measure"]')
            page.select_option("#field-kind", "reference")
            page.wait_for_selector("#field-reference_id")
            ref_id = page.evaluate("""() => {
              const opts = [...document.querySelector('#field-reference_id').options].filter(o => o.value);
              return opts[0].value;
            }""")
            page.select_option("#field-reference_id", ref_id)
            page.wait_for_selector("#field-lamp_comparison_id")
            warm_id = next(o["id"] for o in signing["photos"] if o["role"] == LAMP_ROLES[1])
            std_id = next(o["id"] for o in signing["photos"] if o["role"] == LAMP_ROLES[0])
            page.select_option("#field-photo_id", std_id)
            page.select_option("#field-lamp_comparison_id", warm_id)
            page.select_option("#field-region_id", region["id"])
            page.click("#measurement-form button.primary")
            page.wait_for_selector(".card.stack", timeout=20000)
            body = page.locator("#main").inner_text()
            assert "Standard lamp (D50 booth)" in body
            assert "Workshop warm lamp" in body
            assert "missing — not captured" not in body
            print("dual-lamp match rendered both lamps")

            page.select_option("#field-lamp_comparison_id", "")
            page.click("#measurement-form button.primary")
            page.wait_for_timeout(1500)
            body = page.locator("#main").inner_text()
            assert "missing — not captured" in body
            print("missing warm lamp shown explicitly")

            assert not errors, errors
            browser.close()
        print("FULL UI FLOW PASSED")


if __name__ == "__main__":
    main()
