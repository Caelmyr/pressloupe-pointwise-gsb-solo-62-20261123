#!/usr/bin/env python3
"""End-to-end check for the back-side show-through alignment feature.

Builds a synthetic two-sided thin sheet:

* the back carries a small dark block of "text" on the left that bleeds through
  as a faint mark on the front, plus strong back marks on the right;
* the front carries its own hairline on the right that has no back counterpart
  and must remain front ink, plus a small front-only glyph in the centre;
* the back photograph is framed slightly off the registered plane, and the
  flipped plate sits a few pixels away so the single translation has work to do.

The script runs the preview, retains the alignment, reopens the specimen, runs
an ink coverage measurement and checks the lower/upper front-ink bounds.
Run from the repository root: python3 scripts/verify_showthrough.py
"""
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pressloupe.geometry import DEFAULT_CORNERS  # noqa: E402
from pressloupe.photos import png_bytes, store_capture  # noqa: E402
from pressloupe.showthrough import FLIP_EDGES  # noqa: E402
from pressloupe.store import make_id  # noqa: E402
from pressloupe.web import create_app  # noqa: E402


W, H = 360, 280
PAPER = 0.78


def sheet_with_marks(marks, faint=()):
    image = np.full((H, W, 3), PAPER, dtype=np.float32)
    rng = np.random.default_rng(7)
    image += rng.normal(0, 0.002, image.shape).astype(np.float32)
    for (x0, y0, x1, y1), tone in marks:
        image[y0:y1, x0:x1] = tone
    # Faint bleed is a gentle darkening toward (but not as far as) the tone.
    for (x0, y0, x1, y1), tone in faint:
        image[y0:y1, x0:x1] = np.minimum(image[y0:y1, x0:x1], tone)
    return Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8))


def main():
    workdir = Path(tempfile.mkdtemp(prefix="showthrough-check-"))
    try:
        app = create_app(data_dir=workdir / "data")
        client = app.test_client()

        # --- create specimen -------------------------------------------------
        created = client.post("/api/specimens", json={"name": "Two-sided thin sheet"})
        assert created.status_code == 201, created.data
        tile = created.get_json()
        tile_id = tile["id"]

        # Back plate: small dark "character" on the back-left; with a left-edge
        # flip it bleeds through onto the front at the mirrored (right-hand) spot.
        back_marks = [
            ((40, 60, 62, 80), 0.12),      # small back-left glyph (mirrors to front x 298-319)
            ((30, 200, 52, 220), 0.10),    # second back glyph, mirrored far from front marks
        ]
        # Front: faint show-through of the mirrored back glyph, plus genuinely
        # front-only work (a left-zone hairline and a centre glyph) that must be
        # retained as front ink even though a back plate exists elsewhere.
        front_marks = [
            ((120, 200, 170, 204), 0.18),  # front-only hairline (right anchor zone)
            ((170, 150, 196, 172), 0.16),  # front-only centre glyph
        ]
        faint = [
            # Same small glyph shifted (dy=6, dx=7) relative to the flipped plate:
            # one shared translation recovers the shared core, while the part
            # outside the plate falls in its halo and stays ambiguous (ink or
            # bleed cannot be distinguished there).
            ((305, 66, 330, 86), 0.60),
        ]
        front_image = sheet_with_marks(front_marks, faint)
        back_image = sheet_with_marks(back_marks)

        def upload(image, role):
            reply = client.post(
                f"/api/specimens/{tile_id}/photos",
                data={"photo": (io.BytesIO(png_bytes(image)), f"{role}.png"), "role": role,
                      "version": tile["version"]},
                content_type="multipart/form-data",
            )
            assert reply.status_code == 201, reply.data
            return reply.get_json()

        tile = upload(front_image, "diffuse")
        front_id = tile["photos"][0]["id"]
        tile = upload(back_image, "reference")
        back_id = next(p["id"] for p in tile["photos"] if p["role"] == "reference")

        # Register the back photo to the same plane (no offset in this synthetic
        # case; the flip + translation search still has to recover the shift).
        reg = client.put(
            f"/api/specimens/{tile_id}/registrations/{back_id}",
            json={"version": tile["version"], "corners": DEFAULT_CORNERS},
        )
        assert reg.status_code == 200, reg.data
        tile = reg.get_json()

        # Left anchor encloses the offset bleed-through glyph; right anchor the
        # front-only hairline that the back plate must not be allowed to explain.
        left_anchor = [[0.74, 0.14], [0.96, 0.14], [0.96, 0.40], [0.74, 0.40]]
        right_anchor = [[0.30, 0.66], [0.52, 0.66], [0.52, 0.78], [0.30, 0.78]]
        payload = {
            "front_photo_id": front_id,
            "back_photo_id": back_id,
            "flip_edge": "left",
            "left_anchor": left_anchor,
            "right_anchor": right_anchor,
        }

        # --- preview ---------------------------------------------------------
        preview = client.post(f"/api/specimens/{tile_id}/show-through/preview", json=payload)
        assert preview.status_code == 200, preview.data
        record = preview.get_json()
        dy, dx = record["translation_px"]
        print("translation:", dy, dx)
        print("fit:", json.dumps(record["fit"], indent=2))
        print("anchors:", json.dumps(record["anchors"], indent=2))
        print("bounds:", json.dumps(record["coverage_bounds"], indent=2))

        # Left anchor darkening must be largely explained; right hairline kept.
        left_row = next(a for a in record["anchors"] if a["anchor"] == "show_through")
        right_row = next(a for a in record["anchors"] if a["anchor"] == "front_only")
        assert left_row["dark_explained_percent"] >= 60, left_row
        assert right_row["covered_by_back_plate_px"] == 0, right_row
        assert record["coverage_bounds"]["confirmed_show_through_px"] > 0
        # The offset leaves some dark pixels only in the plate halo: these stay
        # ambiguous, so the upper bound must strictly exceed the lower bound.
        assert record["coverage_bounds"]["ambiguous_px"] > 0, record["coverage_bounds"]
        lower = record["coverage_bounds"]["front_ink_lower_percent"]
        upper = record["coverage_bounds"]["front_ink_upper_percent"]
        assert lower is not None and upper is not None and lower < upper
        print("recovered translation:", record["translation_px"], "(expected near [6, 7])")

        # --- preview overlay PNG endpoint -----------------------------------
        overlay_preview = client.post(
            f"/api/specimens/{tile_id}/show-through/preview/overlay", json=payload
        )
        assert overlay_preview.status_code == 200, overlay_preview.data
        assert overlay_preview.data[:8] == b"\x89PNG\r\n\x1a\n"

        # --- validation: overlapping anchors rejected ------------------------
        bad = dict(payload, right_anchor=payload["left_anchor"])
        rejected = client.post(f"/api/specimens/{tile_id}/show-through/preview", json=bad)
        assert rejected.status_code == 422, rejected.data

        # --- validation: bad flip edge rejected ------------------------------
        bad_edge = dict(payload, flip_edge="diagonal")
        assert client.post(
            f"/api/specimens/{tile_id}/show-through/preview", json=bad_edge
        ).status_code == 422

        # --- validation: same front and back photo rejected ------------------
        same = dict(payload, back_photo_id=front_id)
        assert client.post(
            f"/api/specimens/{tile_id}/show-through/preview", json=same
        ).status_code == 422

        # --- retain (single allowed translation) -----------------------------
        save = client.put(
            f"/api/specimens/{tile_id}/show-through",
            json={"version": tile["version"], **payload},
        )
        assert save.status_code == 201, save.data
        saved = save.get_json()["alignment"]
        assert saved["translation_px"] == [dy, dx]

        # Reopen: alignment must still be present.
        reopened = client.get(f"/api/specimens/{tile_id}").get_json()
        assert reopened.get("showthrough", {}).get("translation_px") == [dy, dx]

        overlay = client.get(f"/api/specimens/{tile_id}/show-through/overlay")
        assert overlay.status_code == 200 and overlay.data[:8] == b"\x89PNG\r\n\x1a\n"

        # --- ordinary coverage still works and carries the new bounds --------
        regions = client.post(
            f"/api/specimens/{tile_id}/regions",
            json={"version": reopened["version"], "name": "Coverage area", "purpose": "inspection",
                  "vertices": [[0.05, 0.08], [0.95, 0.08], [0.95, 0.92], [0.05, 0.92]]},
        )
        assert regions.status_code == 201, regions.data
        region_id = regions.get_json()["regions"][-1]["id"]
        bare = client.post(
            f"/api/specimens/{tile_id}/regions",
            json={"version": regions.get_json()["version"], "name": "Bare paper",
                  "purpose": "bare", "vertices": [[0.02, 0.02], [0.07, 0.02], [0.07, 0.06], [0.02, 0.06]]},
        )
        assert bare.status_code == 201, bare.data
        bare_id = bare.get_json()["regions"][-1]["id"]
        # The front-only centre glyph (170-196 x, 150-172 y) is the solid patch.
        glazed = client.post(
            f"/api/specimens/{tile_id}/regions",
            json={"version": bare.get_json()["version"], "name": "Solid ink patch",
                  "purpose": "glazed",
                  "vertices": [[0.46, 0.52], [0.56, 0.52], [0.56, 0.62], [0.46, 0.62]]},
        )
        assert glazed.status_code == 201, glazed.data
        glazed_id = glazed.get_json()["regions"][-1]["id"]
        measurement = client.post(
            f"/api/specimens/{tile_id}/results",
            json={"kind": "coverage", "photo_id": front_id, "region_id": region_id,
                  "options": {"bare_region": bare_id, "glazed_region": glazed_id,
                              "ambiguity": 0.16, "max_residual_fraction": 0.3, "minimum_island": 22}},
        )
        assert measurement.status_code == 201, measurement.data
        metrics = measurement.get_json()["metrics"]
        assert "show_through_coverage" in metrics, metrics
        stc = metrics["show_through_coverage"]
        print("coverage show-through block:", json.dumps(stc, indent=2))
        assert stc["front_ink_lower_percent"] <= stc["front_ink_upper_percent"]
        assert stc["confirmed_show_through_px"] > 0

        # --- clearing restores ordinary single-sided behaviour ---------------
        current = client.get(f"/api/specimens/{tile_id}").get_json()
        cleared = client.delete(
            f"/api/specimens/{tile_id}/show-through", json={"version": current["version"]}
        )
        assert cleared.status_code == 200, cleared.data
        assert "showthrough" not in cleared.get_json()

        # A coverage run after clearing must be the ordinary single-sided form.
        ordinary = client.post(
            f"/api/specimens/{tile_id}/results",
            json={"kind": "coverage", "photo_id": front_id, "region_id": region_id,
                  "options": {"bare_region": bare_id, "glazed_region": glazed_id,
                              "ambiguity": 0.16, "max_residual_fraction": 0.3, "minimum_island": 22}},
        )
        assert ordinary.status_code == 201, ordinary.data
        assert "show_through_coverage" not in ordinary.get_json()["metrics"]

        print("\nAll show-through checks passed.")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
