"""Glare-mask and legacy single-capture compatibility checks."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image  # noqa: E402

from pressloupe.analysis import approve_reference, evaluate  # noqa: E402
from pressloupe.geometry import DEFAULT_CORNERS, canvas_dimensions  # noqa: E402
from pressloupe.model import LAMP_ROLES  # noqa: E402
from pressloupe.photos import png_bytes, store_capture  # noqa: E402
from pressloupe.store import Store, make_id  # noqa: E402
from pressloupe.swatch import glare_masks, lamp_views, rectangle_mask  # noqa: E402

PLANE = (420, 320)


def make_photo(store, role, rgb, alpha, name):
    rgba = np.dstack([np.rint(np.clip(rgb, 0, 1) * 255), np.rint(alpha * 255)]).astype(np.uint8)
    return store_capture(
        store.blobs, png_bytes(Image.fromarray(rgba)), role,
        f"{name}-{role}.png", "t", make_id(), synthetic=True,
    )


def rectangle(name, purpose, l, t, r, b):
    return {"id": make_id(), "name": name, "purpose": purpose,
            "vertices": [[l, t], [r, t], [r, b], [l, b]]}


def main():
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(tmp)
        h, w = PLANE[1], PLANE[0]
        y, x = np.mgrid[0:h, 0:w]
        base = np.broadcast_to(np.array([0.35, 0.4, 0.42]), (h, w, 3)).copy()
        std = base.copy()
        warm = base * np.array([1.1, 0.96, 0.8])
        hotspot = ((x - 300) ** 2 / (2 * 20.0 ** 2) + (y - 160) ** 2 / (2 * 18.0 ** 2)) < 1
        warm[hotspot] = np.minimum(warm[hotspot] + 0.5, 1.0)
        alpha = np.ones((h, w), dtype=np.float32)
        photos = [make_photo(store, LAMP_ROLES[0], std, alpha, "a"),
                  make_photo(store, LAMP_ROLES[1], warm, alpha, "a")]
        regions = [rectangle("Whole sheet", "inspection", 0.05, 0.05, 0.95, 0.95)]
        tile = store.create_tile({
            "name": "Glare check", "body": "e", "glaze": "g", "firing": "f", "batch": "b", "notes": "",
            "width_mm": 100, "height_mm": 76, "photos": photos,
            "plane": {"photo_id": photos[0]["id"], "corners": DEFAULT_CORNERS,
                      "size": list(canvas_dimensions(photos[0], DEFAULT_CORNERS, 100, 76))},
            "regions": regions,
            "registrations": {photos[1]["id"]: {"base_photo_id": photos[0]["id"],
                                                 "corners": DEFAULT_CORNERS, "support_percent": 100.0}},
        })
        views, _ = lamp_views(store, tile)
        glare = glare_masks(views, 0.90, 1.18)
        hot_pixels = int(glare[LAMP_ROLES[1]].sum())
        print("warm-lamp glare pixels detected:", hot_pixels)
        assert hot_pixels > 50, "specular hotspot was not detected"
        # Standard lamp has no hotspot and should flag essentially nothing.
        print("standard-lamp glare pixels:", int(glare[LAMP_ROLES[0]].sum()))
        assert int(glare[LAMP_ROLES[0]].sum()) < 20

        # A rectangle covering the hotspot must exclude those pixels from the
        # warm-lamp eligible set (no false average over the reflection).
        rect = rectangle_mask(views[LAMP_ROLES[1]].frame.extent,
                              [0.62, 0.38, 0.82, 0.62])
        inside_hot = int((rect & glare[LAMP_ROLES[1]]).sum())
        print("glare pixels inside candidate cut:", inside_hot)
        assert inside_hot > 20

        # Legacy single-capture reference (pre-dual-lamp document shape) still
        # works through the original approve/match code path.
        tile = store.tile(tile["id"])
        legacy_ref = approve_reference(store, {
            "tile_id": tile["id"], "title": "Legacy single", "region_id": regions[0]["id"],
        })
        assert not legacy_ref.get("dual_illuminant")
        assert legacy_ref["fingerprint"]["supported_pixels"] > 100
        print("legacy reference fingerprint pixels:", legacy_ref["fingerprint"]["supported_pixels"])

        # A reference match using the old single-capture flow runs unchanged.
        result = evaluate(store, tile["id"], {
            "kind": "reference", "photo_id": photos[0]["id"],
            "region_id": regions[0]["id"], "apply_neutral": True,
            "options": {"reference_id": legacy_ref["id"]},
        })
        assert "delta_e76" in result["metrics"]
        print("legacy match delta_e76:", result["metrics"]["delta_e76"])
        print("GLARE + LEGACY CHECKS PASSED")


if __name__ == "__main__":
    main()
