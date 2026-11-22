"""End-to-end verification for the dual-illuminant cut swatch workflow.

Builds synthetic signing and target specimens with one standard-lamp and one
warm-lamp capture each (plus a deliberate reflection and out-of-paper area),
then exercises planning, approval, same-lamp matching and the missing-lamp rule.
Run with: python3 -I verify_swatch.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from pressloupe.geometry import canvas_dimensions, DEFAULT_CORNERS  # noqa: E402
from pressloupe.model import LAMP_ROLES  # noqa: E402
from pressloupe.photos import png_bytes, store_capture  # noqa: E402
from pressloupe.store import Store, make_id  # noqa: E402
from pressloupe.swatch import build_plan  # noqa: E402
from pressloupe.analysis import approve_swatch_reference, evaluate  # noqa: E402


PLANE = (520, 400)
PHOTO = (600, 470)


def make_capture(store, role, rgb, alpha, seed_name):
    image = Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    rgba = np.dstack([np.rint(np.clip(rgb, 0, 1) * 255), np.rint(alpha * 255)]).astype(np.uint8)
    return store_capture(
        store.blobs,
        png_bytes(Image.fromarray(rgba)),
        role,
        f"{seed_name}-{role}.png",
        "synthetic verification capture",
        make_id(),
        synthetic=True,
    )


def signing_sheets():
    """Return two lamp linear-RGB planes of a horizontal gradient, with a glare
    hotspot under the warm lamp and an alpha border (out-of-paper pixels)."""
    h, w = PLANE[1], PLANE[0]
    y, x = np.mgrid[0:h, 0:w]
    gx = x / w
    # Gradient ink: left side matches the target under standard light, right side
    # drifts toward a warmer hue (like the described signing strip).
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[..., 0] = 0.32 + 0.30 * gx
    base[..., 1] = 0.40 - 0.10 * gx
    base[..., 2] = 0.45 - 0.18 * gx
    rng = np.random.default_rng(7)
    base += rng.normal(0, 0.006, base.shape).astype(np.float32)

    standard = np.clip(base * np.array([1.00, 1.00, 0.98]), 0, 1)
    warm = np.clip(base * np.array([1.10, 0.96, 0.80]), 0, 1)
    # Specular hotspot only under the warm lamp in a known patch.
    hotspot = ((x - 380) ** 2 / (2 * 26.0 ** 2) + (y - 120) ** 2 / (2 * 22.0 ** 2)) < 1
    warm[hotspot] = np.minimum(warm[hotspot] + 0.45, 1.0)

    alpha = np.ones((h, w), dtype=np.float32)
    alpha[:18, :] = 0.0          # out-of-paper border
    alpha[-14:, :] = 0.0
    alpha[:, :12] = 0.0
    return standard, warm, alpha, hotspot


def target_sheets():
    h, w = PLANE[1], PLANE[0]
    y, x = np.mgrid[0:h, 0:w]
    # Uniform target swatch color, roughly matching the left gradient stop.
    color = np.array([0.33, 0.39, 0.44], dtype=np.float32)
    base = np.broadcast_to(color, (h, w, 3)).copy()
    rng = np.random.default_rng(11)
    base += rng.normal(0, 0.004, base.shape).astype(np.float32)
    standard = np.clip(base * np.array([1.00, 1.00, 0.98]), 0, 1)
    warm = np.clip(base * np.array([1.10, 0.96, 0.80]), 0, 1)
    alpha = np.ones((h, w), dtype=np.float32)
    return standard, warm, alpha


def build_tile(store, name, sheets, regions):
    standard, warm, alpha = sheets[:3]
    photos = [
        make_capture(store, LAMP_ROLES[0], standard, alpha, name),
        make_capture(store, LAMP_ROLES[1], warm, alpha, name),
    ]
    size = canvas_dimensions(photos[0], DEFAULT_CORNERS, 104, 80)
    fields = {
        "name": name,
        "body": "Verification edition",
        "glaze": "First pull",
        "firing": "Dual lamp",
        "batch": "verify",
        "notes": "",
        "width_mm": 104,
        "height_mm": 80,
        "photos": photos,
        "plane": {"photo_id": photos[0]["id"], "corners": DEFAULT_CORNERS, "size": list(size)},
        "regions": regions,
        "registrations": {
            photos[1]["id"]: {
                "base_photo_id": photos[0]["id"],
                "corners": DEFAULT_CORNERS,
                "support_percent": 100.0,
            },
        },
    }
    return store.create_tile(fields)


def rectangle(name, purpose, l, t, r, b):
    return {
        "id": make_id(),
        "name": name,
        "purpose": purpose,
        "vertices": [[l, t], [r, t], [r, b], [l, b]],
    }


def main():
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(tmp)
        sign_regions = [
            rectangle("Signing strip", "inspection", 0.08, 0.18, 0.92, 0.82),
            rectangle("Ruler and crosses", "ignore", 0.45, 0.85, 0.78, 0.97),
        ]
        target_regions = [rectangle("Target swatch", "inspection", 0.2, 0.2, 0.8, 0.8)]
        signing = build_tile(store, "Signing gradient 4127", signing_sheets(), sign_regions)
        target = build_tile(store, "Target 4127", target_sheets(), target_regions)

        params = {
            "tile_id": signing["id"],
            "target_tile_id": target["id"],
            "region_id": sign_regions[0]["id"],
            "target_region_id": target_regions[0]["id"],
            "width_mm": 6.0,
            "height_mm": 8.0,
            "stride_mm": 2.0,
            "glare_quantile": 0.90,
            "glare_ratio": 1.18,
            "min_support_fraction": 0.80,
        }
        plan = build_plan(store, signing, target, params)
        selected = plan["selected"]
        print("== plan ==")
        print("candidates:", len(plan["candidates"]))
        print("selected bounds:", selected["bounds"], selected["orientation"])
        for role in LAMP_ROLES:
            lamp = selected["lamps"][role]
            print(
                f"  {role}: dE={lamp['delta_e76_to_target']} "
                f"pixels={lamp['measured_pixels']} lab={lamp['median_lab']}",
            )
        assert selected["lamps"][LAMP_ROLES[0]]["measured_pixels"] >= 240
        assert selected["lamps"][LAMP_ROLES[1]]["measured_pixels"] >= 240
        # The glare hotspot sits near x~0.73; a valid cut must not center on it.
        cx = (selected["bounds"][0] + selected["bounds"][2]) / 2
        assert not (0.66 < cx < 0.82), "planner selected the warm-lamp glare zone"

        approve_payload = {
            "tile_id": signing["id"],
            "target_tile_id": target["id"],
            "title": "PMS 4127 physical swatch",
            "color_number": "4127",
            "width_mm": 6.0,
            "height_mm": 8.0,
            "bounds": selected["bounds"],
            "orientation": selected["orientation"],
            "region_id": sign_regions[0]["id"],
            "target_region_id": target_regions[0]["id"],
        }
        ref = approve_swatch_reference(store, approve_payload)
        assert ref["dual_illuminant"] is True
        assert set(ref["illuminants"]) == set(LAMP_ROLES)
        print("\n== approved reference ==")
        for role in LAMP_ROLES:
            entry = ref["illuminants"][role]
            print(f"  {role}: target dE={entry['target_delta_e76']} pixels={entry['eligible_pixels']}")
        # Both fingerprints describe the same normalized rectangle.
        assert ref["swatch"]["bounds"] == selected["bounds"]

        # New press sheet: same construction, slightly shifted ink.
        new_sheets = signing_sheets()
        std, warm = new_sheets[0], new_sheets[1]
        std = std * np.array([1.03, 0.98, 1.0])
        warm = warm * np.array([1.04, 0.97, 0.97])
        press = build_tile(store, "New pull 4127", (std, warm, new_sheets[2]), sign_regions)
        lamp_ids = {photo["role"]: photo["id"] for photo in press["photos"]}
        match = evaluate(store, press["id"], {
            "kind": "reference",
            "photo_id": lamp_ids[LAMP_ROLES[0]],
            "lamp_comparison_id": lamp_ids[LAMP_ROLES[1]],
            "region_id": sign_regions[0]["id"],
            "apply_neutral": True,
            "options": {"reference_id": ref["id"]},
        })
        print("\n== dual-lamp reference match ==")
        for entry in match["metrics"]["illuminant_matches"]:
            print(" ", json.dumps(entry))
        statuses = {e["role"]: e["status"] for e in match["metrics"]["illuminant_matches"]}
        assert all(status == "measured" for status in statuses.values())
        assert match["metrics"]["missing_lamp_count"] == 0

        # Missing warm lamp: must be reported, never substituted.
        only_std = evaluate(store, press["id"], {
            "kind": "reference",
            "photo_id": lamp_ids[LAMP_ROLES[0]],
            "region_id": sign_regions[0]["id"],
            "apply_neutral": True,
            "options": {"reference_id": ref["id"]},
        })
        print("\n== missing warm lamp ==")
        for entry in only_std["metrics"]["illuminant_matches"]:
            print(" ", entry["role"], entry["status"], entry.get("note", ""))
        warm_entry = next(e for e in only_std["metrics"]["illuminant_matches"] if e["role"] == LAMP_ROLES[1])
        assert warm_entry["status"] == "not-captured"
        assert only_std["metrics"]["missing_lamp_count"] == 1
        assert only_std["metrics"]["measured_lamp_count"] == 1

        # Wrong-role substitution must be rejected.
        wrong = None
        try:
            evaluate(store, press["id"], {
                "kind": "reference",
                "photo_id": lamp_ids[LAMP_ROLES[0]],
                "lamp_comparison_id": lamp_ids[LAMP_ROLES[0]],
                "region_id": sign_regions[0]["id"],
                "apply_neutral": True,
                "options": {"reference_id": ref["id"]},
            })
        except Exception as exc:  # noqa: BLE001
            wrong = str(exc)
        print("\n== wrong-role guard ==")
        print(" ", wrong)
        assert wrong and "cannot be substituted" in wrong

        # Glare pixels are excluded under the warm lamp (hotspot exists).
        warm_excluded = ref["illuminants"][LAMP_ROLES[1]]["glare_pixels_excluded"]
        print("\nwarm-lamp glare pixels inside cut:", warm_excluded)

        print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
