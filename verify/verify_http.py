"""HTTP-level smoke test for the swatch planning and library endpoints."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from pressloupe.geometry import DEFAULT_CORNERS, canvas_dimensions
from pressloupe.model import LAMP_ROLES
from pressloupe.photos import png_bytes, store_capture
from pressloupe.store import Store, make_id
from pressloupe.web import create_app

PLANE = (460, 360)


def make_photo(store, role, rgb, alpha, name):
    rgba = np.dstack([np.rint(np.clip(rgb, 0, 1) * 255), np.rint(alpha * 255)]).astype(np.uint8)
    return store_capture(store.blobs, png_bytes(Image.fromarray(rgba)), role,
                         f"{name}-{role}.png", "t", make_id(), synthetic=True)


def rectangle(name, l, t, r, b):
    return {"id": make_id(), "name": name, "purpose": "inspection",
            "vertices": [[l, t], [r, t], [r, b], [l, b]]}


def seed(store, name, gradient=False):
    h, w = PLANE[1], PLANE[0]
    y, x = np.mgrid[0:h, 0:w]
    gx = x / w
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
    return store.create_tile({
        "name": name, "body": "e", "glaze": "g", "firing": "f", "batch": "b", "notes": "",
        "width_mm": 100, "height_mm": 78, "photos": photos,
        "plane": {"photo_id": photos[0]["id"], "corners": DEFAULT_CORNERS,
                  "size": list(canvas_dimensions(photos[0], DEFAULT_CORNERS, 100, 78))},
        "regions": [region],
        "registrations": {photos[1]["id"]: {"base_photo_id": photos[0]["id"],
                                             "corners": DEFAULT_CORNERS, "support_percent": 100.0}},
    }), region


def main():
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(tmp)
        client = app.test_client()
        assert client.get("/api/health").status_code == 200
        store = app.extensions["pressloupe.store"]
        signing, region = seed(store, "Signing 7", gradient=True)
        target, target_region = seed(store, "Target 7")

        plan_payload = {
            "tile_id": signing["id"], "target_tile_id": target["id"],
            "region_id": region["id"], "target_region_id": target_region["id"],
            "width_mm": 6, "height_mm": 8, "stride_mm": 2,
            "glare_quantile": 0.9, "glare_ratio": 1.18, "min_support_fraction": 0.8,
        }
        resp = client.post("/api/library/swatch/plan", json=plan_payload)
        assert resp.status_code == 200, resp.get_json()
        plan = resp.get_json()
        assert len(plan["candidates"]) >= 1
        print("plan endpoint ok, candidates:", len(plan["candidates"]))

        resp = client.post("/api/library/swatch/overlay",
                           json={**plan_payload, "candidate": 0})
        assert resp.status_code == 200 and resp.mimetype == "image/png"
        print("overlay endpoint ok, bytes:", len(resp.data))

        chosen = plan["selected"]
        approve = {
            "tile_id": signing["id"], "target_tile_id": target["id"],
            "title": "Color 7 physical swatch", "color_number": "7",
            "width_mm": 6, "height_mm": 8, "bounds": chosen["bounds"],
            "orientation": chosen["orientation"], "region_id": region["id"],
            "target_region_id": target_region["id"],
            "glare_quantile": 0.9, "glare_ratio": 1.18,
        }
        resp = client.post("/api/library/swatch", json=approve)
        assert resp.status_code == 201, resp.get_json()
        ref = resp.get_json()
        assert ref["dual_illuminant"]
        print("approve endpoint ok, id:", ref["id"])

        library = client.get("/api/library").get_json()["items"]
        assert any(item["id"] == ref["id"] and item["dual_illuminant"] for item in library)

        # Reference match with both lamps.
        lamp_ids = {p["role"]: p["id"] for p in signing["photos"]}
        match_payload = {
            "kind": "reference", "photo_id": lamp_ids[LAMP_ROLES[0]],
            "lamp_comparison_id": lamp_ids[LAMP_ROLES[1]],
            "region_id": region["id"], "apply_neutral": True,
            "options": {"reference_id": ref["id"]},
        }
        resp = client.post(f"/api/specimens/{signing['id']}/results", json=match_payload)
        assert resp.status_code == 201, resp.get_json()
        metrics = resp.get_json()["metrics"]
        assert metrics["measured_lamp_count"] == 2 and metrics["missing_lamp_count"] == 0
        print("dual match ok:", [(e["role"], e["status"]) for e in metrics["illuminant_matches"]])

        # Missing warm lamp.
        one_lamp = {**match_payload, "lamp_comparison_id": None}
        resp = client.post(f"/api/specimens/{signing['id']}/results", json=one_lamp)
        assert resp.status_code == 201
        metrics = resp.get_json()["metrics"]
        assert metrics["missing_lamp_count"] == 1
        warm = next(e for e in metrics["illuminant_matches"] if e["role"] == LAMP_ROLES[1])
        assert warm["status"] == "not-captured"
        print("missing-lamp reporting ok:", warm["note"])

        # Wrong role substitution rejected with 422.
        bad = {**match_payload, "lamp_comparison_id": lamp_ids[LAMP_ROLES[0]]}
        resp = client.post(f"/api/specimens/{signing['id']}/results", json=bad)
        assert resp.status_code == 422
        print("wrong-role rejected:", resp.get_json()["error"])

        # Invalid swatch size rejected.
        bad_plan = {**plan_payload, "width_mm": 500}
        resp = client.post("/api/library/swatch/plan", json=bad_plan)
        assert resp.status_code == 422
        print("validation rejected oversized swatch")

        # Evidence packet containing the dual-lamp match must build and summarize.
        results = client.get(f"/api/specimens/{signing['id']}/results").get_json()["items"]
        dual_result = next(
            r for r in results
            if r["brief"].get("illuminant_matches") and r["brief"]["missing_lamp_count"] == 0
        )
        assert dual_result["brief"]["missing_lamp_count"] == 0
        resp = client.post(f"/api/specimens/{signing['id']}/evidence", json={
            "title": "Dual lamp sign-off", "result_ids": [dual_result["id"]],
            "notes": "", "tags": [], "view": "overlay", "include_sources": False,
        })
        assert resp.status_code == 201, resp.get_json()
        packet = resp.get_json()
        html = client.get(f"/api/evidence/{packet['id']}/inspection.html")
        assert html.status_code == 200 and b"Same-illuminant matches" in html.data
        print("evidence packet and HTML report ok")

        # A sheet missing one lamp cannot produce a dual-lamp swatch plan.
        current = store.tile(signing["id"])
        warm_photo = next(p for p in current["photos"] if p["role"] == LAMP_ROLES[1])
        store.edit_tile(
            signing["id"], current["version"],
            lambda t: (t["photos"].remove(next(p for p in t["photos"] if p["id"] == warm_photo["id"])),
                       t["registrations"].pop(warm_photo["id"], None)),
            "capture", "removed warm lamp",
        )
        resp = client.post("/api/library/swatch/plan", json=plan_payload)
        assert resp.status_code == 422 and "both lamp" in resp.get_json()["error"]
        print("single-lamp planning rejected:", resp.get_json()["error"])
        print("HTTP SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
