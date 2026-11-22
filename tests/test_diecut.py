"""End-to-end check of the die-layout workbench and fold-line classification.

Builds a synthetic double-sided hang tag: fine text next to a shaped window on
the front, a barcode near the outer edge on the (mirrored) back, and one
unphotographed patch. Exercises layout creation, the three press positions,
position selection and the per-millimetre fold-line trace.
"""

import tempfile
import unittest

import numpy as np
from PIL import Image

from pressloupe.geometry import DEFAULT_CORNERS, canvas_dimensions
from pressloupe.photos import png_bytes, store_capture
from pressloupe.store import make_id
from pressloupe.web import create_app

MM_W, MM_H = 140.0, 90.0
PXMM = 5
WIDTH, HEIGHT = int(MM_W * PXMM), int(MM_H * PXMM)
OUTER_MM = [[8, 6], [132, 6], [132, 84], [8, 84]]
WINDOW_MM = [[62, 30], [84, 34], [80, 58], [66, 54], [60, 42]]
STEP_MM = 2.0


def mm(points):
    return [[round(x / MM_W, 5), round(y / MM_H, 5)] for x, y in points]


def blank_photo():
    photo = np.full((HEIGHT, WIDTH, 4), 255, dtype=np.uint8)
    photo[..., :3] = (232, 228, 220)
    return photo


def paint_text(photo):
    # Eight fine dark characters along y 36..42 mm, starting at x = 50 mm.
    for index in range(8):
        x0 = round((50 + index * 3.0) * PXMM)
        x1 = round((50 + index * 3.0 + 2.2) * PXMM)
        photo[round(36 * PXMM):round(42 * PXMM), x0:x1, :3] = (38, 36, 34)


def paint_barcode_back(photo):
    # Bars at sheet x 118..135 mm appear mirrored in the back view (horizontal flip).
    for bar in range(9):
        sheet_x = 118 + bar * 2
        view_x0 = round((MM_W - sheet_x - 1) * PXMM)
        view_x1 = round((MM_W - sheet_x) * PXMM)
        photo[round(20 * PXMM):round(55 * PXMM), view_x0:view_x1, :3] = (30, 30, 32)


def make_tile(store, name, photo_array, regions):
    png = png_bytes(Image.fromarray(photo_array, "RGBA"))
    photo = store_capture(store.blobs, png, "diffuse", f"{name}.png", "", make_id())
    size = canvas_dimensions(photo, DEFAULT_CORNERS, MM_W, MM_H)
    return store.create_tile({
        "name": name, "batch": "", "body": "", "glaze": "", "firing": "", "notes": "",
        "width_mm": MM_W, "height_mm": MM_H,
        "photos": [photo],
        "plane": {"photo_id": photo["id"], "corners": DEFAULT_CORNERS, "size": list(size)},
        "regions": regions,
        "registrations": {},
    })


def region(name, x0, y0, x1, y1):
    return {
        "id": make_id(), "name": name, "purpose": "inspection",
        "vertices": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
    }


class DieLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.app = create_app(cls.tmp.name)
        cls.client = cls.app.test_client()
        cls.store = cls.app.extensions["pressloupe.store"]

        front_photo = blank_photo()
        paint_text(front_photo)
        # An unphotographed patch (transparent) over the leftmost character.
        front_photo[round(34 * PXMM):round(44 * PXMM), round(46 * PXMM):round(51 * PXMM), 3] = 0
        cls.text_region = region("Front text line", 0.34, 0.36, 0.55, 0.50)
        cls.front = make_tile(cls.store, "tag-front", front_photo, [cls.text_region])

        back_photo = blank_photo()
        paint_barcode_back(back_photo)
        cls.barcode_region = region("Back barcode", 0.03, 0.19, 0.17, 0.65)
        cls.back = make_tile(cls.store, "tag-back", back_photo, [cls.barcode_region])

        payload = {
            "name": "Tag die",
            "back_tile_id": cls.back["id"],
            "flip": "horizontal",
            "step_mm": STEP_MM,
            "contrast": 0.16,
            "front_region_id": cls.text_region["id"],
            "back_region_id": cls.barcode_region["id"],
            "outer": mm(OUTER_MM),
            "windows": [mm(WINDOW_MM)],
        }
        created = cls.client.post(f"/api/specimens/{cls.front['id']}/layouts", json=payload)
        assert created.status_code == 201, created.get_json()
        cls.layout = created.get_json()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_evaluation_metrics_follow_the_shift(self):
        evaluation = self.layout["evaluation"]
        front = evaluation["sides"]["front"]["per_position"]
        kept = [front[str(p)]["kept_percent"] for p in (-1, 0, 1)]
        self.assertLess(kept[0], kept[1])
        self.assertLess(kept[1], kept[2])
        back = evaluation["sides"]["back"]["per_position"]
        bars_cut = [back[str(p)]["broken_components"] for p in (-1, 0, 1)]
        self.assertEqual(bars_cut, [3, 2, 1])
        self.assertEqual(evaluation["recommended"], 1)
        # The unphotographed patch is reported for review, not judged.
        self.assertGreater(evaluation["sides"]["front"]["review_percent"], 3)
        self.assertGreaterEqual(evaluation["sides"]["front"]["uncertain_components"], 1)
        self.assertEqual(evaluation["sides"]["back"]["review_percent"], 0)

    def test_layout_listed_on_both_sides(self):
        for tile in (self.front, self.back):
            listed = self.client.get(f"/api/specimens/{tile['id']}/layouts")
            self.assertEqual(listed.status_code, 200)
            self.assertEqual([item["id"] for item in listed.get_json()["items"]], [self.layout["id"]])

    def test_overlays_and_document(self):
        for side in ("front", "back"):
            for position in (-1, 0, 1):
                res = self.client.get(f"/api/layouts/{self.layout['id']}/overlay/{side}/{position}")
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.mimetype, "image/png")
                self.assertGreater(len(res.get_data()), 2000)
        self.assertEqual(self.client.get(f"/api/layouts/{self.layout['id']}/overlay/side/0").status_code, 422)
        self.assertEqual(self.client.get(f"/api/layouts/{self.layout['id']}/overlay/front/2").status_code, 422)
        self.assertEqual(self.client.get(f"/api/layouts/{self.layout['id']}/data.json").status_code, 200)

    def test_select_position_with_version_guard(self):
        current = self.client.get(f"/api/layouts/{self.layout['id']}").get_json()
        stale = self.client.post(
            f"/api/layouts/{self.layout['id']}/select",
            json={"version": 99, "position": 1},
        )
        self.assertEqual(stale.status_code, 409)
        chosen = self.client.post(
            f"/api/layouts/{self.layout['id']}/select",
            json={"version": current["version"], "position": 1},
        )
        self.assertEqual(chosen.status_code, 200)
        self.assertEqual(chosen.get_json()["selected"], 1)

    def trace(self, layout_id=None):
        payload = {
            "title": "Fold line", "mode": "trace",
            "points": [[0.30, 0.4], [0.99, 0.4]],
        }
        if layout_id:
            payload["layout_id"] = layout_id
        res = self.client.post(f"/api/specimens/{self.front['id']}/metrology", json=payload)
        self.assertEqual(res.status_code, 201, res.get_json())
        return res.get_json()

    def test_fold_line_classification_at_chosen_position(self):
        # Choose Right first so the trace uses the committed die position.
        self.client.post(
            f"/api/layouts/{self.layout['id']}/select",
            json={"version": self.layout["version"], "position": 1},
        )
        record = self.trace(self.layout["id"])
        shape = record["shape"]
        self.assertEqual(shape["die_layout"]["position"], 1)
        self.assertTrue(shape["die_layout"]["selected"])
        totals = shape["state_totals_mm"]
        self.assertAlmostEqual(totals["window"], 22.7, delta=1.6)
        self.assertAlmostEqual(totals["unphotographed"], 5.0, delta=1.0)
        self.assertAlmostEqual(totals["waste"], 4.6, delta=1.0)
        states = {row["state"] for row in shape["millimeters"]}
        self.assertEqual(states, {"product", "window", "waste", "unphotographed"})
        first = shape["millimeters"][0]
        self.assertEqual(first["index"], 1)
        self.assertEqual(first["from_mm"], 0)
        self.assertAlmostEqual(first["start_xy"][0], 0.30, places=2)
        self.assertAlmostEqual(first["start_xy"][1], 0.40, places=2)
        overlay = self.client.get(f"/api/metrology/{record['id']}/overlay.png")
        self.assertEqual(overlay.status_code, 200)
        self.assertGreater(len(overlay.get_data()), 2000)

    def test_trace_without_layout_stays_plain(self):
        record = self.trace()
        self.assertNotIn("millimeters", record["shape"])

    def test_layout_validation(self):
        base = {
            "name": "Bad die", "back_tile_id": self.back["id"], "flip": "horizontal",
            "step_mm": 2, "contrast": 0.16, "front_region_id": None, "back_region_id": None,
        }
        outside = {**base, "outer": mm(OUTER_MM), "windows": [mm([[200, 30], [210, 30], [205, 40]])]}
        self.assertEqual(self.client.post(f"/api/specimens/{self.front['id']}/layouts", json=outside).status_code, 422)
        crossing = {**base, "outer": mm(OUTER_MM), "windows": [mm([[60, 30], [150, 40], [70, 50]])]}
        self.assertEqual(self.client.post(f"/api/specimens/{self.front['id']}/layouts", json=crossing).status_code, 422)
        overlapping = {
            **base, "outer": mm(OUTER_MM),
            "windows": [mm([[40, 30], [60, 30], [50, 50]]), mm([[50, 35], [70, 35], [60, 55]])],
        }
        self.assertEqual(self.client.post(f"/api/specimens/{self.front['id']}/layouts", json=overlapping).status_code, 422)
        missing = {**base, "back_tile_id": "0" * 24, "outer": mm(OUTER_MM), "windows": [mm(WINDOW_MM)]}
        self.assertEqual(self.client.post(f"/api/specimens/{self.front['id']}/layouts", json=missing).status_code, 404)


class SheetMappingTest(unittest.TestCase):
    def test_flip_mappings(self):
        from pressloupe.diecut import sheet_to_plane
        sheet = [MM_W, MM_H]
        front = sheet_to_plane([[35, 45]], sheet, "front", "horizontal")
        self.assertEqual(front.tolist(), [[0.25, 0.5]])
        horizontal = sheet_to_plane([[35, 45]], sheet, "back", "horizontal")
        self.assertEqual(horizontal.tolist(), [[0.75, 0.5]])
        vertical = sheet_to_plane([[35, 45]], sheet, "back", "vertical")
        self.assertEqual(vertical.tolist(), [[0.25, 0.5]])
        vertical = sheet_to_plane([[35, 30]], sheet, "back", "vertical")
        self.assertAlmostEqual(vertical[0][1], (MM_H - 30) / MM_H)


if __name__ == "__main__":
    unittest.main()
