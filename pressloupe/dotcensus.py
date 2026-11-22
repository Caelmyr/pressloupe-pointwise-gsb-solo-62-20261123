"""Run and review a screen-lattice halftone dot census."""
import numpy as np
from PIL import Image

from . import dotattribution as attribution
from .art import MAX_PREVIEW, base_rgba, load_artifact_picture, stored_preview
from .dots import build_census, downscale_census
from .model import InputError, dictionary, enumeration, model_version, string


def _preview_size(extent):
    width, height = extent
    scale = min(MAX_PREVIEW / max(width, height), 1.0)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


class _Canvas:
    """Frame-shaped view over the stored RGBA preview used for reattribution."""

    def __init__(self, image, scale, valid=None):
        self._image = image
        self.extent = image.size
        self.scale = scale
        self.valid = np.ones((image.size[1], image.size[0]), dtype=bool) if valid is None else valid

    def display(self):
        return np.asarray(self._image.convert("RGB"), dtype=np.float32) / 255


def run_dot_census(store, frame, area, options):
    """Segment the proof, fit one lattice placement and save the pixel plan."""
    full_census = build_census(frame, area, options)
    target_size = _preview_size(frame.extent)
    census = downscale_census(full_census, target_size)
    source_preview = base_rgba(frame)
    source_preview.thumbnail((MAX_PREVIEW, MAX_PREVIEW), Image.Resampling.LANCZOS)
    preview_scale = None
    if frame.scale:
        factor_x = frame.extent[0] / target_size[0]
        factor_y = frame.extent[1] / target_size[1]
        preview_scale = (frame.scale[0] * factor_x, frame.scale[1] * factor_y)
    canvas = _Canvas(source_preview, preview_scale, census["valid"])
    groups = attribution.plan_groups(census, options)
    rows = attribution._dot_rows(census, groups, canvas)
    overlay = attribution.render_overlay(canvas, census, groups, rows)
    ink_png, valid_png, site_png = attribution.encode_maps(census)
    plan = {
        "ink_map_digest": store.blobs.put(ink_png),
        "valid_map_digest": store.blobs.put(valid_png),
        "site_map_digest": store.blobs.put(site_png),
        "site_coordinates": census["site_coordinates"],
        "lattice_fit": census["lattice"],
        "lattice_spans": census["lattice_spans"],
        "source_extent": [int(value) for value in target_size],
        "options": options,
    }
    metrics = {
        **attribution.summarize(census, groups, rows, options),
        "rows": rows[:600],
        "table_truncated": len(rows) > 600,
        "unresolved_groups": attribution.unresolved_listing(groups),
        "note": attribution.note_text(census, groups),
        "census_plan": plan,
    }
    return metrics, overlay, groups


def _review_canvas(store, record, physical_scale, valid):
    """Open the stored source plane at the exact overlay resolution."""
    image = load_artifact_picture(store.blobs, record["pictures"]["source"])
    plan = record["metrics"]["census_plan"]
    if list(image.size) != plan["source_extent"]:
        raise InputError("The saved source plane no longer matches the census dimensions.")
    scale = None
    if physical_scale:
        scale = (physical_scale[0], physical_scale[1])
    return _Canvas(image, scale, valid)


def _load_full_census(store, record, physical_scale=None):
    """Rebuild the in-memory census from the maps frozen with the result."""
    plan = record["metrics"]["census_plan"]
    source = load_artifact_picture(store.blobs, record["pictures"]["source"])
    if list(source.size) != plan["source_extent"]:
        raise InputError("The saved source plane no longer matches the census dimensions.")
    ink, valid, site_map = attribution.decode_maps(store.blobs, plan, source.size[::-1])
    canvas = _review_canvas(store, record, physical_scale, valid)
    census = attribution.reconstruct_census(ink, valid, site_map, plan)
    choices = {
        key: detail["disposition"]
        for key, detail in record.get("dot_decisions", {}).items()
    }
    attribution.apply_review_choices(census["groups"], choices)
    attribution.reflow_group_metadata(census, census["groups"], plan["options"])
    return census, plan, canvas


def review_dot_group(store, record, payload, physical_scale=None):
    data = dictionary(payload, "Dot attribution review")
    group_key = string(data.get("group"), "Dot group ID", maximum=12)
    choice = enumeration(data.get("disposition"), attribution.DISPOSITIONS, "Attribution")
    notes = string(data.get("notes"), "Attribution notes", maximum=240, required=False)
    expected = model_version(data)
    census, plan, canvas = _load_full_census(store, record, physical_scale)
    group = next((item for item in census["groups"] if item["group"] == group_key), None)
    if group is None:
        raise InputError("That ink group is no longer part of this census.")
    attribution.validate_choice(group, choice)
    if choice == "auto":
        group["reviewed_disposition"] = None
    else:
        group["reviewed_disposition"] = choice
    attribution.reflow_group_metadata(census, census["groups"], plan["options"])
    rows = attribution._dot_rows(census, census["groups"], canvas)
    overlay = attribution.render_overlay(canvas, census, census["groups"], rows)
    record["metrics"].update({
        **attribution.summarize(census, census["groups"], rows, plan["options"]),
        "rows": rows[:600],
        "table_truncated": len(rows) > 600,
        "unresolved_groups": attribution.unresolved_listing(census["groups"]),
        "note": attribution.note_text(census, census["groups"]),
    })
    decisions = dict(record.get("dot_decisions", {}))
    if choice == "auto":
        decisions.pop(group_key, None)
    else:
        decisions[group_key] = {
            "disposition": choice,
            "notes": notes,
            "group": group_key,
        }
    record["dot_decisions"] = decisions
    record["pictures"]["overlay"] = stored_preview(overlay, store.blobs)
    return record, expected


def dot_group_catalog(record):
    return {
        "lattice_fit": record["metrics"]["lattice_fit"],
        "unresolved_groups": record["metrics"]["unresolved_groups"],
        "manually_attributed_groups": record["metrics"]["manually_attributed_groups"],
    }


def save_dot_review(store, measurement_id, payload):
    record = store.result(measurement_id)
    if record["kind"] != "dots":
        raise InputError("This measurement has no screen-lattice dot census to review.")
    tile = store.tile(record["tile_id"])
    physical = None
    if tile.get("width_mm") and tile.get("height_mm"):
        source_extent = record["metrics"]["census_plan"]["source_extent"]
        physical = (
            tile["width_mm"] / source_extent[0],
            tile["height_mm"] / source_extent[1],
        )
    updated, expected = review_dot_group(store, record, payload, physical)
    return store.save_dot_attribution(measurement_id, expected, updated)
