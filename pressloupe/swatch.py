"""Plan and measure one physical cut swatch seen under both signing lamps.

The invariant enforced here is that every illuminant fingerprint of a swatch is
computed from the *same* registered plane rectangle. A pixel is allowed into one
lamp's average only when that lamp and the other lamp both see it, it lies on
paper (inside the chosen region and outside exclusion regions), it is supported
and unclipped in that lamp, and it is not part of a specular reflection. Pixels
that fail any of these tests are dropped per lamp; they are never replaced by
pixels from another position and never averaged across the two photographs.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .art import base_rgba, stored_preview
from .color import (
    corrected_linear,
    delta_e,
    float_list,
    linear_lab,
    reference_fingerprint,
)
from .geometry import (
    active_region,
    region_domain,
    specimen_frame,
)
from .model import LAMP_ROLES, InputError
from .photos import luminance


MIN_SWATCH_PIXELS = 240
MIN_COMMON_FRACTION = 0.55
LARGEST_FRAGMENT_FRACTION = 0.70
MAX_CANDIDATES = 400
TOP_RANKED = 12


@dataclass(frozen=True)
class LampView:
    role: str
    frame: object
    linear: np.ndarray
    lab: np.ndarray


def _lamp_frame(store, tile, role):
    photo = next((item for item in tile["photos"] if item["role"] == role), None)
    if photo is None:
        return None
    if photo["id"] not in {tile["plane"]["photo_id"], *tile["registrations"]}:
        raise InputError(
            "Register the {role} capture in the specimen plane before planning a swatch.".format(
                role=role,
            ),
        )
    frame = specimen_frame(store.blobs, tile, photo["id"])
    linear, _ = corrected_linear(tile, frame, True)
    return LampView(role, frame, linear, linear_lab(linear))


def lamp_views(store, tile):
    if tile.get("plane") is None:
        raise InputError("Set a specimen plane from a registered capture first.")
    views = {role: _lamp_frame(store, tile, role) for role in LAMP_ROLES}
    present = [role for role in LAMP_ROLES if views[role] is not None]
    if not present:
        raise InputError(
            "Add the two lamp captures (standard lamp and workshop warm lamp) before planning a swatch.",
        )
    return views, present


def require_both_lamps(views, context):
    missing = [role for role in LAMP_ROLES if views.get(role) is None]
    if missing:
        labels = ", ".join(role for role in missing)
        raise InputError(f"{context} needs both lamp captures; missing {labels}.")


def glare_masks(views, quantile, ratio):
    """Conservative specular-reflection masks for each lamp.

    A pixel counts as glare only when its own luminance clears a high local
    quantile and it is distinctly brighter than the same physical point in the
    other lamp. The two conditions together avoid flagging a genuinely bright
    (but matte) ink as a reflection.
    """
    masks = {}
    roles = [role for role in LAMP_ROLES if views.get(role) is not None]
    for role in roles:
        view = views[role]
        lum = luminance(view.linear)
        local = ndimage.gaussian_filter(lum, sigma=5.0, mode="nearest")
        own_floor = np.nanquantile(lum[view.frame.valid], quantile)
        brighter = np.zeros(lum.shape, dtype=bool)
        for other_role in roles:
            if other_role == role:
                continue
            other = views[other_role].linear
            aligned = np.maximum(luminance(other), 1e-4)
            brighter |= lum > aligned * ratio
        masks[role] = view.frame.valid & (local >= own_floor) & brighter
    return masks


def rectangle_mask(extent, bounds):
    left, top, right, bottom = bounds
    width, height = extent
    mask = np.zeros((height, width), dtype=bool)
    x0 = max(0, int(np.floor(left * width)))
    x1 = min(width, int(np.ceil(right * width)))
    y0 = max(0, int(np.floor(top * height)))
    y1 = min(height, int(np.ceil(bottom * height)))
    mask[y0:y1, x0:x1] = True
    return mask


def _largest_fragment_fraction(mask):
    labeled, count = ndimage.label(mask)
    if count == 0:
        return 0.0
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, count + 1))
    return float(sizes.max() / max(mask.sum(), 1))


def eligible_pixel_sets(rect, views, glare, paper_domain):
    """Return the per-lamp measurement masks and a shared coverage summary."""
    per_lamp = {}
    common = rect.copy()
    for role, view in views.items():
        frame = view.frame
        supported = rect & frame.valid & ~frame.clipped & paper_domain & ~glare[role]
        per_lamp[role] = supported
        common &= frame.valid
    total = int(rect.sum())
    summary = {
        "rectangle_pixels": total,
        "common_visible_pixels": int(common.sum()),
        "common_visible_percent": round(100 * common.sum() / max(total, 1), 2),
    }
    return per_lamp, common, summary


def lamp_fingerprint(view, mask):
    lab = view.lab[mask]
    fingerprint = reference_fingerprint(view.linear, mask)
    median = np.asarray(fingerprint["median_lab"], dtype=np.float32)
    distances = delta_e(lab, median)
    return fingerprint, median, distances


def _lamp_record(view, mask, target_fingerprint):
    frame = view.frame
    fingerprint, median, distances = lamp_fingerprint(view, mask)
    record = {
        "role": view.role,
        "photo_id": frame.photo["id"],
        "photo_name": frame.photo["name"],
        "measured_pixels": int(mask.sum()),
        "median_lab": fingerprint["median_lab"],
        "swatch": fingerprint["swatch"],
        "dispersion_p90_delta_e": fingerprint["dispersion"],
        "within_swatch_p90_delta_e": round(float(np.quantile(distances, 0.90)), 3),
        "within_swatch_p95_delta_e": round(float(np.quantile(distances, 0.95)), 3),
        "available": True,
        "status": "measured",
        "note": "Median of this lamp's own eligible pixels inside the shared rectangle.",
    }
    if target_fingerprint is not None:
        target_median = np.asarray(target_fingerprint["median_lab"], dtype=np.float32)
        record["target_median_lab"] = target_fingerprint["median_lab"]
        record["target_swatch"] = target_fingerprint["swatch"]
        record["delta_e76_to_target"] = round(float(delta_e(median, target_median)), 3)
        record["delta_lab_to_target"] = float_list(median - target_median)
    return record


def target_lamp_fingerprints(store, target_tile, target_region_id):
    """Measure the separately photographed target swatch under each lamp."""
    target_views, present = lamp_views(store, target_tile)
    require_both_lamps(target_views, "The target color swatch")
    region = active_region(target_tile, target_region_id)
    targets = {}
    for role in LAMP_ROLES:
        view = target_views[role]
        domain = region_domain(view.frame.extent, target_tile, region)
        mask = domain & view.frame.valid & ~view.frame.clipped
        if mask.sum() < MIN_SWATCH_PIXELS:
            raise InputError(
                f"The target swatch has too few supported pixels under {role}.",
            )
        fingerprint, _, _ = lamp_fingerprint(view, mask)
        targets[role] = fingerprint
    return targets, region


def scan_candidates(views, paper_domain, glare, params, target_fingerprints, scale_xy):
    extent = next(iter(views.values())).frame.extent
    width_px = params["width_mm"] / scale_xy[0]
    height_px = params["height_mm"] / scale_xy[1]
    stride_px = max(1.0, params["stride_mm"] / float(np.mean(scale_xy)))
    plane_w, plane_h = extent
    min_fraction = params["min_support_fraction"]
    candidates = []
    # Two cut orientations: given width along X, and rotated 90 degrees.
    orientations = (
        ("width-x", width_px, height_px),
        ("height-x", height_px, width_px),
    )
    for orientation, rect_w, rect_h in orientations:
        if rect_w >= plane_w or rect_h >= plane_h:
            continue
        for x0 in np.arange(0, plane_w - rect_w, stride_px):
            for y0 in np.arange(0, plane_h - rect_h, stride_px):
                x1, y1 = x0 + rect_w, y0 + rect_h
                bounds = [
                    round(float(x0 / plane_w), 6),
                    round(float(y0 / plane_h), 6),
                    round(float(x1 / plane_w), 6),
                    round(float(y1 / plane_h), 6),
                ]
                rect = rectangle_mask(extent, bounds)
                per_lamp, common, summary = eligible_pixel_sets(
                    rect, views, glare, paper_domain,
                )
                if common.sum() < MIN_SWATCH_PIXELS:
                    continue
                if common.sum() / max(rect.sum(), 1) < MIN_COMMON_FRACTION:
                    continue
                lamp_records = {}
                worst = 0.0
                ok = True
                for role, view in views.items():
                    mask = per_lamp[role]
                    fraction = mask.sum() / max(rect.sum(), 1)
                    if fraction < min_fraction:
                        ok = False
                        break
                    if _largest_fragment_fraction(mask) < LARGEST_FRAGMENT_FRACTION:
                        ok = False
                        break
                    record = _lamp_record(view, mask, target_fingerprints.get(role))
                    lamp_records[role] = record
                    worst = max(worst, record.get("delta_e76_to_target", 0.0))
                if not ok:
                    continue
                candidates.append({
                    "bounds": bounds,
                    "orientation": orientation,
                    "size_mm": [params["width_mm"], params["height_mm"]],
                    "worst_lamp_delta_e76": round(worst, 3),
                    "mean_lamp_delta_e76": round(
                        float(np.mean([r["delta_e76_to_target"] for r in lamp_records.values()])), 3,
                    ),
                    "coverage": summary,
                    "lamps": lamp_records,
                })
                if len(candidates) >= MAX_CANDIDATES:
                    break
            if len(candidates) >= MAX_CANDIDATES:
                break
    candidates.sort(key=lambda item: item["worst_lamp_delta_e76"])
    return candidates[:TOP_RANKED]


def build_plan(store, tile, target_tile, params):
    views, _ = lamp_views(store, tile)
    require_both_lamps(views, "Swatch planning")
    extent = views[LAMP_ROLES[0]].frame.extent
    if any(view.frame.extent != extent for view in views.values()):
        raise InputError("The two lamp captures must register onto the same plane size.")
    scale = views[LAMP_ROLES[0]].frame.scale
    if not scale:
        raise InputError(
            "Set the physical sheet dimensions (mm) so a 6 × 8 mm cut can be located.",
        )
    region = active_region(tile, params["region_id"])
    paper_domain = region_domain(extent, tile, region)
    glare = glare_masks(views, params["glare_quantile"], params["glare_ratio"])
    target_fingerprints, target_region = target_lamp_fingerprints(
        store, target_tile, params["target_region_id"],
    )
    candidates = scan_candidates(
        views, paper_domain, glare, params, target_fingerprints, scale,
    )
    if not candidates:
        raise InputError(
            "No continuous 6 × 8 mm position stays supported, unclipped and glare-free "
            "in both lamp captures. Loosen the search region or mark reflections as exclusions.",
        )
    chosen = candidates[0]
    return {
        "tile_id": tile["id"],
        "target_tile_id": target_tile["id"],
        "target_region": target_region,
        "search_region": region,
        "plane_size": list(extent),
        "scale_mm_per_pixel": [round(scale[0], 6), round(scale[1], 6)],
        "glare_pixels": {role: int(mask.sum()) for role, mask in glare.items()},
        "candidates": candidates,
        "selected": chosen,
        "lamp_roles": list(LAMP_ROLES),
    }


def measure_swatch_rect(store, tile, params):
    """Re-measure one explicit rectangle (used when approving a planned cut)."""
    views, _ = lamp_views(store, tile)
    require_both_lamps(views, "Swatch approval")
    extent = views[LAMP_ROLES[0]].frame.extent
    glare = glare_masks(views, params["glare_quantile"], params["glare_ratio"])
    region = active_region(tile, params["region_id"])
    paper_domain = region_domain(extent, tile, region)
    rect = rectangle_mask(extent, params["bounds"])
    per_lamp, common, summary = eligible_pixel_sets(rect, views, glare, paper_domain)
    if common.sum() < MIN_SWATCH_PIXELS or common.sum() / max(rect.sum(), 1) < MIN_COMMON_FRACTION:
        raise InputError(
            "The chosen rectangle no longer has enough shared, on-paper pixels in both lamps.",
        )
    return views, per_lamp, rect, summary, glare, region


def plan_overlay(store, tile, plan, *, candidate_index=0):
    """Draw the ranked cut rectangle over the standard-lamp plane."""
    view = lamp_views(store, tile)[0][LAMP_ROLES[0]]
    image = base_rgba(view.frame, view.linear).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line = max(2, round(min(width, height) / 260))
    for index, candidate in enumerate(plan["candidates"]):
        left, top, right, bottom = candidate["bounds"]
        box = (left * width, top * height, right * width, bottom * height)
        if index == candidate_index:
            draw.rectangle(box, outline=(120, 231, 185), width=line + 3)
            draw.rectangle(box, outline=(28, 42, 38), width=line)
        else:
            draw.rectangle(box, outline=(238, 184, 120), width=1)
    chosen = plan["candidates"][candidate_index]
    label = "cut {w} x {h} mm".format(
        w=chosen["size_mm"][0], h=chosen["size_mm"][1],
    )
    draw.text((6, 6), label, fill=(245, 245, 235), stroke_width=2, stroke_fill=(20, 26, 30))
    return image


def swatch_overlay_art(frame, linear, rect, masks):
    """Annotated swatch evidence image with excluded pixels hatched per lamp."""
    image = base_rgba(frame, linear).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Outline the requested rectangle, then mark which pixels each lamp accepted.
    ys, xs = np.nonzero(rect)
    if len(xs):
        draw.rectangle((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1),
                       outline=(245, 245, 235), width=2)
    colors = {
        LAMP_ROLES[0]: (120, 200, 231, 90),
        LAMP_ROLES[1]: (238, 184, 120, 90),
    }
    for role, mask in masks.items():
        yy, xx = np.nonzero(mask)
        for x, y in zip(xx[::3], yy[::3]):
            draw.point((x, y), fill=colors[role])
    return Image.alpha_composite(image, overlay)


def store_swatch_art(store, image):
    return stored_preview(image, store.blobs)


def _match_one_lamp(view, rect, glare, paper_domain, approved_lamp):
    frame = view.frame
    mask = rect & frame.valid & ~frame.clipped & paper_domain & ~glare[view.role]
    if int(mask.sum()) < MIN_SWATCH_PIXELS:
        return None, {
            "role": view.role,
            "available": False,
            "status": "insufficient-eligible-pixels",
            "note": "This lamp has too few on-paper, glare-free pixels in the region; no value is reported.",
            "measured_pixels": int(mask.sum()),
        }
    observed, median, _ = lamp_fingerprint(view, mask)
    approved_median = np.asarray(approved_lamp["fingerprint"]["median_lab"], dtype=np.float32)
    record = {
        "role": view.role,
        "available": True,
        "status": "measured",
        "photo_id": frame.photo["id"],
        "photo_name": frame.photo["name"],
        "measured_pixels": int(mask.sum()),
        "observed_lab": observed["median_lab"],
        "observed_swatch": observed["swatch"],
        "approved_lab": approved_lamp["fingerprint"]["median_lab"],
        "approved_swatch": approved_lamp["fingerprint"]["swatch"],
        "delta_e76": round(float(delta_e(median, approved_median)), 3),
        "delta_lab": float_list(median - approved_median),
        "dispersion_p90_delta_e": round(
            float(observed["dispersion"] - approved_lamp["fingerprint"]["dispersion"]), 3,
        ),
        "capture_note": (
            "Same illuminant and the same physical swatch rectangle as the approved reference."
        ),
    }
    return mask, record


def dual_illuminant_match(store, tile, region_id, lamp_photo_ids, reference, glare_params):
    """Match a new press sheet against one approved swatch, lamp by lamp.

    ``lamp_photo_ids`` maps each lamp role to the capture selected for it on the
    new sheet. A missing role yields an explicit ``not-captured`` entry instead
    of any substitute measurement.
    """
    if tile.get("plane") is None:
        raise InputError("Set a specimen plane before running a reference match.")
    illuminants = reference.get("illuminants")
    if not illuminants:
        return None
    # The approved rectangle is expressed in normalized plane coordinates, so it
    # can be applied to this sheet's own registered plane.
    rect = rectangle_mask(tile["plane"]["size"], reference["swatch"]["bounds"])
    region = active_region(tile, region_id)
    paper_domain = region_domain(tile["plane"]["size"], tile, region)
    selected = {}
    for role in LAMP_ROLES:
        photo_id = lamp_photo_ids.get(role)
        if not photo_id:
            continue
        photo = next((item for item in tile["photos"] if item["id"] == photo_id), None)
        if photo is None:
            raise InputError(f"The selected {role} capture is no longer on this specimen.")
        if photo["role"] != role:
            raise InputError(
                f"The {role} measurement must use a capture labeled {role}; "
                "another position or illuminant cannot be substituted.",
            )
        if photo["id"] not in {tile["plane"]["photo_id"], *tile["registrations"]}:
            raise InputError(f"Register the {role} capture before matching it.")
        frame = specimen_frame(store.blobs, tile, photo["id"])
        linear, _ = corrected_linear(tile, frame, True)
        selected[role] = LampView(role, frame, linear, linear_lab(linear))
    glare = glare_masks(selected, glare_params["glare_quantile"], glare_params["glare_ratio"])
    entries = []
    masks = {}
    for role in LAMP_ROLES:
        approved_lamp = illuminants.get(role)
        if approved_lamp is None:
            entries.append({
                "role": role,
                "available": False,
                "status": "not-in-reference",
                "note": "The approved reference has no fingerprint for this illuminant.",
            })
            continue
        view = selected.get(role)
        if view is None:
            entries.append({
                "role": role,
                "available": False,
                "status": "not-captured",
                "note": (
                    "No capture under this illuminant was supplied for the new sheet; "
                    "the reference value for the other lamp is not used as a substitute."
                ),
            })
            continue
        mask, record = _match_one_lamp(view, rect, glare, paper_domain, approved_lamp)
        if mask is None:
            entries.append(record)
        else:
            masks[role] = mask
            entries.append(record)
    measured = [entry for entry in entries if entry.get("available")]
    if not measured:
        raise InputError(
            "None of the supplied captures matches an illuminant held by the reference. "
            "Add the standard-lamp or warm-lamp capture for this sheet.",
        )
    worst = max((entry["delta_e76"] for entry in measured), default=None)
    return {
        "reference_title": reference["title"],
        "reference_id": reference["id"],
        "reference_image": reference["annotated_digest"],
        "reference_region": reference.get("region"),
        "swatch_bounds": reference["swatch"]["bounds"],
        "swatch_size_mm": reference["swatch"]["size_mm"],
        "illuminant_matches": entries,
        "measured_lamp_count": len(measured),
        "missing_lamp_count": len(entries) - len(measured),
        "worst_delta_e76": worst,
        "mean_delta_e76": round(
            float(np.mean([entry["delta_e76"] for entry in measured])), 3,
        ) if measured else None,
    }, masks, selected
