"""Attribution of detected ink to screen-lattice dot footprints.

A census snapshot keeps the binary ink evidence and the lattice site of every
ink pixel.  Reviewers can change how a connected ink group is read -- one
merged dot or the two real landing points behind a squeezed pair -- and every
row, summary and export is rebuilt from that saved evidence without adding
paper white or assigning one ink pixel twice.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .art import base_rgba, font
from .dots import CONNECTIVITY, MAX_SPLIT_SITES
from .model import InputError
from .photos import png_bytes


TEAL = (83, 230, 207)
AMBER = (253, 166, 63)
VIOLET = (176, 142, 230)
SHADOW = (30, 38, 43)
DISPOSITIONS = ("auto", "single", "split")
BOUNDARY_REASON = "crosses the supported analysis boundary"
OVERSIZED_REASON = "larger than the maximum footprint"
UNDERSIZED_REASON = "smaller than the minimum footprint"
MULTISITE_REASON = "ink joins more lattice sites than can be separated automatically"
SPRAWL_REASON = "connected ink spans too many lattice sites to read as separate dots"
NO_LATTICE_REASON = "no single screen placement fits the visible dot positions"


def footprint_reading(points, extent, scale):
    y, x = points
    xx = x.astype(float) + .5
    yy = y.astype(float) + .5
    center = np.array([xx.mean(), yy.mean()])
    offsets = np.column_stack((xx - center[0], yy - center[1]))
    if scale:
        offsets *= np.asarray(scale)
    eigenvalues, vectors = np.linalg.eigh(offsets.T @ offsets / len(x))
    minor, major = np.maximum(eigenvalues, 0)
    ratio = np.sqrt(major / minor) if minor > 0 else None
    axis = vectors[:, 1]
    angle = np.degrees(np.arctan2(axis[1], axis[0])) % 180
    return {
        "area_px": int(len(x)),
        "area_mm2": round(float(len(x) * scale[0] * scale[1]), 5) if scale else None,
        "axis_ratio": round(float(ratio), 3) if ratio is not None else None,
        "major_angle_deg": round(float(angle), 2) if ratio and ratio > 1.08 else None,
        "center": np.round(center / np.asarray(extent), 6).tolist(),
    }


def _site_pixel_mask(census, site_id):
    site_map = census["site_map"]
    return site_map == site_id


def _component_pixels(census, component_label):
    labels = census["component_labels"]
    return labels == component_label


def _group_pixels(census, group):
    return _component_pixels(census, group["component_label"])


def automatic_disposition(census, group, options, lattice_ok):
    """Decide how a connected ink group is read under the chosen placement."""
    area = group["pixels"]
    sites = group["site_ids"]
    if group["touches_boundary"]:
        return None, BOUNDARY_REASON
    if not lattice_ok or not sites:
        return None, NO_LATTICE_REASON
    if len(sites) == 1:
        if area < options["minimum_area"]:
            return None, UNDERSIZED_REASON
        if area > options["maximum_area"]:
            return None, OVERSIZED_REASON
        return "single", None
    if len(sites) > MAX_SPLIT_SITES:
        return None, SPRAWL_REASON
    fragments = [
        int(np.count_nonzero(_site_pixel_mask(census, site_id) & _group_pixels(census, group)))
        for site_id in sites
    ]
    if any(fragment < options["minimum_area"] for fragment in fragments):
        return None, MULTISITE_REASON
    if area > options["maximum_area"] * len(sites):
        return None, OVERSIZED_REASON
    return "split", None


def plan_groups(census, options):
    lattice_ok = census["lattice"] is not None
    planned = []
    for group in census["groups"]:
        disposition, reason = automatic_disposition(census, group, options, lattice_ok)
        planned.append({
            **group,
            "site_count": len(group["site_ids"]),
            "automatic": disposition,
            "reason": reason,
            "reviewed_disposition": None,
        })
    return planned


def validate_choice(group, choice):
    if choice not in DISPOSITIONS:
        raise InputError("A dot group can stay automatic, be read as one dot or split.")
    if choice == "auto":
        return
    if choice == "split":
        if group["site_count"] < 2:
            raise InputError("This ink group covers one lattice site and cannot be split.")
        if group["site_count"] > MAX_SPLIT_SITES:
            raise InputError("This connected ink spans too many lattice sites to read as separate dots.")
    if choice == "single" and group["touches_boundary"]:
        raise InputError("Boundary-crossing ink cannot be counted: its full footprint is missing.")


def chosen_disposition(group):
    override = group["reviewed_disposition"]
    if override and override != "auto":
        return override
    return group["automatic"]


def _dot_rows(census, groups, frame):
    """Build footprint rows from the saved pixel evidence.

    Every counted ink pixel belongs to exactly one row; paper white never
    enters a footprint because rows are sampled from the binary ink mask.
    """
    rows = []
    site_map = census["site_map"]
    labels = census["component_labels"]
    for group in groups:
        disposition = chosen_disposition(group)
        if disposition is None:
            continue
        component_mask = labels == group["component_label"]
        if disposition == "single":
            y, x = np.nonzero(component_mask)
            reading = footprint_reading((y, x), frame.extent, frame.scale)
            rows.append({
                "group": group["group"],
                "site": None,
                "origin": "manual" if group["reviewed_disposition"] == "single" else "auto",
                **reading,
            })
            continue
        for site_id in group["site_ids"]:
            fragment = component_mask & (site_map == site_id)
            if not fragment.any():
                continue
            y, x = np.nonzero(fragment)
            reading = footprint_reading((y, x), frame.extent, frame.scale)
            rows.append({
                "group": group["group"],
                "site": census["site_coordinates"][site_id],
                "origin": "manual" if group["reviewed_disposition"] == "split" else "auto",
                **reading,
            })
    rows.sort(key=lambda row: (row["center"][1], row["center"][0]))
    for number, row in enumerate(rows, start=1):
        row["dot"] = number
    return rows


def _resolved_mask(census, groups):
    labels = census["component_labels"]
    site_map = census["site_map"]
    mask = np.zeros(labels.shape, dtype=bool)
    for group in groups:
        disposition = chosen_disposition(group)
        if disposition is None:
            continue
        component_mask = labels == group["component_label"]
        if disposition == "single":
            mask |= component_mask
        else:
            mask |= component_mask & np.isin(site_map, group["site_ids"])
    return mask


def render_overlay(frame, census, groups, rows):
    image = base_rgba(frame)
    picture = np.asarray(image).copy()
    resolved = _resolved_mask(census, groups)
    unresolved = census["ink"] & ~resolved
    outline = resolved & ~ndimage.binary_erosion(resolved, structure=CONNECTIVITY)
    picture[outline, :3] = TEAL
    picture[unresolved, :3] = (
        .65 * picture[unresolved, :3] + .35 * np.array(AMBER)
    ).astype(np.uint8)
    image = Image.fromarray(picture)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line = max(1, round(min(width, height) / 360))
    typeface = font(max(10, min(17, round(min(width, height) / 46))))
    for group in groups:
        if chosen_disposition(group) is not None:
            continue
        left, top, right, bottom = group["bounds"]
        draw.rectangle((left, top, right - 1, bottom - 1), outline=SHADOW, width=line + 2)
        draw.rectangle((left, top, right - 1, bottom - 1), outline=AMBER, width=line)
        if len(groups) <= 220:
            draw.text((left, max(0, top - 13)), group["group"], fill=AMBER,
                      font=typeface, stroke_width=2, stroke_fill=SHADOW)
    # Mark the two real landing points of manually separated squeezed pairs.
    for row in rows:
        if row["origin"] != "manual" or row["site"] is None:
            continue
        x, y = row["center"]
        cx, cy = x * width, y * height
        radius = max(3, round(min(width, height) / 150))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                     outline=SHADOW, width=line + 1)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                     outline=VIOLET, width=line)
    return image


def summarize(census, groups, rows, options):
    ink_pixels = int(np.count_nonzero(census["ink"]))
    resolved_pixels = sum(row["area_px"] for row in rows)
    unresolved_pixels = ink_pixels - resolved_pixels
    denominator = int(np.count_nonzero(census["valid"]))
    unresolved_groups = [group for group in groups if chosen_disposition(group) is None]
    manual_groups = sorted({
        row["group"] for row in rows if row["origin"] == "manual"
    })
    areas = [row["area_px"] for row in rows]
    lattice = census["lattice"]
    return {
        "dot_count": len(rows),
        "median_area_px": round(float(np.median(areas)), 2) if areas else None,
        "ink_area_percent": round(100 * resolved_pixels / denominator, 3),
        "unresolved_ink_percent": round(100 * unresolved_pixels / denominator, 3),
        "supported_pixels": denominator,
        "ink_pixels": ink_pixels,
        "resolved_ink_pixels": resolved_pixels,
        "unresolved_ink_pixels": unresolved_pixels,
        "unresolved_group_count": len(unresolved_groups),
        "manually_attributed_groups": manual_groups,
        "lattice_fit": lattice,
        "footprint_bounds_px2": [options["minimum_area"], options["maximum_area"]],
    }


def unresolved_listing(groups):
    items = []
    for group in groups:
        if chosen_disposition(group) is not None:
            continue
        items.append({
            "group": group["group"],
            "pixels": group["pixels"],
            "site_count": group["site_count"],
            "touches_boundary": group["touches_boundary"],
            "automatic": group["automatic"],
            "reason": group["reason"],
            "available_actions": available_actions(group),
        })
    items.sort(key=lambda item: (-item["pixels"], item["group"]))
    return items


def available_actions(group):
    actions = ["auto"]
    if not group["touches_boundary"]:
        actions.append("single")
    if 2 <= group["site_count"] <= MAX_SPLIT_SITES:
        actions.append("split")
    return actions


def note_text(census, groups):
    lattice = census["lattice"]
    if lattice is None:
        lead = ("No single screen placement fits the visible dots. All ink stays "
                "amber and unresolved until the thresholds expose clear isolated dots.")
    elif lattice["ambiguous_placement"]:
        lead = ("Two screen placements explain the photographed pinhole positions almost "
                "equally well; the lower-residual placement is in use and is recorded below.")
    else:
        lead = "Ink is assigned to the fitted screen lattice placement, one ink pixel to one footprint."
    pending = sum(1 for group in groups if chosen_disposition(group) is None)
    tail = (f" {pending} ink group(s) remain amber and unresolved." if pending else
            " Every detected ink pixel is attributed to a counted footprint.")
    return (lead + " Paper white is never added and no ink pixel is counted twice." + tail)


def encode_maps(census):
    """Persist the exact pixel evidence as compact PNG label maps."""
    ink_png = png_bytes(Image.fromarray((census["ink"] * 255).astype(np.uint8), mode="L"))
    valid_png = png_bytes(Image.fromarray((census["valid"] * 255).astype(np.uint8), mode="L"))
    site_png = png_bytes(_rg_png(census["site_map"].astype(np.int32)))
    return ink_png, valid_png, site_png


def _rg_png(values):
    # Positive site ids fit in 31 bits; R/G carry the low/high 16 bits.
    low = (values & 0xFFFF).astype(np.uint8)
    high = ((values >> 16) & 0xFF).astype(np.uint8)
    blue = np.zeros(values.shape, dtype=np.uint8)
    return Image.fromarray(np.dstack((low, high, blue)), mode="RGB")


def decode_maps(blobs, plan, shape):
    ink_image = Image.open(blobs.require(plan["ink_map_digest"]))
    ink = np.asarray(ink_image, dtype=np.uint8) > 0
    valid = np.asarray(Image.open(blobs.require(plan["valid_map_digest"])), dtype=np.uint8) > 0
    site_image = Image.open(blobs.require(plan["site_map_digest"])).convert("RGB")
    channels = np.asarray(site_image, dtype=np.int32)
    site_map = channels[..., 0] | (channels[..., 1] << 16)
    if ink.shape != shape or site_map.shape != shape or valid.shape != shape:
        raise InputError("The saved census maps no longer match the source image size.")
    if np.any(site_map[~ink] != 0):
        raise InputError("The saved census assigns a paper-white pixel to a dot.")
    if np.any(ink & ~valid):
        raise InputError("The saved census places ink outside the supported paper.")
    maximum_id = int(site_map.max(initial=0))
    if maximum_id >= len(plan["site_coordinates"]):
        raise InputError("The saved census references a lattice site that is not in the plan.")
    return ink, valid, site_map


def reconstruct_census(ink, valid, site_map, plan):
    """Rebuild connected ink groups from the saved binary evidence."""
    labels, count = ndimage.label(ink, structure=CONNECTIVITY)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    objects = ndimage.find_objects(labels)
    support_edge = valid & ~ndimage.binary_erosion(valid, structure=CONNECTIVITY)
    touching = set(np.unique(labels[support_edge]))
    groups = []
    for index, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        y, x = np.nonzero(labels[bounds] == index)
        y += bounds[0].start
        x += bounds[1].start
        site_ids = [int(value) for value in np.unique(site_map[y, x]) if value > 0]
        groups.append({
            "group": f"C{index:04d}",
            "component_label": int(index),
            "pixels": int(sizes[index]),
            "bounds": [int(bounds[1].start), int(bounds[0].start),
                       int(bounds[1].stop), int(bounds[0].stop)],
            "touches_boundary": index in touching,
            "site_ids": site_ids,
            "site_count": len(site_ids),
            "automatic": None,
            "reason": None,
            "reviewed_disposition": None,
        })
    return {
        "ink": ink,
        "valid": valid,
        "site_map": site_map,
        "component_labels": labels,
        "site_coordinates": plan["site_coordinates"],
        "lattice": plan["lattice_fit"],
        "lattice_spans": plan["lattice_spans"],
        "groups": groups,
    }


def apply_review_choices(groups, choices):
    """Overlay saved manual dispositions, keyed by the stable group id."""
    lookup = {group["group"]: group for group in groups}
    for key, choice in choices.items():
        group = lookup.get(key)
        if group is None:
            raise InputError(f"Dot group {key} is no longer part of this census.")
        validate_choice(group, choice)
        group["reviewed_disposition"] = None if choice == "auto" else choice
    return groups


def reflow_group_metadata(census, groups, options):
    """Recompute the automatic disposition and reason for every group."""
    lattice_ok = census["lattice"] is not None
    for group in groups:
        disposition, reason = automatic_disposition(census, group, options, lattice_ok)
        group["automatic"] = disposition
        group["reason"] = reason
    return groups
