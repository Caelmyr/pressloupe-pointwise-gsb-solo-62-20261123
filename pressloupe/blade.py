"""Follow a printed control strip and repeat a stored width profile."""

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .art import base_rgba
from .color import linear_lab
from .coverage import reference_region, sample_chip
from .geometry import region_domain
from .model import InputError


INK_LEVEL = .52
MERGE_WIDTH_RATIO = 1.6
TOUCH_FRACTION = .5
STEP_LIMIT = 1.25
COMPARE_TOLERANCE_PX = 2.

SEEN = (159, 222, 187)
LEFT_MISSING = (245, 139, 93)
RIGHT_MISSING = (111, 160, 232)
MERGED = (188, 146, 214)
PREDICTED = (196, 178, 120)
ANCHOR = (252, 222, 107)
ORIGINAL = (242, 211, 138)
NOT_COMPARABLE = (167, 181, 174)

STATE_COLOURS = {
    "seen both edges": SEEN,
    "left edge not seen": LEFT_MISSING,
    "right edge not seen": RIGHT_MISSING,
    "crossing refused": MERGED,
    "centre predicted": PREDICTED,
}
SINGLE_LEGEND = {"mode": "categorical", "labels": [
    {"name": "Both edges seen", "color": "#9fdebb"},
    {"name": "Left edge not seen", "color": "#f58b5d"},
    {"name": "Right edge not seen", "color": "#6fa0e8"},
    {"name": "Crossing refused", "color": "#bc92d6"},
    {"name": "Centre carried", "color": "#c4b278"},
]}
REPEAT_LEGEND = {"mode": "categorical", "labels": [
    {"name": "Original cross-section", "color": "#f2d38a"},
    {"name": "Repeat, both edges", "color": "#9fdebb"},
    {"name": "Repeat, one edge", "color": "#f58b5d"},
    {"name": "Not comparable", "color": "#a7b5ae"},
]}


def scale_of(frame):
    return np.asarray(frame.scale or (1., 1.), dtype=float)


def extent_of(frame):
    return np.asarray(frame.extent, dtype=float)


def point_px(point, frame):
    return np.asarray(point, dtype=float) * extent_of(frame)


def normalised(point, frame):
    return np.round(np.asarray(point, dtype=float) / extent_of(frame), 6).tolist()


def strip_geometry(frame, settings):
    size = extent_of(frame) - 1.
    start = np.asarray(settings["start"], dtype=float) * size
    finish = np.asarray(settings["end"], dtype=float) * size
    scale = scale_of(frame)
    vector = finish - start
    length_px = float(np.linalg.norm(vector))
    if float(np.linalg.norm(vector * scale)) < 3 * float(scale.min()):
        raise InputError("Mark two points more than three pixels apart along the strip.")
    direction = vector / length_px
    normal = np.array([-direction[1], direction[0]])
    mm_per_px = float(np.hypot(normal[0] * scale[0], normal[1] * scale[1]))
    chord_mm = float(np.linalg.norm(vector * scale))
    return start, finish, direction, normal, length_px, mm_per_px, chord_mm


def width_offsets(frame, settings, mm_per_px):
    half = float(min(extent_of(frame) * scale_of(frame))) * settings["half_band"]
    count = min(450, max(2, int(np.ceil(half / max(mm_per_px, 1e-9)))))
    return np.linspace(-half, half, count * 2 + 1), half


def strip_projection(tile, frame, linear, settings):
    lab = linear_lab(linear)
    reference_domain = region_domain(frame.extent, tile)
    paper_region = reference_region(tile, "bare", settings["bare_region"])
    leaf_region = reference_region(tile, "glazed", settings["glazed_region"])
    paper, _, _ = sample_chip(frame, lab, reference_domain, paper_region)
    leaf, _, _ = sample_chip(frame, lab, reference_domain, leaf_region)
    delta = leaf - paper
    distance = float(np.linalg.norm(delta))
    if distance < 5:
        raise InputError("The unprinted paper and printed stock samples are too similar for a width reading.")
    shifted = lab - paper
    fraction = shifted @ delta / (distance ** 2)
    residual = shifted - fraction[..., None] * delta
    acceptable = np.linalg.norm(residual, axis=2) < distance * .46
    return fraction, acceptable


def sample_one(fraction, trusted, base, normal, offsets, mm_per_px):
    coords = (base[1] + normal[1] * offsets / mm_per_px, base[0] + normal[0] * offsets / mm_per_px)
    values = ndimage.map_coordinates(fraction, coords, order=1, mode="constant", cval=0.)
    support = ndimage.map_coordinates(
        trusted.astype(np.float32), coords, order=1, mode="constant", cval=0.,
    ) >= .9999
    return values, support


def ink_runs(values, support):
    """Contiguous supported runs at or above the ink level; a single sub-level sample splits a run."""
    ink = support & (values >= INK_LEVEL)
    transitions = np.diff(np.pad(ink.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1) - 1
    return [(int(first), int(last)) for first, last in zip(starts, stops)]

def section_candidate(run, support, offsets):
    first, last = run
    closed_left = first > 0 and bool(support[first - 1])
    closed_right = last < len(offsets) - 1 and bool(support[last + 1])
    left = float((offsets[first - 1] + offsets[first]) / 2.) if closed_left else None
    right = float((offsets[last] + offsets[last + 1]) / 2.) if closed_right else None
    return {
        "interval": (float(offsets[first]), float(offsets[last])),
        "left": left, "right": right,
        "closed_left": closed_left, "closed_right": closed_right,
        "width": (right - left) if closed_left and closed_right else None,
    }


def interval_gap(first, second):
    a, b = first
    c, d = second
    if b < c:
        return c - b
    if d < a:
        return a - d
    return 0.


def overlap(first, second):
    return min(first[1], second[1]) - max(first[0], second[0])


def predicted(prior, reason):
    middle = 0. if prior is None else (prior[0] + prior[1]) / 2.
    return {"state": "centre predicted", "side": "both", "width": None, "middle": middle,
            "left": None, "right": None, "interval": prior, "reason": reason}


def refused(reason):
    return {"state": "crossing refused", "side": None, "width": None, "middle": None,
            "left": None, "right": None, "interval": None, "reason": reason}


def choose_section(values, support, offsets, prior, width_hint):
    """Continue one strip run. A second branch that meets it is refused, never merged."""
    candidates = [section_candidate(run, support, offsets) for run in ink_runs(values, support)]
    if not candidates:
        return predicted(prior, "no printed ink reaches this cross-section")
    if prior is not None:
        ranked = sorted(
            candidates,
            key=lambda item: (-overlap(item["interval"], prior),
                              abs(sum(item["interval"]) / 2. - sum(prior) / 2.)),
        )
        chosen = ranked[0]
        if overlap(chosen["interval"], prior) <= 0:
            return predicted(prior, "no printed run continues the carried strip here")
    else:
        chosen = min(candidates, key=lambda item: abs(sum(item["interval"]) / 2.))
    for other in candidates:
        if other is chosen:
            continue
        span = min(chosen["interval"][1] - chosen["interval"][0],
                   other["interval"][1] - other["interval"][0])
        if interval_gap(chosen["interval"], other["interval"]) <= TOUCH_FRACTION * span:
            return refused("a second printed line meets this strip; the two branches are not one band")
    if chosen["width"] is not None and width_hint and chosen["width"] > MERGE_WIDTH_RATIO * width_hint:
        return refused("the printed ink is wider than one strip; a crossing is not measured as one band")
    if chosen["width"] is not None:
        return {"state": "seen both edges", "side": None, "width": chosen["width"],
                "middle": (chosen["left"] + chosen["right"]) / 2.,
                "left": chosen["left"], "right": chosen["right"],
                "interval": chosen["interval"], "reason": None}
    reference = width_hint if width_hint else chosen["width"]
    span = chosen["interval"][1] - chosen["interval"][0]
    if chosen["closed_left"]:
        middle = chosen["left"] + (reference or span) / 2.
        return {"state": "right edge not seen", "side": "right", "width": None,
                "middle": middle, "left": chosen["left"], "right": None,
                "interval": chosen["interval"],
                "reason": "the left edge is seen; the right edge leaves the sampled band"}
    if chosen["closed_right"]:
        middle = chosen["right"] - (reference or span) / 2.
        return {"state": "left edge not seen", "side": "left", "width": None,
                "middle": middle, "left": None, "right": chosen["right"],
                "interval": chosen["interval"],
                "reason": "the right edge is seen; the left edge leaves the sampled band"}
    return predicted(prior, "both strip edges leave the sampled band")


def walk_strip(fraction, trusted, chord, normals, offsets, mm_per_px):
    """Carry the strip centre section by section in physical units across the strip."""
    carried = 0.
    prior = None
    hint = None
    rows = []
    for index in range(len(chord)):
        anchor = carried
        base = chord[index] + normals[index] * (anchor / mm_per_px)
        values, support = sample_one(fraction, trusted, base, normals[index], offsets, mm_per_px)
        local = None if prior is None else (prior[0] - anchor, prior[1] - anchor)
        found = choose_section(values, support, offsets, local, hint)
        step = found["middle"]
        if step is not None and hint and abs(step) > STEP_LIMIT * hint:
            found = predicted(local, "the nearest printed ink is more than one strip away")
            step = found["middle"]
        if step is not None:
            carried += step
        if found["state"] == "seen both edges":
            prior = (found["left"] + anchor, found["right"] + anchor)
            hint = found["width"] if hint is None else .7 * hint + .3 * found["width"]
        elif found["state"] in ("left edge not seen", "right edge not seen") and hint:
            prior = (carried - hint / 2., carried + hint / 2.)
        rows.append({"base": base, "normal": normals[index], "found": found, "carried": carried})
    return rows


def local_normals(direction, normal, lateral, spacing_mm):
    """Square each section to the carried centre so a curving strip is measured across itself."""
    if len(lateral) < 3:
        return np.repeat(normal[None, :], len(lateral), axis=0)
    smooth = ndimage.gaussian_filter1d(np.asarray(lateral, dtype=float), 1.)
    slope = np.gradient(smooth, spacing_mm)
    tangents = direction[None, :] + normal[None, :] * slope[:, None]
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    return np.stack((-tangents[:, 1], tangents[:, 0]), axis=1)


def follow_stations(frame, fraction, trusted, settings):
    start, finish, direction, normal, length_px, mm_per_px, chord_mm = strip_geometry(frame, settings)
    offsets, half = width_offsets(frame, settings, mm_per_px)
    count = int(settings["stations"])
    positions = np.linspace(0., length_px, count)
    chord = start[None, :] + direction[None, :] * positions[:, None]
    spacing_mm = chord_mm / (count - 1)
    straight = np.repeat(normal[None, :], count, axis=0)
    first = walk_strip(fraction, trusted, chord, straight, offsets, mm_per_px)
    lateral = [row["carried"] for row in first]
    normals = local_normals(direction, normal, lateral, spacing_mm)
    traced = walk_strip(fraction, trusted, chord, normals, offsets, mm_per_px)
    rows, widths = [], []
    for index, item in enumerate(traced):
        found = item["found"]
        if found["state"] == "seen both edges":
            widths.append(found["width"])
        middle = found["middle"] if found["middle"] is not None else 0.
        centre = item["base"] + item["normal"] * (middle / mm_per_px)
        rows.append({
            "position": round(index * chord_mm / (count - 1), 3),
            "center": normalised(centre, frame),
            "normal": np.round(item["normal"], 8).tolist(),
            "millimetres_per_pixel": round(mm_per_px, 8),
            "state": found["state"],
            "side_missing": found["side"],
            "width": round(found["width"], 3) if found["width"] is not None else None,
            "left_edge": round(found["left"], 4) if found["left"] is not None else None,
            "right_edge": round(found["right"], 4) if found["right"] is not None else None,
            "left_point": normalised(item["base"] + item["normal"] * (found["left"] / mm_per_px), frame)
            if found["left"] is not None else None,
            "right_point": normalised(item["base"] + item["normal"] * (found["right"] / mm_per_px), frame)
            if found["right"] is not None else None,
            "center_mm_across": round(float(item["carried"]), 4),
            "reason": found["reason"],
        })
    geometry = (start, finish, normal, length_px, mm_per_px, chord_mm, offsets, half)
    return rows, widths, geometry


def predicted_stretches(rows):
    stretches, opening = [], None
    for index, row in enumerate(rows):
        weak = row["state"] != "seen both edges"
        if weak and opening is None:
            opening = index
        if not weak and opening is not None:
            stretches.append((opening, index - 1))
            opening = None
    if opening is not None:
        stretches.append((opening, len(rows) - 1))
    return stretches


def draw_edges(image, rows, frame, colour_for, connectors=True):
    draw = ImageDraw.Draw(image)
    previous = None
    for row in rows:
        colour = colour_for(row)
        left, right = row.get("left_point"), row.get("right_point")
        if left and right:
            draw.line([tuple(point_px(left, frame)), tuple(point_px(right, frame))], fill=colour, width=1)
        for point in (left, right):
            if point:
                x, y = point_px(point, frame)
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=colour)
        centre = point_px(row["center"], frame)
        if connectors and previous is not None:
            draw.line([tuple(previous), tuple(centre)], fill=NOT_COMPARABLE, width=1)
        previous = centre


def follow_width(frame, area, fraction, acceptable, settings):
    trusted = frame.valid & area & ~frame.clipped & acceptable
    rows, widths, geometry = follow_stations(frame, fraction, trusted, settings)
    start, finish, normal, length_px, mm_per_px, chord_mm, offsets, half = geometry
    if rows[0]["state"] != "seen both edges":
        raise InputError(
            "Mark the head of the strip on a cross-section where both edges are visible in this photograph. "
            "The centre can be carried through an obscured stretch, but it needs a seen starting edge."
        )
    states = [row["state"] for row in rows]
    left_seen = sum(row["side_missing"] != "left" for row in rows)
    right_seen = sum(row["side_missing"] != "right" for row in rows)
    tail_seen = rows[-1]["state"] == "seen both edges"
    caution = None if tail_seen else (
        "The carried centre did not reach a cross-section where both edges are visible, so the marked tail is not confirmed."
    )
    image = base_rgba(frame)
    draw_edges(image, rows, frame, lambda row: STATE_COLOURS.get(row["state"], PREDICTED))
    painter = ImageDraw.Draw(image)
    for point in (start, finish):
        x, y = point
        painter.ellipse((x - 4, y - 4, x + 4, y + 4), outline=ANCHOR, width=2)
    metrics = {
        "mode": "follow",
        "axis_length": round(chord_mm, 3),
        "length_unit": "mm" if frame.scale else "px",
        "requested_stations": len(rows),
        "observed_stations": len(widths),
        "both_edges_stations": states.count("seen both edges"),
        "predicted_stations": sum(state != "seen both edges" for state in states),
        "refused_stations": states.count("crossing refused"),
        "left_edge_seen_percent": round(100 * left_seen / len(rows), 2),
        "right_edge_seen_percent": round(100 * right_seen / len(rows), 2),
        "median_width": round(float(np.median(widths)), 3) if widths else None,
        "head": {"marked": np.round(settings["start"], 6).tolist(), "found": rows[0]["center"]},
        "tail": {"marked": np.round(settings["end"], 6).tolist(), "found": rows[-1]["center"],
                 "reached": tail_seen},
        "unseen_stretches": [
            {"first_station": first + 1, "last_station": last + 1,
             "first_position": rows[first]["position"], "last_position": rows[last]["position"],
             "stations": last - first + 1,
             "states": sorted({rows[index]["state"] for index in range(first, last + 1)})}
            for first, last in predicted_stretches(rows)
        ],
        "half_band": round(half, 4),
        "plane_size": list(frame.extent),
        "caution": caution,
        "stations": [{key: value for key, value in row.items() if key != "reason"} for row in rows],
        "note": (
            "Cross-sections follow the printed strip from the marked head toward the marked tail. A width is reported only "
            "where both strip edges are visible. Where one edge is lost the centre is carried and that side is named; a "
            "crossing is refused rather than measured as one wide band."
        ),
    }
    return metrics, image, SINGLE_LEGEND


def prior_summary(record):
    source = record["provenance"]["photo"]
    return {
        "id": record["id"],
        "created_at": record["created_at"],
        "source_version": record["source_version"],
        "photo": source["display_name"],
        "photo_role": source["role"],
        "photo_digest": source["original_digest"],
        "stations": len(record["metrics"].get("stations", [])),
    }


def stored_sections(prior, frame):
    metrics = prior["metrics"]
    if metrics.get("mode") != "follow":
        raise InputError("Choose an original strip measurement as the stored profile, not an earlier repeat.")
    if list(metrics.get("plane_size", [])) != list(frame.extent):
        raise InputError("The stored profile was measured on a different registered plane. Re-measure the strip first.")
    rows = [row for row in metrics.get("stations", [])
            if row.get("left_point") and row.get("right_point")
            and row.get("left_edge") is not None and row.get("right_edge") is not None
            and row.get("normal")]
    if not rows:
        raise InputError("The selected earlier measurement never saw both strip edges in one cross-section.")
    return rows


def compare_row(row, found, shifts, tolerance):
    left_shift, right_shift = shifts
    both = left_shift is not None and right_shift is not None
    width_change = round(right_shift - left_shift, 3) if both else None
    translation = round((left_shift + right_shift) / 2., 3) if both else None
    if both:
        if abs(width_change) <= tolerance:
            verdict = "shifted without width change" if abs(translation) > tolerance else "unchanged"
        else:
            verdict = "width changed"
    elif found["state"] in ("left edge not seen", "right edge not seen"):
        verdict = "one edge only"
    else:
        verdict = "not comparable"
    return {
        "position": row["position"],
        "old_left": round(float(row["left_edge"]), 3),
        "new_left": None if left_shift is None else round(float(row["left_edge"]) + left_shift, 3),
        "left_shift": left_shift,
        "old_right": round(float(row["right_edge"]), 3),
        "new_right": None if right_shift is None else round(float(row["right_edge"]) + right_shift, 3),
        "right_shift": right_shift,
        "old_width": row["width"],
        "new_width": round((float(row["right_edge"]) - float(row["left_edge"]))
                           + width_change, 3) if both else None,
        "width_change": width_change,
        "translation": translation,
        "state": found["state"],
        "verdict": verdict,
        "center": row["center"],
        "left_point": row.get("left_point"),
        "right_point": row.get("right_point"),
    }


def repeat_profile(frame, fraction, acceptable, settings, prior):
    stored = stored_sections(prior, frame)
    trusted = frame.valid & ~frame.clipped & acceptable
    results = []
    for row in stored:
        width = float(row["right_edge"]) - float(row["left_edge"])
        mm_per_px = float(row["millimetres_per_pixel"])
        margin = max(float(settings["half_band"]) * width, .6 * width, .05)
        reach = width + 2 * margin
        step = max(width / 60., 1e-4)
        offsets = np.linspace(-margin, width + margin, min(1600, max(17, int(np.ceil(reach / step)) + 1)))
        left_point = point_px(row["left_point"], frame)
        right_point = point_px(row["right_point"], frame)
        per_mm = (right_point - left_point) / width
        points = left_point[None, :] + per_mm[None, :] * offsets[:, None]
        coords = (points[:, 1], points[:, 0])
        values = ndimage.map_coordinates(fraction, coords, order=1, mode="constant", cval=0.)
        support = ndimage.map_coordinates(
            trusted.astype(np.float32), coords, order=1, mode="constant", cval=0.,
        ) >= .9999
        found = choose_section(values, support, offsets, (0., width), width)
        left_shift = round(found["left"], 3) if found["left"] is not None else None
        right_shift = round(found["right"] - width, 3) if found["right"] is not None else None
        result = compare_row(row, found, (left_shift, right_shift), COMPARE_TOLERANCE_PX * mm_per_px)
        if found["state"] == "seen both edges":
            result["left_point"] = normalised(left_point + per_mm * found["left"], frame)
            result["right_point"] = normalised(left_point + per_mm * found["right"], frame)
        result["reason"] = found["reason"]
        results.append(result)
    comparable = [item for item in results if item["width_change"] is not None]
    width_changes = [item["width_change"] for item in comparable]
    translations = [item["translation"] for item in comparable]
    image = base_rgba(frame)
    draw_edges(image, stored, frame, lambda row: ORIGINAL, connectors=False)
    draw_edges(image, results, frame,
               lambda row: SEEN if row["state"] == "seen both edges"
               else LEFT_MISSING if row["state"] in ("left edge not seen", "right edge not seen")
               else NOT_COMPARABLE, connectors=False)
    metrics = {
        "mode": "repeat",
        "length_unit": "mm" if frame.scale else "px",
        "prior": prior_summary(prior),
        "requested_stations": len(results),
        "comparable_stations": len(comparable),
        "one_edge_stations": sum(item["verdict"] == "one edge only" for item in results),
        "not_comparable_stations": sum(item["verdict"] == "not comparable" for item in results),
        "unchanged_stations": sum(item["verdict"] == "unchanged" for item in results),
        "shifted_without_width_change": sum(item["verdict"] == "shifted without width change" for item in results),
        "width_changed_stations": sum(item["verdict"] == "width changed" for item in results),
        "median_width_change": round(float(np.median(width_changes)), 3) if width_changes else None,
        "median_translation": round(float(np.median(translations)), 3) if translations else None,
        "stations": results,
        "note": (
            "Every cross-section is measured at the stored sheet position and along the stored across direction, so a whole "
            "strip shift appears as a shared translation of both edges and never as an unchanged width. Shifts are signed "
            "toward the stored right edge. Only edges seen in both photographs are compared; a lost edge is reported, not "
            "filled in."
        ),
    }
    return metrics, image, REPEAT_LEGEND


def blade_width(tile, frame, area, linear, settings, prior=None):
    fraction, acceptable = strip_projection(tile, frame, linear, settings)
    if prior is not None:
        return repeat_profile(frame, fraction, acceptable, settings, prior)
    return follow_width(frame, area, fraction, acceptable, settings)
