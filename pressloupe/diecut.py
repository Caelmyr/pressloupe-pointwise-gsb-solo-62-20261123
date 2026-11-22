"""Die layout: place one mounted die (outer contour plus shaped windows) on a
two-sided sheet, compare the three press shift positions against the printed
strokes, and classify traced fold lines millimetre by millimetre."""

import math
from itertools import combinations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .art import INK, SHADOW, base_rgba, curve_points, stored_preview
from .dots import local_ink_contrast
from .geometry import polygon_mask, segments_intersect, signed_area, specimen_frame
from .model import InputError, collection, dictionary, enumeration, identity, numeric_pair, scalar, string, timestamp
from .photos import uint8_image


POSITIONS = (-1, 0, 1)
POSITION_LABELS = {-1: "Left", 0: "Center", 1: "Right"}
FLIP_LABELS = {
    "horizontal": "Left-right turn (vertical axis)",
    "vertical": "Top-bottom turn (horizontal axis)",
}
STATE_ORDER = ("product", "window", "waste", "unphotographed")
STATE_COLORS = {
    "product": (111, 196, 152),
    "window": (246, 139, 96),
    "waste": (238, 98, 68),
    "unphotographed": (176, 176, 176),
}
KEPT_INK = np.array([83, 230, 207], dtype=np.float32) / 255
LOST_INK = np.array([238, 98, 68], dtype=np.float32) / 255
MAX_WINDOWS = 4
MAX_TRACE_BUCKETS = 800


# ---------------------------------------------------------------- validation

def die_polygon(value, label, minimum, maximum):
    raw = collection(value, label, minimum=minimum, maximum=maximum)
    return [numeric_pair(point, label, lower=-0.25, upper=1.25) for point in raw]


def layout_fields(payload):
    data = dictionary(payload, "Die layout")
    return {
        "name": string(data.get("name"), "Layout name", maximum=90),
        "back_tile_id": identity(data.get("back_tile_id"), "Back specimen ID"),
        "flip": enumeration(data.get("flip"), FLIP_LABELS, "Flip direction"),
        "step_mm": scalar(data.get("step_mm"), "Press shift step", lower=0.1, upper=10),
        "contrast": scalar(data.get("contrast", 0.16), "Stroke contrast", lower=0.05, upper=0.6),
        "outer": die_polygon(data.get("outer"), "Outer contour", 3, 48),
        "windows": [
            die_polygon(window, "Window", 3, 40)
            for window in collection(data.get("windows"), "Windows", minimum=1, maximum=MAX_WINDOWS)
        ],
        "front_region_id": identity(data["front_region_id"], "Front region ID") if data.get("front_region_id") else None,
        "back_region_id": identity(data["back_region_id"], "Back region ID") if data.get("back_region_id") else None,
    }


def simple_polygon(vertices, label, min_area):
    if abs(signed_area(vertices)) < min_area:
        raise InputError(f"{label} needs an area of at least {min_area:g} mm².")
    if len({tuple(point) for point in vertices}) != len(vertices):
        raise InputError(f"{label} cannot use the same vertex more than once.")
    size = len(vertices)
    for index in range(size):
        a, b = vertices[index], vertices[(index + 1) % size]
        for other in range(index + 2, size):
            if index == 0 and other == size - 1:
                continue
            c, d = vertices[other], vertices[(other + 1) % size]
            if segments_intersect(a, b, c, d):
                raise InputError(f"{label} edges cannot cross.")


def point_inside(vertices, point):
    inside = False
    x, y = point
    for a, b in zip(vertices, [*vertices[1:], vertices[0]]):
        (ax, ay), (bx, by) = a, b
        if (ay > y) != (by > y):
            cross = (bx - ax) * (y - ay) / (by - ay) + ax
            if x < cross:
                inside = not inside
    return inside


def polygon_edges(vertices):
    return list(zip(vertices, [*vertices[1:], vertices[0]]))


def validate_die(outer, windows):
    simple_polygon(outer, "The outer contour", 25)
    for window in windows:
        simple_polygon(window, "Each window", 2)
        if not all(point_inside(outer, point) for point in window):
            raise InputError("Each window must sit inside the outer contour.")
        if any(
            segments_intersect(a, b, c, d)
            for a, b in polygon_edges(window)
            for c, d in polygon_edges(outer)
        ):
            raise InputError("Window edges cannot cross the outer contour.")
    for first, second in combinations(windows, 2):
        if point_inside(first, second[0]) or point_inside(second, first[0]) or any(
            segments_intersect(a, b, c, d)
            for a, b in polygon_edges(first)
            for c, d in polygon_edges(second)
        ):
            raise InputError("Windows cannot overlap each other.")


# -------------------------------------------------------------- coordinates

def sheet_to_plane(points_mm, sheet_mm, side, flip):
    """Map sheet millimetres (front view) to normalized plane coordinates."""
    width, height = sheet_mm
    points = np.asarray(points_mm, dtype=np.float64).copy()
    if side == "back":
        if flip == "horizontal":
            points[:, 0] = width - points[:, 0]
        else:
            points[:, 1] = height - points[:, 1]
    return points / [width, height]


def die_outlines(layout, position, side):
    shift = position * layout["step_mm"]
    outer = [[x + shift, y] for x, y in layout["outer_mm"]]
    windows = [[[x + shift, y] for x, y in window] for window in layout["windows_mm"]]
    return (
        sheet_to_plane(outer, layout["sheet_mm"], side, layout["flip"]),
        [sheet_to_plane(window, layout["sheet_mm"], side, layout["flip"]) for window in windows],
    )


def die_masks(extent, layout, position, side):
    outer_norm, windows_norm = die_outlines(layout, position, side)
    outer_mask = polygon_mask(extent, outer_norm)
    window_mask = np.zeros(outer_mask.shape, dtype=bool)
    for window in windows_norm:
        window_mask |= polygon_mask(extent, window)
    return outer_mask, window_mask


def layout_side(layout, tile_id):
    if layout["tile_id"] == tile_id:
        return "front"
    if layout["back_tile_id"] == tile_id:
        return "back"
    raise InputError("This specimen is not part of the chosen die layout.")


def require_sheet_mm(tile, label):
    if tile.get("width_mm") is None or tile.get("height_mm") is None:
        raise InputError(f"Enter the sheet dimensions (mm) for the {label} specimen first.")
    if not tile.get("plane"):
        raise InputError(f"Register the photographed plane of the {label} specimen first.")
    return [tile["width_mm"], tile["height_mm"]]


# --------------------------------------------------------------- evaluation

def ink_components(frame, support, contrast):
    px_per_mm = 1.0 / float(np.mean(frame.scale))
    radius = int(np.clip(round(1.6 * px_per_mm), 3, 28))
    field, background_valid = local_ink_contrast(frame.linear, support, radius)
    ink = background_valid & (field >= contrast)
    labels, count = ndimage.label(ink, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    kept = np.zeros(count + 1, dtype=bool)
    kept[1:] = sizes[1:] >= max(3, int(round(0.35 * px_per_mm)))
    return labels, sizes, kept, radius


def position_metrics(labels, sizes, kept, outer_mask, window_mask):
    ids = np.flatnonzero(kept)
    ids = ids[ids > 0]
    ink_mask = kept[labels]
    total = int(sizes[ids].sum())
    in_window = ink_mask & window_mask
    in_waste = ink_mask & ~outer_mask & ~window_mask
    lost_per = np.bincount(labels[in_window | in_waste], minlength=len(sizes))
    broken = ids[(lost_per[ids] >= 2) & (lost_per[ids] >= 0.04 * sizes[ids])]
    lost_window = int(np.count_nonzero(in_window))
    lost_outer = int(np.count_nonzero(in_waste))
    lost = lost_window + lost_outer
    return {
        "ink_pixels": total,
        "lost_pixels": lost,
        "lost_to_window_px": lost_window,
        "lost_to_outer_px": lost_outer,
        "kept_percent": round(100 * (total - lost) / total, 2) if total else None,
        "lost_percent": round(100 * lost / total, 2) if total else None,
        "broken_components": int(len(broken)),
        "component_count": int(len(ids)),
    }


def position_overlay(frame, region, ink_mask, lost_mask, review_mask, outer_norm, windows_norm, caption):
    base = frame.display() * 0.52
    kept = ink_mask & ~lost_mask
    base[kept] = 0.30 * base[kept] + 0.70 * KEPT_INK
    base[lost_mask] = 0.25 * base[lost_mask] + 0.75 * LOST_INK
    if np.any(review_mask):
        rows, cols = np.ogrid[: review_mask.shape[0], : review_mask.shape[1]]
        stripe = ((rows + cols) % 12 < 4) & review_mask
        base[review_mask] = 0.20
        base[stripe] = 0.52
    image = uint8_image(base, frame.valid.astype(np.float32)).convert("RGBA")
    draw = ImageDraw.Draw(image)
    stroke = max(1, round(min(image.size) / 320))
    outlines = [(outer_norm, INK)] + [(window, (246, 139, 96)) for window in windows_norm]
    if region is not None:
        outlines.append((region["vertices"], (199, 212, 151)))
    for vertices, color in outlines:
        points = curve_points(vertices, image.size)
        draw.line([*points, points[0]], fill=SHADOW, width=stroke + 2, joint="curve")
        draw.line([*points, points[0]], fill=color, width=stroke, joint="curve")
    draw.text((10, 8), caption, fill=INK, stroke_width=2, stroke_fill=SHADOW)
    return image


def evaluate_side(store, layout, tile, region_id, side):
    frame = specimen_frame(store.blobs, tile)
    region = None
    if region_id:
        region = next((item for item in tile["regions"] if item["id"] == region_id), None)
        if region is None:
            raise InputError(f"The {side} region was removed from its specimen.")
        if region["purpose"] == "ignore":
            raise InputError(f"The {side} region cannot be an exclusion region.")
    extent = frame.extent
    region_mask = (
        polygon_mask(extent, region["vertices"])
        if region
        else np.ones((extent[1], extent[0]), dtype=bool)
    )
    support = region_mask & frame.valid & ~frame.clipped
    if np.count_nonzero(support) < 144:
        raise InputError(f"The {side} side has too little photographed print inside its region.")
    labels, sizes, kept, radius = ink_components(frame, support, layout["contrast"])
    ink_mask = kept[labels]
    if np.count_nonzero(ink_mask) < 8:
        raise InputError(
            f"No printed strokes were found on the {side} side inside its region. "
            "Widen the region or lower the stroke contrast."
        )
    review_mask = region_mask & ~frame.valid
    # Strokes within roughly one background radius of missing paper may continue
    # into the unphotographed area; flag them for review instead of judging them.
    if np.any(review_mask):
        distance, nearest = ndimage.distance_transform_edt(~ink_mask, return_indices=True)
        close = review_mask & (distance <= radius + 3)
        uncertain = int(len(set(np.unique(labels[tuple(nearest[:, close])]).tolist()) - {0}))
    else:
        uncertain = 0
    per_position = {}
    digests = {}
    for position in POSITIONS:
        outer_mask, window_mask = die_masks(extent, layout, position, side)
        per_position[str(position)] = position_metrics(labels, sizes, kept, outer_mask, window_mask)
        lost_mask = ink_mask & (window_mask | ~outer_mask)
        outer_norm, windows_norm = die_outlines(layout, position, side)
        shift = position * layout["step_mm"]
        caption = f"{POSITION_LABELS[position]} · die {shift:+.2f} mm · {side}"
        image = position_overlay(
            frame, region, ink_mask, lost_mask, review_mask, outer_norm, windows_norm, caption,
        )
        digests[str(position)] = stored_preview(image, store.blobs)
    region_pixels = int(np.count_nonzero(region_mask))
    review_pixels = int(np.count_nonzero(review_mask))
    summary = {
        "tile_id": tile["id"],
        "tile_name": tile["name"],
        "region": (
            {"id": region["id"], "name": region["name"], "vertices": region["vertices"]}
            if region else None
        ),
        "region_pixels": region_pixels,
        "review_pixels": review_pixels,
        "review_percent": round(100 * review_pixels / region_pixels, 2) if region_pixels else 0,
        "ink_pixels": int(sizes[np.flatnonzero(kept)].sum()),
        "component_count": int(np.count_nonzero(kept)),
        "uncertain_components": uncertain,
        "per_position": per_position,
    }
    return summary, digests


def recommend_position(sides):
    front = sides["front"]["per_position"]
    back = sides["back"]["per_position"]

    def score(position):
        kept = front[str(position)]["kept_percent"]
        return (kept, -back[str(position)]["broken_components"], -back[str(position)]["lost_pixels"], -abs(position))

    return max(POSITIONS, key=score)


def evaluate_layout(store, layout, front, back):
    sides = {}
    overlays = {}
    for side, tile, region_id in (
        ("front", front, layout["front_region_id"]),
        ("back", back, layout["back_region_id"]),
    ):
        summary, digests = evaluate_side(store, layout, tile, region_id, side)
        sides[side] = summary
        overlays[side] = digests
    notes = [
        "Teal ink stays on the product; red ink falls into the window or the outer waste at the shown position.",
        "The whole die shifts as one piece: the outer contour and the window move together.",
    ]
    if any(sides[side]["review_pixels"] for side in sides):
        notes.append("Hatched paper was not photographed clearly. Review those areas on the proof before committing.")
    return {
        "computed_at": timestamp(),
        "positions": list(POSITIONS),
        "position_labels": {str(key): label for key, label in POSITION_LABELS.items()},
        "recommended": recommend_position(sides),
        "sides": sides,
        "overlays": overlays,
        "notes": notes,
    }


def create_layout(store, specimen_id, payload):
    fields = layout_fields(payload)
    front = store.tile(specimen_id)
    sheet_mm = require_sheet_mm(front, "front")
    back = store.tile(fields["back_tile_id"])
    back_mm = require_sheet_mm(back, "back")
    if front["id"] == back["id"]:
        raise InputError("Choose a different specimen for the back side.")
    if abs(sheet_mm[0] - back_mm[0]) > 0.75 or abs(sheet_mm[1] - back_mm[1]) > 0.75:
        raise InputError("Front and back sheet dimensions differ by more than 0.75 mm.")
    outer_mm = [[u * sheet_mm[0], v * sheet_mm[1]] for u, v in fields["outer"]]
    windows_mm = [[[u * sheet_mm[0], v * sheet_mm[1]] for u, v in window] for window in fields["windows"]]
    validate_die(outer_mm, windows_mm)
    layout = {
        "name": fields["name"],
        "tile_id": front["id"],
        "back_tile_id": back["id"],
        "flip": fields["flip"],
        "step_mm": fields["step_mm"],
        "contrast": fields["contrast"],
        "sheet_mm": sheet_mm,
        "outer_mm": outer_mm,
        "windows_mm": windows_mm,
        "front_region_id": fields["front_region_id"],
        "back_region_id": fields["back_region_id"],
    }
    layout["evaluation"] = evaluate_layout(store, layout, front, back)
    return store.save_layout(layout)


# ----------------------------------------------------------- fold-line walk

def walk_trace(frame, outer_mask, window_mask, points):
    extent = np.asarray(frame.extent, dtype=np.float64)
    xy = np.asarray(points, dtype=np.float64) * extent - 0.5
    scale = np.asarray(frame.scale, dtype=np.float64)
    height, width = outer_mask.shape
    arcs = [0.0]
    states = []
    coords = [xy[0]]
    arc = 0.0
    for start, stop in zip(xy, xy[1:]):
        delta = stop - start
        length_px = float(np.linalg.norm(delta))
        count = min(1800, max(2, int(math.ceil(length_px)) + 1))
        samples = np.linspace(start, stop, count)[1:]
        seg_mm = float(np.linalg.norm(delta * scale))
        rows = np.clip(samples[:, 1], 0, height - 1)
        cols = np.clip(samples[:, 0], 0, width - 1)
        valid = ndimage.map_coordinates(frame.valid.astype(np.float32), (rows, cols), order=1, mode="nearest") >= 0.995
        inside_window = ndimage.map_coordinates(window_mask.astype(np.float32), (rows, cols), order=1, mode="nearest") >= 0.5
        inside_outer = ndimage.map_coordinates(outer_mask.astype(np.float32), (rows, cols), order=1, mode="nearest") >= 0.5
        state = np.where(~valid, 3, np.where(inside_window, 1, np.where(~inside_outer, 2, 0)))
        step = seg_mm / (count - 1)
        for index in range(count - 1):
            arc += step
            arcs.append(arc)
            coords.append(samples[index])
        states.extend(state.tolist())
    return np.asarray(arcs), np.asarray(states, dtype=np.int8), np.asarray(coords) / extent, arc


def bucket_trace(arcs, states, coords, total_mm):
    bucket_mm = 1.0
    if total_mm > MAX_TRACE_BUCKETS:
        bucket_mm = float(math.ceil(total_mm / MAX_TRACE_BUCKETS))
    count = max(1, int(math.ceil(total_mm / bucket_mm)))
    lengths = np.zeros((count, 4))
    intervals = np.diff(arcs)
    owners = np.minimum((arcs[:-1] / bucket_mm).astype(int), count - 1)
    for state in range(4):
        lengths[:, state] = np.bincount(owners, weights=intervals * (states == state), minlength=count)

    def locate(position):
        return [
            round(float(np.interp(position, arcs, coords[:, 0])), 5),
            round(float(np.interp(position, arcs, coords[:, 1])), 5),
        ]

    rows = []
    for index in range(count):
        start_arc = index * bucket_mm
        stop_arc = min((index + 1) * bucket_mm, total_mm)
        state = int(np.argmax(lengths[index])) if lengths[index].sum() > 0 else 3
        rows.append({
            "index": index + 1,
            "from_mm": round(start_arc, 3),
            "to_mm": round(stop_arc, 3),
            "state": STATE_ORDER[state],
            "states": {
                name: round(float(lengths[index][slot]), 3)
                for slot, name in enumerate(STATE_ORDER)
            },
            "start_xy": locate(start_arc),
            "end_xy": locate(stop_arc),
        })
    totals = {
        name: round(float(lengths[:, slot].sum()), 3)
        for slot, name in enumerate(STATE_ORDER)
    }
    return rows, totals, bucket_mm


def state_polylines(states, coords_px):
    lines = []
    run_state = int(states[0])
    run = [tuple(coords_px[0])]
    for index, state in enumerate(states):
        if state != run_state:
            run.append(tuple(coords_px[index]))
            lines.append((run_state, run))
            run = [tuple(coords_px[index])]
            run_state = int(state)
        run.append(tuple(coords_px[index + 1]))
    lines.append((run_state, run))
    return lines


def decimate(points, limit=400):
    if len(points) <= limit:
        return points
    stride = math.ceil(len(points) / limit)
    return [*points[::stride], points[-1]]


def trace_overlay(frame, layout, position, side, arcs, states, coords, bucket_count, bucket_mm):
    outer_norm, windows_norm = die_outlines(layout, position, side)
    image = base_rgba(frame)
    dimmed = np.asarray(image).copy()
    dimmed[:, :, :3] = (dimmed[:, :, :3].astype(np.float32) * 0.55).astype(np.uint8)
    image = Image.fromarray(dimmed)
    draw = ImageDraw.Draw(image)
    stroke = max(1, round(min(image.size) / 320))
    for vertices, color in [(outer_norm, INK), *[(window, (246, 139, 96)) for window in windows_norm]]:
        points = curve_points(vertices, image.size)
        draw.line([*points, points[0]], fill=SHADOW, width=stroke + 2, joint="curve")
        draw.line([*points, points[0]], fill=color, width=stroke, joint="curve")
    extent = np.asarray(frame.extent, dtype=np.float64)
    coords_px = coords * extent
    lines = state_polylines(states, coords_px)
    for _, run in lines:
        draw.line(decimate(run), fill=SHADOW, width=stroke + 3, joint="curve")
    for state, run in lines:
        draw.line(decimate(run), fill=STATE_COLORS[STATE_ORDER[state]], width=stroke + 1, joint="curve")
    total_mm = float(arcs[-1])
    every = max(1, math.ceil(bucket_count / 24))
    for index in range(bucket_count):
        if index % every:
            continue
        position = min((index + 0.5) * bucket_mm, total_mm)
        x = float(np.interp(position, arcs, coords_px[:, 0]))
        y = float(np.interp(position, arcs, coords_px[:, 1]))
        draw.ellipse((x - 2.5, y - 2.5, x + 2.5, y + 2.5), fill=INK, outline=SHADOW)
        offset = -15 if (index // every) % 2 == 0 else 9
        draw.text((x - 3, y + offset), str(index + 1), fill=INK, stroke_width=2, stroke_fill=SHADOW)
    start_x, start_y = coords_px[0]
    draw.ellipse((start_x - 4, start_y - 4, start_x + 4, start_y + 4), outline=INK, width=stroke + 1)
    draw.text((start_x + 6, start_y + 4), "0", fill=INK, stroke_width=2, stroke_fill=SHADOW)
    return image


def trace_against_layout(store, tile, frame, points, layout_id, base_detail):
    layout = store.layout(identity(layout_id, "Die layout ID"))
    side = layout_side(layout, tile["id"])
    if frame.scale is None:
        raise InputError("Enter the sheet dimensions (mm) for this specimen before classifying a fold line against the die.")
    position = layout["selected"] if layout.get("selected") is not None else 0
    outer_mask, window_mask = die_masks(frame.extent, layout, position, side)
    arcs, states, coords, total_mm = walk_trace(frame, outer_mask, window_mask, points)
    rows, totals, bucket_mm = bucket_trace(arcs, states, coords, total_mm)
    overlay = trace_overlay(frame, layout, position, side, arcs, states, coords, len(rows), bucket_mm)
    base_detail.update({
        "die_layout": {
            "id": layout["id"],
            "name": layout["name"],
            "side": side,
            "position": position,
            "position_label": POSITION_LABELS[position],
            "selected": layout.get("selected") is not None,
        },
        "bucket_mm": bucket_mm,
        "state_totals_mm": totals,
        "millimeters": rows,
    })
    return base_detail, overlay
