"""Back-side show-through ("see-through") alignment and attribution.

A thin sheet photographed on both sides lets the reverse plate explain faint
dark marks on the front.  The back plate is flipped around the edge the sheet
is turned on and then moved with a *single whole-plane translation*: local
features may never be nudged independently.  Two hand-marked anchors direct
the translation search:

* the left anchor is known show-through (a small back-side character that
  visibly bleeds through) and should land on a matching front darkening;
* the right anchor is known front-only work (a hairline that must be kept as
  front ink) and should not be covered by a back-side stroke.

Only dark marks with a real back-plate stroke behind them are attributed to
show-through.  Paper white inside a coarse anchor circle never counts, and the
report records what each anchor gained and lost at the chosen translation.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .art import base_rgba
from .geometry import polygon_mask, specimen_frame
from .model import InputError
from .photos import luminance, uint8_image


FLIP_EDGES = ("left", "right", "top", "bottom")
HORIZONTAL_FLIPS = ("left", "right")
SEARCH_LIMIT = 60          # whole-plane translation reach, registered pixels
COARSE_STEP = 4
MIN_ANCHOR_PIXELS = 24
HALO_PIXELS = 2            # ambiguous fringe around a confirmed back stroke
PLATE_CLOSE_RADIUS = 1

# Overlay colours (sRGB 0-255).
CONFIRMED_COLOR = (94, 205, 178)    # teal: back stroke with a matching front mark
CONFIRMED_SOFT = (120, 190, 170)    # faint teal: back stroke landing on paper white
AMBIGUOUS_COLOR = (238, 184, 92)    # amber: explainable either way, kept as possible ink
PROTECTED_COLOR = (120, 166, 232)   # blue: front-only anchor protected from the back plate
ANCHOR_LOSS_COLOR = (228, 104, 92)  # red: anchor darkening left unexplained / wrongly covered
OUTLINE_COLOR = (38, 46, 37)


@dataclass(frozen=True)
class ShowthroughViews:
    front_linear: np.ndarray
    front_valid: np.ndarray
    back_linear: np.ndarray          # already flipped to front orientation
    back_valid: np.ndarray
    extent: tuple
    scale: tuple | None
    flip_edge: str


def _dark_contrast(linear, support, radius):
    """Local paper-relative darkening in [0, 1].

    A small local maximum catches narrow hairlines, while a broad Gaussian
    reference recovers the paper level in the interior of a glyph wider than the
    local window.  Taking the brighter of the two keeps both small features and
    large blocks measurable without a grey-closing hole inside wide strokes.
    """
    gray = luminance(linear)
    local = ndimage.maximum_filter(gray, size=(radius * 2 + 1,) * 2, mode="nearest")
    broad_sigma = max(radius * 4.0, 12.0)
    broad = ndimage.gaussian_filter(gray, sigma=broad_sigma, mode="nearest")
    paper = np.maximum(local, broad)
    footprint = (radius * 2 + 1, radius * 2 + 1)
    usable = ndimage.minimum_filter(support, size=footprint, mode="constant", cval=False)
    contrast = np.clip((paper - gray) / np.maximum(paper, 0.025), 0, 1)
    return contrast.astype(np.float32), usable


def flipped_back(frame, flip_edge):
    """Mirror a registered back frame into the front reading orientation."""
    axis = 1 if flip_edge in HORIZONTAL_FLIPS else 0
    return np.flip(frame.linear, axis=axis).copy(), np.flip(frame.valid, axis=axis).copy()


def load_views(blobs, tile, front_id, back_id, flip_edge):
    front = specimen_frame(blobs, tile, front_id)
    back = specimen_frame(blobs, tile, back_id)
    if front_id == back_id:
        raise InputError("Choose two different photographs: the front view and the back view.")
    if front.extent != back.extent:
        raise InputError(
            "Both sides must share the registered specimen plane. Re-register the back view before aligning it."
        )
    back_linear, back_valid = flipped_back(back, flip_edge)
    return ShowthroughViews(
        front.linear, front.valid, back_linear, back_valid, front.extent, front.scale, flip_edge
    )


def anchor_mask(extent, vertices):
    mask = polygon_mask(extent, vertices)
    if np.count_nonzero(mask) < MIN_ANCHOR_PIXELS:
        raise InputError("Each anchor needs to enclose at least a few dozen registered pixels.")
    return mask


def plate_masks(views, back_radius, back_contrast):
    """Darkening on the flipped back view, plus the fringe that stays ambiguous."""
    contrast, usable = _dark_contrast(views.back_linear, views.back_valid, back_radius)
    plate = usable & (contrast >= back_contrast)
    # Close the small screened gaps inside a printed character so a whole
    # glyph reads as one plate stroke rather than isolated dots.
    plate = ndimage.binary_closing(
        plate, structure=np.ones((PLATE_CLOSE_RADIUS * 2 + 1,) * 2, dtype=bool)
    )
    halo = ndimage.binary_dilation(
        plate, iterations=HALO_PIXELS + 1
    ) & ~ndimage.binary_dilation(plate, iterations=1)
    return plate, halo, contrast, usable


def front_dark(views, front_radius, front_contrast):
    contrast, usable = _dark_contrast(views.front_linear, views.front_valid, front_radius)
    return usable & (contrast >= front_contrast), contrast, usable


def shift_array(mask, dy, dx):
    """Translate a boolean plane by (dy, dx); vacated pixels are empty."""
    return ndimage.shift(mask.astype(np.uint8), (dy, dx), order=0, mode="constant", cval=0).astype(bool)


def translation_score(views, front_dark_mask, plate, left_anchor, right_anchor, dy, dx, penalty):
    """Whole-plane fit at one translation; anchors share one identical shift."""
    moved_plate = shift_array(plate, dy, dx)
    moved_valid = shift_array(views.back_valid, dy, dx)
    common = views.front_valid & moved_valid
    left = left_anchor & common
    right = right_anchor & common
    if left.sum() < MIN_ANCHOR_PIXELS or right.sum() < MIN_ANCHOR_PIXELS:
        return None
    left_area = max(int(left.sum()), 1)
    right_area = max(int(right.sum()), 1)
    # Left anchor: the known bleed-through character should meet a back stroke.
    left_front = front_dark_mask & left
    left_hit = int((left_front & moved_plate).sum())
    left_gain = left_hit / left_area
    # Right anchor: the protected front hairline must not be explained away.
    right_front = front_dark_mask & right
    right_loss = int((right_front & moved_plate).sum())
    right_fraction = right_loss / right_area
    # Coverage tie-breaks otherwise-equal translations toward full paper support.
    coverage = (left.sum() / max(int(left_anchor.sum()), 1)
                + right.sum() / max(int(right_anchor.sum()), 1))
    return {
        "dy": dy,
        "dx": dx,
        "left_gain": left_gain,
        "left_hit_px": left_hit,
        "left_dark_px": int(left_front.sum()),
        "left_support_px": int(left.sum()),
        "right_loss_px": right_loss,
        "right_dark_px": int(right_front.sum()),
        "right_support_px": int(right.sum()),
        "right_loss_fraction": right_fraction,
        "coverage_term": round(coverage, 5),
        "score": left_gain - penalty * right_fraction + 1e-4 * coverage,
    }


def search_translation(views, front_dark_mask, plate, left_anchor, right_anchor, penalty):
    """Coarse-to-fine search over one shared whole-plane translation."""
    best = None
    for step in (COARSE_STEP, 1):
        offsets = range(-SEARCH_LIMIT, SEARCH_LIMIT + 1, step)
        for dy in offsets:
            for dx in offsets:
                if best is not None and step == 1:
                    if max(abs(dy - best["dy"]), abs(dx - best["dx"])) > COARSE_STEP:
                        continue
                reading = translation_score(
                    views, front_dark_mask, plate, left_anchor, right_anchor, dy, dx, penalty
                )
                if reading is None:
                    continue
                if best is None or reading["score"] > best["score"]:
                    best = reading
        if best is None:
            raise InputError(
                "No translation keeps both anchors on photographed paper. Widen the anchors or check the flip edge."
            )
    return best


def anchor_diagnostics(views, front_dark_mask, plate, left_anchor, right_anchor, dy, dx):
    """What each anchor gained and lost under the single adopted translation."""
    moved_plate = shift_array(plate, dy, dx)
    moved_valid = shift_array(views.back_valid, dy, dx)
    common = views.front_valid & moved_valid
    rows = []
    for label, anchor, expect in (
        ("show_through", left_anchor, "back"),
        ("front_only", right_anchor, "front"),
    ):
        domain = anchor & common
        dark = front_dark_mask & domain
        covered = dark & moved_plate
        support = anchor & views.front_valid
        rows.append({
            "anchor": label,
            "support_px": int(domain.sum()),
            "front_dark_px": int(dark.sum()),
            "covered_by_back_plate_px": int(covered.sum()),
            "dark_explained_percent": round(100 * covered.sum() / max(dark.sum(), 1), 2),
            "dark_unexplained_px": int(dark.sum() - covered.sum()),
            "paper_white_kept_px": int((domain & ~front_dark_mask & ~moved_plate).sum()),
            "support_coverage_percent": round(100 * domain.sum() / max(support.sum(), 1), 2),
            "expectation": expect,
        })
    return rows


def classify_showthrough(views, front_dark_mask, plate, halo, dy, dx, area):
    """Attribute front dark pixels with the flipped, translated back plate."""
    moved_plate = shift_array(plate, dy, dx)
    moved_halo = shift_array(halo, dy, dx)
    moved_valid = shift_array(views.back_valid, dy, dx)
    common = views.front_valid & moved_valid
    domain = area & common
    dark = front_dark_mask & domain
    confirmed = dark & moved_plate
    ambiguous = dark & ~moved_plate & moved_halo
    front_ink = dark & ~moved_plate & ~moved_halo
    paper_white = domain & ~front_dark_mask
    return {
        "domain": domain,
        "common_support": common,
        "dark": dark,
        "confirmed": confirmed,
        "ambiguous": ambiguous,
        "front_ink": front_ink,
        "paper_white": paper_white,
        "moved_plate": moved_plate,
        "moved_halo": moved_halo,
    }


def coverage_bounds(layers, scale):
    """Front-ink coverage interval over the paper visible from both sides."""
    domain_pixels = int(layers["domain"].sum())
    confirmed = int(layers["confirmed"].sum())
    ambiguous = int(layers["ambiguous"].sum())
    confident_ink = int(layers["front_ink"].sum())
    lower = confident_ink
    upper = confident_ink + ambiguous
    factor = (scale[0] * scale[1]) if scale else None

    def percent(value):
        return round(100 * value / domain_pixels, 3) if domain_pixels else None

    def area_mm2(value):
        return round(value * factor, 3) if factor else None

    return {
        "common_support_px": domain_pixels,
        "confirmed_show_through_px": confirmed,
        "ambiguous_px": ambiguous,
        "confident_front_ink_px": confident_ink,
        "front_ink_lower_percent": percent(lower),
        "front_ink_upper_percent": percent(upper),
        "front_ink_lower_mm2": area_mm2(lower),
        "front_ink_upper_mm2": area_mm2(upper),
        "excluded_show_through_percent": percent(confirmed),
        "denominator": "paper visible in both the front and flipped-back registered views",
        "note": (
            "Lower bound counts only dark marks with no back-plate explanation. "
            "The upper bound additionally keeps fringe pixels that could be front ink. "
            "Confirmed show-through is excluded from both bounds."
        ),
    }


def _polygon_outline(draw, vertices, extent, color, width):
    width_px, height_px = extent
    points = [(round(x * width_px - 0.5), round(y * height_px - 0.5)) for x, y in vertices]
    draw.line([*points, points[0]], fill=OUTLINE_COLOR, width=width + 2)
    draw.line([*points, points[0]], fill=color, width=width)


def showthrough_overlay(views, layers, anchors, front_frame, dy, dx):
    """Front view with confirmed show-through, ambiguous fringe and anchors."""
    image = base_rgba(front_frame)
    picture = np.asarray(image).copy()
    picture[layers["ambiguous"], :3] = AMBIGUOUS_COLOR
    picture[layers["confirmed"], :3] = CONFIRMED_COLOR
    # Show back-plate strokes that land on paper white as a faint teal outline,
    # proving that white inside the anchor circles is never counted as show-through.
    plate_on_white = layers["moved_plate"] & layers["paper_white"] & layers["domain"]
    picture[plate_on_white, :3] = (
        0.55 * picture[plate_on_white, :3] + 0.45 * np.asarray(CONFIRMED_SOFT, dtype=np.float32)
    ).astype(np.uint8)
    image = Image.fromarray(picture)
    draw = ImageDraw.Draw(image)
    line = max(1, round(min(image.size) / 360))
    left_vertices, right_vertices = anchors
    _polygon_outline(draw, left_vertices, views.extent, ANCHOR_LOSS_COLOR, line)
    _polygon_outline(draw, right_vertices, views.extent, PROTECTED_COLOR, line)
    draw.text((8, 6), f"whole-plane translation dy={dy} dx={dx} px", fill=(245, 241, 223),
              stroke_width=2, stroke_fill=OUTLINE_COLOR)
    return image


def run_showthrough(blobs, tile, payload):
    """Parse, align and attribute the back plate against the front view."""
    from .model import enumeration, identity, integer, scalar
    from .store import make_id

    front_id = identity(payload.get("front_photo_id"), "Front photo ID")
    back_id = identity(payload.get("back_photo_id"), "Back photo ID")
    flip_edge = enumeration(payload.get("flip_edge", "left"), FLIP_EDGES, "Flip edge")
    left_vertices = _anchor_vertices(payload, "left_anchor")
    right_vertices = _anchor_vertices(payload, "right_anchor")
    back_radius = integer(payload.get("back_radius", 3), "Back plate radius", lower=1, upper=12)
    back_contrast = scalar(payload.get("back_contrast", 0.16), "Back plate contrast", lower=0.03, upper=0.8)
    front_radius = integer(payload.get("front_radius", 3), "Front mark radius", lower=1, upper=12)
    front_contrast = scalar(payload.get("front_contrast", 0.05), "Front mark contrast", lower=0.03, upper=0.8)
    penalty = scalar(payload.get("protect_penalty", 1.0), "Front-anchor protection", lower=0.0, upper=8.0)

    views = load_views(blobs, tile, front_id, back_id, flip_edge)
    left_anchor = anchor_mask(views.extent, left_vertices)
    right_anchor = anchor_mask(views.extent, right_vertices)
    if np.count_nonzero(left_anchor & right_anchor):
        raise InputError("The two anchors must not overlap; keep the show-through and front-only zones apart.")

    plate, halo, _, _ = plate_masks(views, back_radius, back_contrast)
    front_dark_mask, _, _ = front_dark(views, front_radius, front_contrast)
    if plate.sum() < MIN_ANCHOR_PIXELS:
        raise InputError("The flipped back view shows no dark plate at these settings. Lower the back contrast.")

    chosen = search_translation(views, front_dark_mask, plate, left_anchor, right_anchor, penalty)
    dy, dx = chosen["dy"], chosen["dx"]
    anchor_rows = anchor_diagnostics(views, front_dark_mask, plate, left_anchor, right_anchor, dy, dx)
    inspection_vertices = _inspection_vertices(payload)
    area = polygon_mask(views.extent, inspection_vertices)
    layers = classify_showthrough(views, front_dark_mask, plate, halo, dy, dx, area)
    bounds = coverage_bounds(layers, views.scale)

    front_frame = specimen_frame(blobs, tile, front_id)
    overlay = showthrough_overlay(
        views, layers, (left_vertices, right_vertices), front_frame, dy, dx
    )
    from .art import stored_preview
    overlay_digest = stored_preview(overlay, blobs)

    record = {
        "id": make_id(),
        "front_photo_id": front_id,
        "back_photo_id": back_id,
        "flip_edge": flip_edge,
        "translation_px": [dy, dx],
        "left_anchor": left_vertices,
        "right_anchor": right_vertices,
        "inspection_vertices": inspection_vertices,
        "parameters": {
            "back_radius": back_radius,
            "back_contrast": back_contrast,
            "front_radius": front_radius,
            "front_contrast": front_contrast,
            "protect_penalty": penalty,
        },
        "fit": chosen,
        "anchors": anchor_rows,
        "coverage_bounds": bounds,
        "overlay_digest": overlay_digest,
    }
    return record, layers, views


def _anchor_vertices(payload, key):
    from .model import collection, numeric_pair
    raw = collection(payload.get(key), key, minimum=3, maximum=40)
    return [numeric_pair(item, "Anchor vertex", lower=0, upper=1) for item in raw]


def _inspection_vertices(payload):
    if payload.get("inspection_vertices"):
        return _anchor_vertices(payload, "inspection_vertices")
    return [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
