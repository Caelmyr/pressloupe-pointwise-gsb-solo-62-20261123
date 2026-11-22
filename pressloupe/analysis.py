import numpy as np

from .art import (
    annotated_regions,
    base_rgba,
    coverage_overlay,
    magnitude_overlay,
    profile_overlay,
    reviewed_markers,
    signed_overlay,
    stored_preview,
)
from .color import (
    calibration_for,
    capture_basis,
    compare_reference,
    corrected_linear,
    reference_fingerprint,
    reference_map,
    summarize_colors,
)
from .condition import aggregate_candidates, scan_marking, survey_joint
from .coverage import measure_coverage
from .blade import blade_width
from .defects import measure_defects
from .craze import measure_craze
from .rim import measure_rim
from .geometry import active_region, analysis_mask, region_domain, specimen_frame
from .lighting import measure_direction, measure_glare
from .model import LAMP_ROLES, ANALYSIS_LABELS, InputError, library_fields, swatch_approve_fields
from .parameters import TOOLS, analysis_parameters
from .profile import measure_profile
from .dots import measure_dots
from .store import key_for
from .swatch import (
    dual_illuminant_match,
    lamp_fingerprint,
    lamp_views,
    measure_swatch_rect,
    swatch_overlay_art,
    store_swatch_art,
)
from .texture import measure_texture


ANALYZER_VERSION = 3


def selected_capture(tile, explicit, expected_role=None):
    captures = tile["photos"]
    plane = tile.get("plane")
    if not plane:
        raise InputError("Add a specimen photograph before measuring.")
    registered = {plane["photo_id"], *tile["registrations"]}
    if explicit:
        found = next((item for item in captures if item["id"] == explicit), None)
        if found is None:
            raise InputError("The selected photo is no longer attached to this specimen.")
        if found["id"] not in registered:
            raise InputError("Register this photo in the tile plane before measuring it.")
        if expected_role and found["role"] != expected_role:
            raise InputError(f"Choose a {expected_role} photo for this measurement.")
        return found["id"]
    if expected_role is None:
        return plane["photo_id"]
    choices = [item["id"] for item in captures if item["role"] == expected_role and item["id"] in registered]
    if not choices:
        raise InputError(f"Add and register a {expected_role} photo before running this measurement.")
    if len(choices) > 1:
        raise InputError(f"Choose which registered {expected_role} photograph to measure.")
    return choices[0]


def capture_provenance(frame):
    source = frame.photo
    return {
        "id": source["id"],
        "display_name": source["name"],
        "role": source["role"],
        "original_digest": source["source_digest"],
        "working_digest": source["working_digest"],
        "raw_size": source["original_size"],
        "upright_size": [source["width"], source["height"]],
        "working_color_space": source["working_color_space"],
        "plane_to_capture": list(frame.transform),
        "synthetic": source["synthetic"],
    }


def specimen_provenance(tile):
    return {
        "id": tile["id"],
        "version": tile["version"],
        "name": tile["name"],
        "body": tile["body"],
        "glaze": tile["glaze"],
        "firing": tile["firing"],
        "batch": tile["batch"],
        "physical_mm": [tile["width_mm"], tile["height_mm"]],
    }


def support_digest(frame, area):
    total = int(np.count_nonzero(area))
    available = area & frame.valid
    taken = int(np.count_nonzero(available))
    clipped = int(np.count_nonzero(available & frame.clipped))
    return {
        "registered_image_extent": list(frame.extent),
        "requested_pixels": total,
        "available_pixels": taken,
        "clipped_pixels": clipped,
        "available_percent": round(100 * taken / total, 2) if total else 0,
        "calibrated_area_mm2": round(taken * frame.scale[0] * frame.scale[1], 3) if frame.scale else None,
    }


def reference_color_sample(store, tile, photo_id, chosen_region, apply_neutral):
    frame = specimen_frame(store.blobs, tile, photo_id)
    region = active_region(tile, chosen_region)
    domain = region_domain(frame.extent, tile, region)
    valid = analysis_mask(frame, domain, minimum=80, allow_clipped=False)
    linear, correction = corrected_linear(tile, frame, apply_neutral)
    return {
        "fingerprint": reference_fingerprint(linear, valid),
        "basis": capture_basis(tile, frame, correction),
        "region": region,
        "source": capture_provenance(frame),
        "specimen": specimen_provenance(tile),
        "annotated_digest": stored_preview(annotated_regions(frame, [region] if region else [], linear), store.blobs),
    }


def approve_reference(store, payload):
    fields = library_fields(payload)
    tile = store.tile(fields["tile_id"])
    reference = reference_color_sample(store, tile, None, fields["region_id"], True)
    return store.add_reference({**fields, **reference})


def approve_swatch_reference(store, payload):
    """Freeze one physical cut swatch with a fingerprint for each lamp."""
    fields = swatch_approve_fields(payload)
    tile = store.tile(fields["tile_id"])
    target_tile = store.tile(fields["target_tile_id"])
    views, per_lamp, rect, coverage, glare, region = measure_swatch_rect(store, tile, fields)
    target_views, _ = lamp_views(store, target_tile)
    target_region = active_region(target_tile, fields["target_region_id"])
    target_domain = region_domain(
        target_views[LAMP_ROLES[0]].frame.extent, target_tile, target_region,
    )
    target_fingerprints = {}
    for role in LAMP_ROLES:
        view = target_views[role]
        target_mask = target_domain & view.frame.valid & ~view.frame.clipped
        target_fingerprints[role], _, _ = lamp_fingerprint(view, target_mask)
    illuminants = {}
    for role in LAMP_ROLES:
        view = views[role]
        fingerprint, _, _ = lamp_fingerprint(view, per_lamp[role])
        target = target_fingerprints[role]
        correction = calibration_for(tile, view.frame)
        illuminants[role] = {
            "fingerprint": fingerprint,
            "source": capture_provenance(view.frame),
            "basis": capture_basis(tile, view.frame, correction),
            "eligible_pixels": int(per_lamp[role].sum()),
            "glare_pixels_excluded": int((glare[role] & rect).sum()),
            "target_fingerprint": target,
            "target_delta_e76": round(
                float(np.linalg.norm(
                    np.asarray(fingerprint["median_lab"]) - np.asarray(target["median_lab"]),
                )),
                3,
            ),
        }
    standard_frame = views[LAMP_ROLES[0]].frame
    annotated = swatch_overlay_art(standard_frame, views[LAMP_ROLES[0]].linear, rect, per_lamp)
    record = {
        "tile_id": fields["tile_id"],
        "title": fields["title"],
        "family": fields["family"],
        "color_number": fields["color_number"],
        "notes": fields["notes"],
        "region": region,
        "specimen": specimen_provenance(tile),
        "target_specimen": specimen_provenance(target_tile),
        "target_region": target_region,
        "illuminants": illuminants,
        "swatch": {
            "bounds": fields["bounds"],
            "size_mm": [fields["width_mm"], fields["height_mm"]],
            "orientation": fields["orientation"],
            "plane_size": list(standard_frame.extent),
            "coverage": coverage,
        },
        "glare_settings": {
            "glare_quantile": fields["glare_quantile"],
            "glare_ratio": fields["glare_ratio"],
        },
        "dual_illuminant": True,
        "annotated_digest": store_swatch_art(store, annotated),
        # Retained so older single-capture code paths and records keep working.
        "fingerprint": illuminants[LAMP_ROLES[0]]["fingerprint"],
        "basis": illuminants[LAMP_ROLES[0]]["basis"],
        "source": illuminants[LAMP_ROLES[0]]["source"],
    }
    return store.add_reference(record)


def resolve_parameters(store, tile, payload):
    params = analysis_parameters(payload)
    kind = params["kind"]
    setting = TOOLS[kind]
    roles = setting.get("capture_pair")
    params["photo_id"] = selected_capture(tile, params["photo_id"], roles[0] if roles else None)
    params["comparison_id"] = selected_capture(tile, params["comparison_id"], roles[1]) if roles else None
    if not roles and payload.get("comparison_id"):
        raise InputError("This measurement uses one capture.")
    reference = None
    if kind == "reference":
        reference = store.reference(params["options"]["reference_id"])
        if reference["retired"]:
            raise InputError("The selected reference has been retired.")
        params["lamp_photo_ids"] = resolve_lamp_photos(tile, reference, params)
    if kind == "defects" and params["options"]["maximum_area"] <= params["options"]["minimum_area"]:
        raise InputError("Maximum spot size must exceed the minimum.")
    return params, reference


def resolve_lamp_photos(tile, reference, params):
    """Map each signing lamp to the explicitly chosen capture, never silently.

    A lamp that the caller did not supply is left unmapped; the match then
    reports that illuminant as missing instead of substituting another photo.
    """
    if not reference.get("dual_illuminant"):
        return {}
    registered = {tile["plane"]["photo_id"], *tile["registrations"]}
    by_id = {photo["id"]: photo for photo in tile["photos"]}
    chosen = {}
    explicit_primary = params["photo_id"]
    explicit_second = params.get("lamp_comparison_id")
    for role in LAMP_ROLES:
        wanted = explicit_primary if role == LAMP_ROLES[0] else explicit_second
        if not wanted:
            continue
        photo = by_id.get(wanted)
        if photo is None or photo["id"] not in registered:
            raise InputError(f"Register the selected {role} capture before matching.")
        if photo["role"] != role:
            raise InputError(
                f"The {role} comparison needs a capture labeled {role}. "
                "A capture from another position or illuminant cannot be substituted.",
            )
        chosen[role] = wanted
    return chosen


def theoretical_legend(kind, scale=None):
    if kind == "color" or kind == "reference":
        return {"label": "ΔE76", "low": 0, "high": scale, "midpoint": None, "mode": "sequential"}
    if kind == "texture":
        return {"label": "Local intensity RMS", "low": 0, "high": scale, "midpoint": None, "mode": "sequential"}
    if kind == "glare":
        return {"label": "Fraction of parallel intensity", "low": 0, "high": scale, "midpoint": None, "mode": "sequential"}
    if kind == "direction":
        return {"label": "Raking contrast · right to left", "low": -scale, "high": scale, "midpoint": 0, "mode": "signed"}
    if kind == "coverage":
        return {"mode": "categorical", "labels": [
            {"name": "Bare", "color": "#ebb25e"},
            {"name": "Mixed", "color": "#dad892"},
            {"name": "Printed stock", "color": "#5ba88f"},
            {"name": "Unmeasured", "color": "#353a3a"},
        ]}
    return None


def formula_scale(values, support, floor):
    available = values[support]
    return max(float(np.quantile(available, 0.97)), floor) if len(available) else floor


def compute_single(store, tile, frame, region, area, linear, correction, params, reference):
    kind = params["kind"]
    options = params["options"]
    support = analysis_mask(frame, area, allow_clipped=False)
    if kind == "dots":
        metrics, image = measure_dots(frame, area, linear, options)
        return metrics, image, None
    if kind == "blade":
        metrics, image = blade_width(tile, frame, area, linear, options)
        return metrics, image, None
    if kind in {"marking", "joint", "aggregate"}:
        operation = {"marking": scan_marking, "joint": survey_joint, "aggregate": aggregate_candidates}[kind]
        metrics, image = operation(frame, area, linear, options)
        return metrics, image, None
    if kind == "color":
        metrics, map_values = summarize_colors(linear, support, options["grid"])
        scale = formula_scale(map_values, support, 6)
        image = magnitude_overlay(frame, map_values, support, scale, source=linear, unknown=area & ~support)
        return metrics, image, theoretical_legend(kind, scale)
    if kind == "coverage":
        unrestricted = region_domain(frame.extent, tile)
        metrics, states = measure_coverage(tile, frame, linear, area, unrestricted, options)
        return metrics, coverage_overlay(frame, states, area), theoretical_legend(kind)
    if kind == "defects":
        metrics = measure_defects(frame, linear, area, options)
        image = reviewed_markers(frame, metrics["observations"], source=linear)
        return metrics, image, None
    if kind == "texture":
        metrics, roughness, valid = measure_texture(frame, linear, area, options)
        scale = formula_scale(roughness, valid, 0.01)
        return metrics, magnitude_overlay(frame, roughness, valid, scale), theoretical_legend(kind, scale)
    if kind == "craze":
        metrics, image = measure_craze(frame, linear, area, options)
        return metrics, image, None
    if kind == "rim":
        metrics, image = measure_rim(frame, linear, area, options)
        return metrics, image, None
    if kind == "profile":
        metrics = measure_profile(frame, linear, area, options)
        image = profile_overlay(frame, metrics["endpoints"], options["band_pixels"], linear)
        return metrics, image, None
    if kind == "reference":
        if reference.get("dual_illuminant"):
            # Handled directly in evaluate() because the overlay can belong to a
            # different lamp frame than the primary capture.
            return None, None, None
        fingerprint = reference_fingerprint(linear, support)
        basis = capture_basis(tile, frame, correction)
        metrics = compare_reference(fingerprint, reference["fingerprint"], basis, reference["basis"])
        metrics["reference_title"] = reference["title"]
        metrics["reference_id"] = reference["id"]
        metrics["reference_image"] = reference["annotated_digest"]
        metrics["reference_region"] = reference["region"]
        map_values = reference_map(linear, reference["fingerprint"])
        scale = formula_scale(map_values, support, 12)
        return metrics, magnitude_overlay(frame, map_values, support, scale), theoretical_legend(kind, scale)
    raise RuntimeError(f"Unsupported image measurement: {kind}.")


def evaluate_dual_illuminant(store, tile, params, reference, area):
    glare_params = reference.get("glare_settings", {"glare_quantile": 0.90, "glare_ratio": 1.18})
    outcome, masks, selected = dual_illuminant_match(
        store, tile, params["region_id"], params["lamp_photo_ids"], reference, glare_params,
    )
    # The map uses one lamp's own frame; glare and pixels that lamp cannot see
    # are marked unknown instead of being folded into a false average.
    map_role = LAMP_ROLES[0] if LAMP_ROLES[0] in masks else next(iter(masks))
    primary = selected[map_role]
    primary_mask = masks[map_role]
    approved_median = np.asarray(
        reference["illuminants"][map_role]["fingerprint"]["median_lab"], dtype=np.float32,
    )
    map_values = np.linalg.norm(linear_lab_for(primary) - approved_median, axis=-1)
    outcome["map_illuminant"] = map_role
    support = analysis_mask(primary.frame, primary_mask, minimum=80, allow_clipped=False)
    unknown = area & ~support
    scale = formula_scale(map_values, support, 12)
    image = magnitude_overlay(primary.frame, map_values, support, scale, unknown=unknown)
    outcome["capture_note"] = (
        "Per-lamp ΔE values each compare same-illuminant captures of the same physical swatch. "
        "Missing lamps are reported explicitly and are never substituted by another position."
    )
    return outcome, image, theoretical_legend("reference", scale), primary, selected


def linear_lab_for(view):
    from .color import linear_lab
    return linear_lab(view.linear)


def compute_pair(tile, first, second, area, params):
    kind = params["kind"]
    options = params["options"]
    if kind == "glare":
        ties = active_region(tile, options["balance_region"])
        calibration_area = area if ties is None else region_domain(first.extent, tile, ties)
        metrics, map_values, valid, unknown = measure_glare(first, second, area, calibration_area, options)
        scale = max(options["glare_fraction"] * 2, 0.45)
        image = magnitude_overlay(first, map_values, valid, scale, unknown=unknown)
        return metrics, image, theoretical_legend(kind, scale)
    metrics, values, valid, unknown = measure_direction(first, second, area, options)
    scale = max(metrics["local_threshold"] * 2.4, 0.06)
    image = signed_overlay(first, values, valid, scale, unknown=unknown)
    return metrics, image, theoretical_legend(kind, scale)


def result_key(tile, params, reference):
    return key_for({
        "analyzer_version": ANALYZER_VERSION,
        "params": params,
        "tile_id": tile["id"],
        "tile_version": tile["version"],
        "reference_version": reference["version"] if reference else None,
    })


def evaluate(store, tile_id, payload):
    tile = store.tile(tile_id)
    params, reference = resolve_parameters(store, tile, payload)
    cache_key = result_key(tile, params, reference)
    cached = store.get_cached_result(cache_key)
    if cached:
        return cached
    frame = specimen_frame(store.blobs, tile, params["photo_id"])
    region = active_region(tile, params["region_id"])
    area = region_domain(frame.extent, tile, region)
    pair = specimen_frame(store.blobs, tile, params["comparison_id"]) if params["comparison_id"] else None
    correction = None
    linear = frame.linear
    map_frame = frame
    dual_lamp_views = None
    if pair is not None:
        metrics, picture, legend = compute_pair(tile, frame, pair, area, params)
    elif params["kind"] == "reference" and reference.get("dual_illuminant"):
        metrics, picture, legend, map_frame, dual_lamp_views = evaluate_dual_illuminant(
            store, tile, params, reference, area,
        )
        linear, correction = corrected_linear(tile, frame, params["apply_neutral"])
    else:
        linear, correction = corrected_linear(tile, frame, params["apply_neutral"])
        metrics, picture, legend = compute_single(store, tile, frame, region, area, linear, correction, params, reference)
    result = {
        "tile_id": tile_id,
        "source_version": tile["version"],
        "kind": params["kind"],
        "label": ANALYSIS_LABELS[params["kind"]],
        "parameters": params,
        "metrics": metrics,
        "legend": legend,
        "basis": capture_basis(tile, frame, correction),
        "support": support_digest(frame, area),
        "provenance": {
            "analyzer_version": ANALYZER_VERSION,
            "specimen": specimen_provenance(tile),
            "photo": capture_provenance(frame),
            "paired_photo": capture_provenance(pair) if pair else None,
            "lamp_photos": [
                capture_provenance(view.frame)
                for role, view in sorted(dual_lamp_views.items())
                if view is not None and role in params["lamp_photo_ids"]
            ] if dual_lamp_views is not None else None,
            "region": region,
            "exclusions": [item for item in tile["regions"] if item["purpose"] == "ignore"],
            "neutral_card": correction,
            "reference": {
                "id": reference["id"],
                "version": reference["version"],
                "specimen": reference["specimen"],
                "capture": reference.get("source"),
                "basis": reference.get("basis"),
                "dual_illuminant": bool(reference.get("dual_illuminant")),
                "illuminants": sorted(reference["illuminants"].keys())
                if reference.get("dual_illuminant") else None,
            } if reference else None,
        },
        "pictures": {
            "source": stored_preview(base_rgba(map_frame.frame, map_frame.linear), store.blobs)
            if dual_lamp_views is not None
            else stored_preview(base_rgba(frame, linear), store.blobs),
            "overlay": stored_preview(picture, store.blobs),
            "paired": stored_preview(base_rgba(pair), store.blobs) if pair else None,
        },
    }
    return store.save_result(cache_key, result)
