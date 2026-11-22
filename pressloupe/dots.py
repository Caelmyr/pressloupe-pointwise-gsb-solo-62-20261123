"""Screen-lattice halftone ink census.

The screened print repeats on a fixed lattice, just like the pinhole cross and
the transparent spacing grid used at the loupe.  Ink pixels are segmented from
the photographed proof, one lattice placement is fitted to the visible dot
positions, and every detected ink pixel is then handed to exactly one lattice
site (or left unattributed when no lattice can explain it).  Paper white is
never added to a footprint and one ink pixel can never belong to two dots.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

from .model import InputError


CONNECTIVITY = np.ones((3, 3), dtype=bool)
# A lattice placement is accepted only when most isolated dots sit close to a
# fitted site.  Two placements within this residual band are both reported.
FIT_MIN_ANCHORS = 12
FIT_MIN_PITCH = 4.0
FIT_MAX_PITCH = 80.0
FIT_ACCEPT_RMS = 2.6
FIT_AMBIGUITY_BAND = 1.18
# Squeezed pairs/triplets can be separated by their lattice sites; a connected
# ink group spanning more sites is a solid patch, rule or smear, not a cluster
# of countable dots.
MAX_SPLIT_SITES = 3


def local_ink_contrast(linear, support, radius):
    # Grey closing estimates light paper across narrow dark marks. Mask the
    # support erosion too: a missing margin cannot supply a white reference.
    luminance = linear @ np.array([.2126, .7152, .0722], dtype=np.float32)
    footprint = (radius * 2 + 1, radius * 2 + 1)
    paper = ndimage.grey_closing(luminance, size=footprint, mode="nearest")
    valid = ndimage.minimum_filter(support, size=footprint, mode="constant", cval=False)
    return np.clip((paper - luminance) / np.maximum(paper, .025), 0, 1), valid


def segment_ink(frame, area, options):
    if options["minimum_area"] >= options["maximum_area"]:
        raise InputError("The smallest dot must be smaller than the largest dot.")
    support = area & frame.valid & ~frame.clipped
    contrast, valid = local_ink_contrast(frame.linear, support, options["radius"])
    if np.count_nonzero(valid) < 144:
        raise InputError("The region has too little supported paper for this background radius.")
    ink = valid & (contrast >= options["contrast"])
    return ink, valid


def component_record(labels, index, bounds):
    y, x = np.nonzero(labels[bounds] == index)
    y += bounds[0].start
    x += bounds[1].start
    return y, x


def isolated_anchors(labels, sizes, objects, minimum, maximum):
    """Centres of size-qualified components: the photographed pinhole positions."""
    anchors = []
    for index, bounds in enumerate(objects, start=1):
        if bounds is None or not minimum <= sizes[index] <= maximum:
            continue
        y, x = component_record(labels, index, bounds)
        anchors.append((x.mean() + 0.5, y.mean() + 0.5, len(x)))
    if len(anchors) < FIT_MIN_ANCHORS:
        return None
    points = np.asarray([(a, b) for a, b, _ in anchors], dtype=np.float64)
    weights = np.asarray([w for _, _, w in anchors], dtype=np.float64)
    return points, weights


def _direction_cluster(vectors):
    """Cluster unsigned directions (180 deg periodic) into one broad bin."""
    angles = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), np.pi)
    order = np.argsort(angles)
    sorted_angles = angles[order]
    # Duplicate the circle so a cluster straddling 0/pi stays contiguous.
    wrapped = np.concatenate((sorted_angles, sorted_angles + np.pi))
    best = None
    start = 0
    for end in range(len(wrapped)):
        while wrapped[end] - wrapped[start] > np.pi / 6:
            start += 1
        if end - start + 1 >= 4 and (best is None or end - start > best[1] - best[0]):
            best = (start, end)
    if best is None:
        return None
    chosen = vectors[order[np.mod(np.arange(best[0], best[1] + 1), len(order))]]
    # Orient every member the same unsigned way before averaging.
    flips = chosen[:, 1] < 0
    chosen[flips] *= -1
    return chosen.mean(axis=0)


def _shortest_period(lengths, deltas, near, mask=None, floor_ratio=0.12):
    """Find the shortest prominent neighbour spacing in a displacement subset."""
    selected = near if mask is None else near & mask
    bins = np.arange(FIT_MIN_PITCH, FIT_MAX_PITCH + 0.25, 0.25)
    histogram, edges = np.histogram(lengths[selected], bins=bins)
    floor = max(6, int(histogram.max() * floor_ratio))
    # Local maxima relative to a three-bin neighbourhood; harmonics are taller
    # but the fundamental pitch is the first prominent peak from the left.
    peaks = []
    for index in range(1, len(histogram) - 1):
        if histogram[index] < floor:
            continue
        if histogram[index] >= histogram[index - 1] and histogram[index] >= histogram[index + 1]:
            peaks.append((int(histogram[index]), index))
    if not peaks:
        return None
    _, peak_bin = min(peaks, key=lambda item: item[1])
    pitch_lo, pitch_hi = edges[peak_bin], edges[peak_bin + 1]
    period_vectors = deltas[selected & (lengths >= pitch_lo) & (lengths < pitch_hi)]
    if len(period_vectors) < 6:
        return None
    return period_vectors


def fit_lattice(anchors):
    """Fit one screen placement from isolated dot centres.

    Returns ``(basis, phase, diagnostics)`` where ``basis`` holds two basis
    vectors in pixels and ``phase`` is the lattice coordinate origin.  ``None``
    is returned when the photographed positions cannot support one placement.
    """
    packed = isolated_anchors(*anchors)
    if packed is None:
        return None
    points, weights = packed
    # Keep the displacement table tractable on dense screens.
    if len(points) > 320:
        keep = np.linspace(0, len(points) - 1, 320).astype(int)
        points, weights = points[keep], weights[keep]
    deltas = points[:, None, :] - points[None, :, :]
    lengths = np.linalg.norm(deltas, axis=2)
    paired = np.triu(np.ones(lengths.shape, dtype=bool), k=1)
    near = paired & (lengths >= FIT_MIN_PITCH) & (lengths <= FIT_MAX_PITCH)
    if int(near.sum()) < 16:
        return None
    first_family = _shortest_period(lengths, deltas, near)
    if first_family is None:
        return None
    basis_a = _direction_cluster(first_family)
    if basis_a is None:
        return None
    # The second family is the shortest prominent period running in a clearly
    # different direction, so a harmonic of the first basis cannot win.
    angular_gap = np.abs(np.arctan2(
        basis_a[1] * deltas[..., 0] - basis_a[0] * deltas[..., 1],
        basis_a[0] * deltas[..., 0] + basis_a[1] * deltas[..., 1],
    ))
    transverse = (angular_gap > np.pi / 4) & (angular_gap < 3 * np.pi / 4)
    if int((near & transverse).sum()) < 8:
        return None
    second_family = _shortest_period(lengths, deltas, near, transverse)
    if second_family is None:
        return None
    basis_b = _direction_cluster(second_family)
    if basis_b is None:
        return None
    basis = np.column_stack((basis_a, basis_b))
    if abs(np.linalg.det(basis)) < 9.0:
        return None
    inverse = np.linalg.inv(basis)
    coords = points @ inverse.T
    fractions = coords - np.round(coords)
    # Phase candidates come from the observed fractional offsets; the same
    # placement is reached through equivalent rounded lattice coordinates.
    candidates = np.unique(np.round(fractions, 4), axis=0)
    if len(candidates) > 400:
        spread = np.ptp(fractions, axis=0)
        steps = np.maximum(np.round(spread * 18).astype(int), 1)
        grid = np.array(
            np.meshgrid(np.linspace(fractions[:, 0].min(), fractions[:, 0].max(), steps[0] + 1),
                        np.linspace(fractions[:, 1].min(), fractions[:, 1].max(), steps[1] + 1),
                        indexing="ij"),
        ).reshape(2, -1).T
        candidates = grid
    residual_sq = ((fractions[None, :, :] - candidates[:, None, :]) ** 2).sum(axis=2)
    scores = np.sqrt((residual_sq * weights).sum(axis=1) / weights.sum())
    order = np.argsort(scores)
    # Sub-pixel jitter produces many near-identical phase candidates; cluster
    # them before comparing placements.  A genuinely different screen placement
    # (the two pinhole-cross readings) shifts the phase by a visible fraction
    # of the pitch, not by a few hundredths of a lattice unit.
    cluster_centres = []
    cluster_scores = []
    for rank in order:
        phase_candidate = candidates[rank]
        for index, centre in enumerate(cluster_centres):
            if np.linalg.norm(phase_candidate - centre) <= 0.15:
                cluster_scores[index] = min(cluster_scores[index], float(scores[rank]))
                break
        else:
            cluster_centres.append(phase_candidate)
            cluster_scores.append(float(scores[rank]))
    best = cluster_scores[0]
    runner_up = cluster_scores[1] if len(cluster_scores) > 1 else best
    if best > FIT_ACCEPT_RMS:
        return None
    phase = cluster_centres[0]
    diagnostics = {
        "anchor_components": int(len(weights)),
        "fit_rms_px": round(float(best), 3),
        "second_placement_rms_px": round(float(runner_up), 3),
        "ambiguous_placement": bool(best < runner_up <= best * FIT_AMBIGUITY_BAND),
        "placement_candidates": len(cluster_centres),
        "pitch_px": [round(float(np.linalg.norm(basis_a)), 3), round(float(np.linalg.norm(basis_b)), 3)],
        "angle_deg": round(float(np.degrees(np.arctan2(basis_a[1], basis_a[0])) % 180), 2),
    }
    return basis, phase, diagnostics


def assign_sites(ink, basis, phase):
    """Give every ink pixel to its nearest lattice site.

    Returns the site index map (0 on paper white / unsupported pixels) and the
    lookup of lattice (i, j) coordinates per positive site index.
    """
    height, width = ink.shape
    yy, xx = np.nonzero(ink)
    pixel_centres = np.column_stack((xx + 0.5, yy + 0.5)).astype(np.float64)
    inverse = np.linalg.inv(basis)
    lattice = np.round((pixel_centres @ inverse.T) - phase).astype(np.int64)
    site_lookup = {}
    site_map = np.zeros((height, width), dtype=np.int32)
    # Row-major site ids keep neighbouring sites close in id space.
    def site_id(ij):
        key = (int(ij[0]), int(ij[1]))
        identifier = site_lookup.get(key)
        if identifier is None:
            identifier = len(site_lookup) + 1
            site_lookup[key] = identifier
        return identifier
    ids = np.empty(len(lattice), dtype=np.int32)
    for row in range(len(lattice)):
        ids[row] = site_id(lattice[row])
    site_map[yy, xx] = ids
    coordinates = [None] * (len(site_lookup) + 1)
    for (i, j), identifier in site_lookup.items():
        coordinates[identifier] = [i, j]
    ij_values = np.asarray(list(site_lookup.keys()), dtype=np.int64)
    spans = (ij_values.max(axis=0) - ij_values.min(axis=0) + 1).tolist()
    return site_map, coordinates, spans


def ink_components(ink, valid):
    labels, count = ndimage.label(ink, structure=CONNECTIVITY)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    objects = ndimage.find_objects(labels)
    # A component touching the eroded edge of the supported paper has part of
    # its footprint outside the measurable domain; its own outline is not a
    # boundary even though every component has one.
    support_edge = valid & ~ndimage.binary_erosion(valid, structure=CONNECTIVITY)
    touching = set(np.unique(labels[support_edge]))
    return labels, count, sizes, objects, touching


def _nearest_mask(mask, target_size):
    width, height = target_size
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
            (width, height), Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    ) > 0


def _nearest_labelled_site(site_map, ink):
    """Fill the rare resampled ink pixel whose site label did not survive."""
    unlabelled = ink & (site_map == 0)
    if not unlabelled.any():
        return site_map
    nearest = ndimage.distance_transform_edt(
        site_map == 0, return_distances=False, return_indices=True,
    )
    filled = site_map.copy()
    filled[unlabelled] = site_map[tuple(nearest[:, unlabelled])]
    return filled


def downscale_census(census, target_size):
    """Resample the frozen pixel evidence to the stored preview resolution.

    Label maps use nearest-neighbour sampling so ink pixels stay whole and no
    paper white is invented by interpolation; component groups are rebuilt on
    the resampled evidence so every stored number matches the saved maps.
    """
    width, height = target_size
    if target_size == census["ink"].shape[::-1]:
        return census
    ink = _nearest_mask(census["ink"], target_size)
    valid = _nearest_mask(census["valid"], target_size)
    site_values = census["site_map"].astype(np.int32)
    low = (site_values & 0xFFFF).astype(np.uint8)
    high = ((site_values >> 16) & 0xFF).astype(np.uint8)
    packed = Image.fromarray(np.dstack((low, high, np.zeros_like(low))), mode="RGB")
    packed = packed.resize((width, height), Image.Resampling.NEAREST)
    channels = np.asarray(packed, dtype=np.int32)
    site_map = channels[..., 0] | (channels[..., 1] << 16)
    site_map[~ink] = 0
    # Both maps use the same nearest sampling, so an ink pixel can lose its
    # site only when the lattice placement itself was unavailable (all zeros).
    if int(site_map.max(initial=0)) > 0:
        site_map[ink & (site_map == 0)] = _nearest_labelled_site(site_map, ink)
    labels, count = ndimage.label(ink, structure=CONNECTIVITY)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    objects = ndimage.find_objects(labels)
    border = valid & ~ndimage.binary_erosion(valid, structure=CONNECTIVITY)
    touching = set(np.unique(labels[border]))
    groups = []
    for index, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        y, x = component_record(labels, index, bounds)
        site_ids = [int(value) for value in np.unique(site_map[y, x]) if value > 0]
        groups.append({
            "group": f"C{index:04d}",
            "component_label": int(index),
            "pixels": int(sizes[index]),
            "bounds": [int(bounds[1].start), int(bounds[0].start),
                       int(bounds[1].stop), int(bounds[0].stop)],
            "touches_boundary": index in touching,
            "site_ids": site_ids,
        })
    return {
        **census,
        "ink": ink,
        "valid": valid,
        "site_map": site_map.astype(np.int32),
        "component_labels": labels,
        "component_sizes": sizes,
        "component_objects": objects,
        "boundary_components": touching,
        "groups": groups,
    }


def build_census(frame, area, options):
    """Segment ink, fit one lattice placement and partition every ink pixel."""
    ink, valid = segment_ink(frame, area, options)
    labels, count, sizes, objects, touching = ink_components(ink, valid)
    anchors = (labels, sizes, objects, options["minimum_area"], options["maximum_area"])
    fitted = fit_lattice(anchors)
    if fitted is None:
        site_map = np.zeros(ink.shape, dtype=np.int32)
        coordinates, spans, lattice = [], None, None
    else:
        basis, phase, lattice = fitted
        site_map, coordinates, spans = assign_sites(ink, basis, phase)
    groups = []
    for index, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        y, x = component_record(labels, index, bounds)
        site_ids = np.unique(site_map[y, x])
        site_ids = site_ids[site_ids > 0]
        groups.append({
            "group": f"C{index:04d}",
            "component_label": int(index),
            "pixels": int(sizes[index]),
            "bounds": [int(bounds[1].start), int(bounds[0].start),
                       int(bounds[1].stop), int(bounds[0].stop)],
            "touches_boundary": index in touching,
            "site_ids": [int(value) for value in site_ids],
        })
    return {
        "ink": ink,
        "valid": valid,
        "component_labels": labels,
        "component_sizes": sizes,
        "component_objects": objects,
        "boundary_components": touching,
        "site_map": site_map,
        "site_coordinates": coordinates,
        "lattice": lattice,
        "lattice_spans": spans,
        "groups": groups,
    }
