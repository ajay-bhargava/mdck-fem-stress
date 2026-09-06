"""Raster-to-vector geometry primitives for Stage 3.

Raster pixels are represented as unit squares centered on their zero-based
(column, row) image coordinates. Thus pixel (0, 0) spans [-0.5, 0.5] in both
axes, preserving raster area exactly while remaining consistent with the
persisted Stage 2 pixel-center transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from shapely import make_valid
from shapely.affinity import affine_transform
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree


Polygonal = Polygon | MultiPolygon


@dataclass
class PolygonCleanup:
    geometry: Polygonal
    original_area: float
    cleaned_area: float
    relative_area_change: float
    was_valid: bool
    is_valid: bool
    is_empty: bool
    multipolygon: bool
    cleanup_operations: list[str]


def polygonal_parts(geometry: object) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    if isinstance(geometry, MultiPolygon):
        return [part for part in geometry.geoms if not part.is_empty]
    if isinstance(geometry, GeometryCollection):
        parts: list[Polygon] = []
        for item in geometry.geoms:
            parts.extend(polygonal_parts(item))
        return parts
    return []


def _boundary_segments(mask: np.ndarray, row_offset: int, col_offset: int) -> list[LineString]:
    """Return exposed pixel edges in image pixel-center coordinates."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    segments: list[LineString] = []

    for row, col in np.argwhere(mask & ~up):
        x0 = col_offset + col - 0.5
        y0 = row_offset + row - 0.5
        segments.append(LineString(((x0, y0), (x0 + 1.0, y0))))
    for row, col in np.argwhere(mask & ~right):
        x0 = col_offset + col + 0.5
        y0 = row_offset + row - 0.5
        segments.append(LineString(((x0, y0), (x0, y0 + 1.0))))
    for row, col in np.argwhere(mask & ~down):
        x0 = col_offset + col + 0.5
        y0 = row_offset + row + 0.5
        segments.append(LineString(((x0, y0), (x0 - 1.0, y0))))
    for row, col in np.argwhere(mask & ~left):
        x0 = col_offset + col - 0.5
        y0 = row_offset + row + 0.5
        segments.append(LineString(((x0, y0), (x0, y0 - 1.0))))
    return segments


def raster_mask_to_polygon(
    mask: np.ndarray,
    *,
    row_offset: int = 0,
    col_offset: int = 0,
) -> Polygonal:
    """Polygonize a binary raster without smoothing or dropping components."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return Polygon()
    segments = _boundary_segments(mask, row_offset, col_offset)
    candidates = list(polygonize(segments))
    selected: list[Polygon] = []
    height, width = mask.shape
    for candidate in candidates:
        point = candidate.representative_point()
        col = int(np.floor(point.x - col_offset + 0.5))
        row = int(np.floor(point.y - row_offset + 0.5))
        if 0 <= row < height and 0 <= col < width and mask[row, col]:
            selected.append(candidate)
    if not selected:
        return Polygon()
    merged = unary_union(selected)
    parts = polygonal_parts(merged)
    if len(parts) == 1:
        return parts[0]
    return MultiPolygon(parts)


def clean_polygon(
    geometry: Polygonal,
    *,
    simplify_tolerance_px: float = 0.0,
) -> PolygonCleanup:
    """Apply only validity repair and optional topology-preserving simplification."""
    original_area = float(geometry.area)
    was_valid = bool(geometry.is_valid)
    operations: list[str] = []
    cleaned: object = geometry
    if not geometry.is_valid:
        cleaned = make_valid(geometry)
        operations.append("make_valid")
    parts = polygonal_parts(cleaned)
    if len(parts) == 1:
        cleaned = parts[0]
    elif len(parts) > 1:
        cleaned = MultiPolygon(parts)
    else:
        cleaned = Polygon()
    if simplify_tolerance_px > 0 and not cleaned.is_empty:
        cleaned = cleaned.simplify(simplify_tolerance_px, preserve_topology=True)
        operations.append(f"simplify_preserve_topology:{simplify_tolerance_px}")
    if not cleaned.is_valid and not cleaned.is_empty:
        cleaned = cleaned.buffer(0)
        operations.append("buffer_0")
    parts = polygonal_parts(cleaned)
    result: Polygonal
    if len(parts) == 1:
        result = parts[0]
    elif parts:
        result = MultiPolygon(parts)
    else:
        result = Polygon()
    cleaned_area = float(result.area)
    denominator = original_area if original_area else 1.0
    return PolygonCleanup(
        geometry=result,
        original_area=original_area,
        cleaned_area=cleaned_area,
        relative_area_change=abs(cleaned_area - original_area) / denominator,
        was_valid=was_valid,
        is_valid=bool(result.is_valid),
        is_empty=bool(result.is_empty),
        multipolygon=isinstance(result, MultiPolygon),
        cleanup_operations=operations,
    )


def image_to_physical_um(geometry: Polygonal, pixel_size_um: float) -> Polygonal:
    return affine_transform(geometry, [pixel_size_um, 0, 0, pixel_size_um, 0, 0])


def image_to_fem_um(
    geometry: Polygonal,
    *,
    scale_x_m: float,
    scale_y_m: float,
    offset_x_m: float,
    offset_y_m: float,
) -> Polygonal:
    """Apply the persisted Stage 2 image-index → FEM affine transform."""
    metres_to_micrometres = 1.0e6
    return affine_transform(
        geometry,
        [
            scale_x_m * metres_to_micrometres,
            0,
            0,
            scale_y_m * metres_to_micrometres,
            offset_x_m * metres_to_micrometres,
            offset_y_m * metres_to_micrometres,
        ],
    )


def geometry_rings(geometry: Polygonal) -> Iterable[tuple[np.ndarray, bool]]:
    """Yield exterior then interior rings for every polygon component."""
    for polygon in polygonal_parts(geometry):
        yield np.asarray(polygon.exterior.coords, dtype=np.float64), False
        for interior in polygon.interiors:
            yield np.asarray(interior.coords, dtype=np.float64), True


def adjacency_and_overlap(
    geometries: list[Polygonal],
    cell_ids: np.ndarray,
    *,
    minimum_shared_boundary_px: float,
) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    """Measure true shared boundaries and overlap using a spatial index."""
    tree = STRtree(geometries)
    edges: list[dict[str, float | int]] = []
    overlap_pair_count = 0
    overlap_total = 0.0
    overlap_maximum = 0.0
    for index, geometry in enumerate(geometries):
        if geometry.is_empty:
            continue
        for other_index in tree.query(geometry, predicate="intersects"):
            other_index = int(other_index)
            if other_index <= index:
                continue
            other = geometries[other_index]
            overlap_area = float(geometry.intersection(other).area)
            if overlap_area > 1.0e-9:
                overlap_pair_count += 1
                overlap_total += overlap_area
                overlap_maximum = max(overlap_maximum, overlap_area)
            shared = float(geometry.boundary.intersection(other.boundary).length)
            if shared >= minimum_shared_boundary_px:
                edges.append(
                    {
                        "cell_id_a": int(cell_ids[index]),
                        "cell_id_b": int(cell_ids[other_index]),
                        "shared_boundary_pixels": shared,
                    }
                )
    return edges, {
        "overlapping_cell_pairs": overlap_pair_count,
        "total_overlap_area_pixels": overlap_total,
        "maximum_overlap_area_pixels": overlap_maximum,
    }
