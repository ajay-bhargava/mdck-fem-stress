"""Bilinear traction interpolation and consistent P1 load integration.

All mechanics calculations use SI units: coordinates in metres, traction in
Pa (N/m²), nodal loads in N, and moments in N·m.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import shapely
from shapely.strtree import STRtree


@dataclass(frozen=True)
class TriangleQuadrature:
    name: str
    polynomial_order: int
    barycentric: np.ndarray
    weights: np.ndarray


@dataclass
class BilinearMap:
    ix: np.ndarray
    iy_ascending: np.ndarray
    wx: np.ndarray
    wy: np.ndarray
    outside: np.ndarray
    reverse_source_y: bool

    def interpolate(self, values: np.ndarray) -> np.ndarray:
        """Interpolate source values shaped (ny, nx, frames) at mapped points."""
        source = values[::-1, :, :] if self.reverse_source_y else values
        v00 = source[self.iy_ascending, self.ix]
        v10 = source[self.iy_ascending, self.ix + 1]
        v01 = source[self.iy_ascending + 1, self.ix]
        v11 = source[self.iy_ascending + 1, self.ix + 1]
        wx = self.wx[:, None]
        wy = self.wy[:, None]
        result = (
            (1.0 - wx) * (1.0 - wy) * v00
            + wx * (1.0 - wy) * v10
            + (1.0 - wx) * wy * v01
            + wx * wy * v11
        )
        if np.any(self.outside):
            result[self.outside] = np.nan
        return result


@dataclass
class MappingResult:
    tx_element_pa: np.ndarray
    ty_element_pa: np.ndarray
    fx_node_n: np.ndarray
    fy_node_n: np.ndarray
    metrics: list[dict[str, float | int | None]]
    quadrature_points_total: int
    quadrature_points_outside: int
    runtime_seconds: float


def quadrature_rule(kind: str) -> TriangleQuadrature:
    if kind == "centroid":
        return TriangleQuadrature(
            "centroid_1_point", 1, np.asarray([[1 / 3, 1 / 3, 1 / 3]]), np.ones(1)
        )
    if kind == "degree2":
        barycentric = np.asarray(
            [[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]]
        )
        return TriangleQuadrature("symmetric_3_point", 2, barycentric, np.full(3, 1 / 3))
    if kind == "degree5":
        points = [[1 / 3, 1 / 3, 1 / 3]]
        weights = [0.225]
        for a, b, weight in (
            (0.059715871789770, 0.470142064105115, 0.132394152788506),
            (0.797426985353087, 0.101286507323456, 0.125939180544827),
        ):
            points.extend(((a, b, b), (b, a, b), (b, b, a)))
            weights.extend((weight, weight, weight))
        return TriangleQuadrature(
            "Dunavant_7_point", 5, np.asarray(points), np.asarray(weights)
        )
    if kind == "degree7":
        points = [[1 / 3, 1 / 3, 1 / 3]]
        weights = [-0.149570044467682]
        for a, b, weight in (
            (0.479308067841923, 0.260345966079038, 0.175615257433208),
            (0.869739794195568, 0.065130102902216, 0.053347235608839),
        ):
            points.extend(((a, b, b), (b, a, b), (b, b, a)))
            weights.extend((weight, weight, weight))
        a, b, c, weight = (
            0.638444188569809,
            0.312865496004874,
            0.048690315425316,
            0.077113760890257,
        )
        points.extend(((a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)))
        weights.extend((weight,) * 6)
        return TriangleQuadrature(
            "Dunavant_13_point", 7, np.asarray(points), np.asarray(weights)
        )
    raise ValueError(f"unknown triangle quadrature rule: {kind}")


def transform_source_grid_to_fem_m(
    x_source: np.ndarray,
    y_source: np.ndarray,
    *,
    pixel_size_m: float,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the persisted Stage 2 one-based source → FEM transform."""
    x_fem_m = (np.asarray(x_source, dtype=np.float64) - 1.0) * pixel_size_m
    y_fem_m = (image_height - np.asarray(y_source, dtype=np.float64)) * pixel_size_m
    return x_fem_m, y_fem_m


def transform_source_vectors_to_fem(
    tx_source: np.ndarray,
    ty_source: np.ndarray,
    *,
    semantic_sign: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reflect vector basis with y and optionally apply a semantic action/reaction sign."""
    return semantic_sign * np.asarray(tx_source), -semantic_sign * np.asarray(ty_source)


def prepare_bilinear_map(
    x_grid: np.ndarray, y_grid: np.ndarray, query_xy: np.ndarray
) -> BilinearMap:
    x_axis = np.asarray(x_grid[0], dtype=np.float64)
    y_source_axis = np.asarray(y_grid[:, 0], dtype=np.float64)
    reverse_y = bool(y_source_axis[0] > y_source_axis[-1])
    y_axis = y_source_axis[::-1] if reverse_y else y_source_axis
    if np.any(np.diff(x_axis) <= 0) or np.any(np.diff(y_axis) <= 0):
        raise ValueError("bilinear interpolation requires strictly monotonic regular axes")
    if not np.allclose(x_grid, np.broadcast_to(x_axis, x_grid.shape), atol=1e-15, rtol=0):
        raise ValueError("traction x grid is not rectilinear")
    expected_y = np.broadcast_to(y_source_axis[:, None], y_grid.shape)
    if not np.allclose(y_grid, expected_y, atol=1e-15, rtol=0):
        raise ValueError("traction y grid is not rectilinear")

    qx = query_xy[:, 0]
    qy = query_xy[:, 1]
    tolerance = 1.0e-12 * max(np.ptp(x_axis), np.ptp(y_axis), 1.0)
    outside = (
        (qx < x_axis[0] - tolerance)
        | (qx > x_axis[-1] + tolerance)
        | (qy < y_axis[0] - tolerance)
        | (qy > y_axis[-1] + tolerance)
    )
    ix = np.clip(np.searchsorted(x_axis, qx, side="right") - 1, 0, len(x_axis) - 2)
    iy = np.clip(np.searchsorted(y_axis, qy, side="right") - 1, 0, len(y_axis) - 2)
    wx = (qx - x_axis[ix]) / (x_axis[ix + 1] - x_axis[ix])
    wy = (qy - y_axis[iy]) / (y_axis[iy + 1] - y_axis[iy])
    return BilinearMap(ix, iy, wx, wy, outside, reverse_y)


def triangle_quadrature_points(
    nodes_xy_m: np.ndarray, triangles: np.ndarray, rule: TriangleQuadrature
) -> np.ndarray:
    triangle_nodes = nodes_xy_m[triangles]
    return np.einsum("qi,eid->eqd", rule.barycentric, triangle_nodes)


def map_traction_frames(
    *,
    nodes_xy_um: np.ndarray,
    triangles: np.ndarray,
    x_grid_fem_m: np.ndarray,
    y_grid_fem_m: np.ndarray,
    tx_fem_pa: np.ndarray,
    ty_fem_pa: np.ndarray,
    frame_indices: np.ndarray,
    moment_reference_xy_m: np.ndarray,
    rule: TriangleQuadrature,
    frame_chunk_size: int = 4,
) -> MappingResult:
    import time

    started = time.perf_counter()
    nodes_m = np.asarray(nodes_xy_um, dtype=np.float64) * 1.0e-6
    triangles = np.asarray(triangles, dtype=np.int64)
    triangle_nodes = nodes_m[triangles]
    cross = (
        (triangle_nodes[:, 1, 0] - triangle_nodes[:, 0, 0])
        * (triangle_nodes[:, 2, 1] - triangle_nodes[:, 0, 1])
        - (triangle_nodes[:, 1, 1] - triangle_nodes[:, 0, 1])
        * (triangle_nodes[:, 2, 0] - triangle_nodes[:, 0, 0])
    )
    area_m2 = 0.5 * cross
    if np.any(area_m2 <= 0):
        raise ValueError("consistent load integration requires positive triangle areas")

    quadrature_xy = triangle_quadrature_points(nodes_m, triangles, rule)
    element_count, point_count, _ = quadrature_xy.shape
    interpolation = prepare_bilinear_map(
        x_grid_fem_m, y_grid_fem_m, quadrature_xy.reshape(-1, 2)
    )
    outside_count = int(np.sum(interpolation.outside))
    if outside_count:
        raise ValueError(
            f"{outside_count}/{len(interpolation.outside)} quadrature points outside traction support"
        )

    selected_count = len(frame_indices)
    tx_element = np.empty((selected_count, element_count), dtype=np.float32)
    ty_element = np.empty((selected_count, element_count), dtype=np.float32)
    fx_node = np.zeros((selected_count, len(nodes_m)), dtype=np.float64)
    fy_node = np.zeros_like(fx_node)
    metrics: list[dict[str, float | int | None]] = []
    weights = rule.weights
    barycentric = rule.barycentric
    qx = quadrature_xy[..., 0]
    qy = quadrature_xy[..., 1]

    for chunk_start in range(0, selected_count, frame_chunk_size):
        chunk_stop = min(selected_count, chunk_start + frame_chunk_size)
        requested = frame_indices[chunk_start:chunk_stop]
        tx_values = interpolation.interpolate(tx_fem_pa[..., requested]).reshape(
            element_count, point_count, -1
        )
        ty_values = interpolation.interpolate(ty_fem_pa[..., requested]).reshape(
            element_count, point_count, -1
        )
        average_tx = np.einsum("q,eqf->ef", weights, tx_values)
        average_ty = np.einsum("q,eqf->ef", weights, ty_values)
        tx_element[chunk_start:chunk_stop] = average_tx.T.astype(np.float32)
        ty_element[chunk_start:chunk_stop] = average_ty.T.astype(np.float32)

        local_fx = (
            area_m2[:, None, None]
            * np.einsum("q,qi,eqf->eif", weights, barycentric, tx_values)
        )
        local_fy = (
            area_m2[:, None, None]
            * np.einsum("q,qi,eqf->eif", weights, barycentric, ty_values)
        )
        flattened_nodes = triangles.ravel()
        for local_frame in range(len(requested)):
            output_frame = chunk_start + local_frame
            fx_node[output_frame] = np.bincount(
                flattened_nodes,
                weights=local_fx[..., local_frame].ravel(),
                minlength=len(nodes_m),
            )
            fy_node[output_frame] = np.bincount(
                flattened_nodes,
                weights=local_fy[..., local_frame].ravel(),
                minlength=len(nodes_m),
            )

            tx_frame = tx_values[..., local_frame]
            ty_frame = ty_values[..., local_frame]
            source_fx = float(np.sum(area_m2[:, None] * weights * tx_frame))
            source_fy = float(np.sum(area_m2[:, None] * weights * ty_frame))
            traction_magnitude = np.hypot(tx_frame, ty_frame)
            integrated_magnitude = float(
                np.sum(area_m2[:, None] * weights * traction_magnitude)
            )
            moment_density = (
                (qx - moment_reference_xy_m[0]) * ty_frame
                - (qy - moment_reference_xy_m[1]) * tx_frame
            )
            source_moment = float(np.sum(area_m2[:, None] * weights * moment_density))
            integrated_abs_moment = float(
                np.sum(area_m2[:, None] * weights * np.abs(moment_density))
            )
            fem_fx = float(np.sum(fx_node[output_frame]))
            fem_fy = float(np.sum(fy_node[output_frame]))
            fem_moment = float(
                np.sum(
                    (nodes_m[:, 0] - moment_reference_xy_m[0]) * fy_node[output_frame]
                    - (nodes_m[:, 1] - moment_reference_xy_m[1]) * fx_node[output_frame]
                )
            )
            force_error = float(np.hypot(fem_fx - source_fx, fem_fy - source_fy))
            moment_error = abs(fem_moment - source_moment)
            source_force = float(np.hypot(source_fx, source_fy))
            fem_force = float(np.hypot(fem_fx, fem_fy))
            force_relative = (
                force_error / source_force
                if source_force > integrated_magnitude * 1.0e-12
                else None
            )
            moment_relative = (
                moment_error / abs(source_moment)
                if abs(source_moment) > integrated_abs_moment * 1.0e-12
                else None
            )
            metrics.append(
                {
                    "frame": int(requested[local_frame]),
                    "source_fx_N": source_fx,
                    "source_fy_N": source_fy,
                    "source_force_magnitude_N": source_force,
                    "source_integrated_traction_magnitude_N": integrated_magnitude,
                    "source_net_force_fraction": source_force / integrated_magnitude,
                    "source_moment_Nm": source_moment,
                    "source_integrated_abs_moment_Nm": integrated_abs_moment,
                    "source_net_moment_fraction": abs(source_moment) / integrated_abs_moment,
                    "fem_fx_N": fem_fx,
                    "fem_fy_N": fem_fy,
                    "fem_force_magnitude_N": fem_force,
                    "fem_moment_Nm": fem_moment,
                    "absolute_fx_error_N": abs(fem_fx - source_fx),
                    "absolute_fy_error_N": abs(fem_fy - source_fy),
                    "absolute_force_mapping_error_N": force_error,
                    "relative_force_mapping_error": force_relative,
                    "force_mapping_error_over_integrated_traction": (
                        force_error / integrated_magnitude
                    ),
                    "absolute_moment_mapping_error_Nm": moment_error,
                    "relative_moment_mapping_error": moment_relative,
                    "moment_mapping_error_over_integrated_abs_moment": (
                        moment_error / integrated_abs_moment
                    ),
                    "quadrature_outside_support_fraction": 0.0,
                }
            )
    return MappingResult(
        tx_element,
        ty_element,
        fx_node,
        fy_node,
        metrics,
        element_count * point_count,
        outside_count,
        time.perf_counter() - started,
    )


def source_grid_to_element_index(
    nodes_xy_um: np.ndarray,
    triangles: np.ndarray,
    source_grid_xy_m: np.ndarray,
) -> np.ndarray:
    """Locate source points in mesh triangles for a defined back-comparison."""
    triangle_polygons = shapely.polygons(nodes_xy_um[triangles])
    points_um = shapely.points(source_grid_xy_m * 1.0e6)
    pairs = STRtree(triangle_polygons).query(points_um, predicate="covered_by")
    result = np.full(len(points_um), -1, dtype=np.int64)
    if pairs.size:
        point_indices, triangle_indices = pairs
        order = np.lexsort((triangle_indices, point_indices))
        point_indices = point_indices[order]
        triangle_indices = triangle_indices[order]
        first = np.concatenate(([True], point_indices[1:] != point_indices[:-1]))
        result[point_indices[first]] = triangle_indices[first]
    return result
