"""Constrained continuum meshing and triangle-quality primitives for Stage 4."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import meshpy.triangle as triangle_backend
import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon


@dataclass
class ConstrainedMesh:
    nodes_xy_um: np.ndarray
    triangles: np.ndarray
    boundary_edges: np.ndarray
    boundary_component_id: np.ndarray
    boundary_ring_id: np.ndarray
    boundary_is_hole: np.ndarray
    component_id_per_triangle: np.ndarray
    component_policy: list[dict[str, Any]]
    meshed_polygons: list[Polygon]
    boundary_modification: list[dict[str, float | int]]


def polygon_components(geometry: Polygon | MultiPolygon) -> list[Polygon]:
    components = [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)
    return sorted(components, key=lambda polygon: polygon.area, reverse=True)


def densify_ring(coordinates: np.ndarray, maximum_segment_length: float) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if np.allclose(coordinates[0], coordinates[-1]):
        coordinates = coordinates[:-1]
    points: list[np.ndarray] = []
    for index in range(len(coordinates)):
        start = coordinates[index]
        stop = coordinates[(index + 1) % len(coordinates)]
        length = float(np.linalg.norm(stop - start))
        subdivisions = max(1, int(math.ceil(length / maximum_segment_length)))
        for step in range(subdivisions):
            points.append(start + (stop - start) * (step / subdivisions))
    return np.asarray(points, dtype=np.float64)


def _prepare_component(
    component: Polygon,
    *,
    simplify_tolerance_um: float,
) -> tuple[Polygon, dict[str, float | int]]:
    original = component
    prepared = (
        component.simplify(simplify_tolerance_um, preserve_topology=True)
        if simplify_tolerance_um > 0
        else component
    )
    if not isinstance(prepared, Polygon) or prepared.is_empty or not prepared.is_valid:
        prepared = original
    elif not original.covers(prepared):
        # Simplification may otherwise move a chord outside a concavity. Keep
        # the meshing copy strictly within the authoritative Stage 3 domain.
        contained = original.intersection(prepared)
        prepared = contained if isinstance(contained, Polygon) and contained.is_valid else original
    return prepared, {
        "original_area_um2": float(original.area),
        "meshing_boundary_area_um2": float(prepared.area),
        "absolute_area_change_um2": float(abs(prepared.area - original.area)),
        "relative_area_change": float(abs(prepared.area - original.area) / original.area),
        "boundary_hausdorff_distance_um": float(original.boundary.hausdorff_distance(prepared.boundary)),
        "original_boundary_vertices": int(
            len(original.exterior.coords)
            + sum(len(interior.coords) for interior in original.interiors)
        ),
        "simplified_boundary_vertices": int(
            len(prepared.exterior.coords)
            + sum(len(interior.coords) for interior in prepared.interiors)
        ),
    }


def mesh_domain(
    domain: Polygon | MultiPolygon,
    *,
    target_edge_length_um: float,
    minimum_component_area_um2: float,
    boundary_simplify_tolerance_um: float,
    boundary_target_edge_length_um: float,
    minimum_angle_degrees: float,
) -> ConstrainedMesh:
    """Mesh each disconnected component independently with constrained Triangle."""
    if target_edge_length_um <= 0 or boundary_target_edge_length_um <= 0:
        raise ValueError("mesh edge lengths must be positive")
    maximum_triangle_area = math.sqrt(3.0) * target_edge_length_um**2 / 4.0

    all_nodes: list[np.ndarray] = []
    all_triangles: list[np.ndarray] = []
    all_boundary_edges: list[np.ndarray] = []
    all_boundary_component_ids: list[np.ndarray] = []
    all_boundary_ring_ids: list[np.ndarray] = []
    all_boundary_holes: list[np.ndarray] = []
    all_triangle_components: list[np.ndarray] = []
    component_policy: list[dict[str, Any]] = []
    meshed_polygons: list[Polygon] = []
    boundary_modification: list[dict[str, float | int]] = []
    node_offset = 0
    global_ring_id = 0

    for component_id, original_component in enumerate(polygon_components(domain)):
        area = float(original_component.area)
        equivalent_radius = math.sqrt(area / math.pi)
        meshed = area >= minimum_component_area_um2
        policy: dict[str, Any] = {
            "component_id": component_id,
            "area_um2": area,
            "equivalent_radius_um": equivalent_radius,
            "meshed": meshed,
            "reason": (
                "area_at_or_above_minimum_component_threshold"
                if meshed
                else "excluded_below_minimum_component_area_um2"
            ),
        }
        component_policy.append(policy)
        if not meshed:
            continue

        component, modification = _prepare_component(
            original_component,
            simplify_tolerance_um=boundary_simplify_tolerance_um,
        )
        modification["component_id"] = component_id
        boundary_modification.append(modification)
        meshed_polygons.append(component)

        rings = [(component.exterior, False)] + [
            (interior, True) for interior in component.interiors
        ]
        points: list[tuple[float, float]] = []
        facets: list[tuple[int, int]] = []
        facet_markers: list[int] = []
        local_marker_to_global: dict[int, tuple[int, bool]] = {}
        for local_ring_index, (ring, is_hole) in enumerate(rings):
            ring_points = densify_ring(
                np.asarray(ring.coords), boundary_target_edge_length_um
            )
            start = len(points)
            points.extend(map(tuple, ring_points))
            marker = local_ring_index + 1
            local_marker_to_global[marker] = (global_ring_id, is_hole)
            global_ring_id += 1
            for index in range(len(ring_points)):
                facets.append((start + index, start + ((index + 1) % len(ring_points))))
                facet_markers.append(marker)

        mesh_info = triangle_backend.MeshInfo()
        mesh_info.set_points(points)
        mesh_info.set_facets(facets, facet_markers=facet_markers)
        hole_points = [
            tuple(Polygon(interior).representative_point().coords[0])
            for interior in component.interiors
        ]
        if hole_points:
            mesh_info.set_holes(hole_points)
        generated = triangle_backend.build(
            mesh_info,
            max_volume=maximum_triangle_area,
            min_angle=minimum_angle_degrees,
            allow_boundary_steiner=True,
            allow_volume_steiner=True,
            quality_meshing=True,
            generate_faces=True,
        )
        nodes = np.asarray(generated.points, dtype=np.float64)
        elements = np.asarray(generated.elements, dtype=np.int64)
        boundary_edges = np.asarray(generated.facets, dtype=np.int64)
        markers = np.asarray(generated.facet_markers, dtype=np.int64)
        if not len(elements):
            raise ValueError(f"component {component_id} produced no triangles")

        all_nodes.append(nodes)
        all_triangles.append(elements + node_offset)
        all_boundary_edges.append(boundary_edges + node_offset)
        all_boundary_component_ids.append(
            np.full(len(boundary_edges), component_id, dtype=np.int64)
        )
        all_triangle_components.append(
            np.full(len(elements), component_id, dtype=np.int64)
        )
        all_boundary_ring_ids.append(
            np.asarray([local_marker_to_global[int(marker)][0] for marker in markers])
        )
        all_boundary_holes.append(
            np.asarray([local_marker_to_global[int(marker)][1] for marker in markers])
        )
        node_offset += len(nodes)

    if not all_triangles:
        raise ValueError("component policy excluded the entire domain")
    return ConstrainedMesh(
        nodes_xy_um=np.concatenate(all_nodes),
        triangles=np.concatenate(all_triangles),
        boundary_edges=np.concatenate(all_boundary_edges),
        boundary_component_id=np.concatenate(all_boundary_component_ids),
        boundary_ring_id=np.concatenate(all_boundary_ring_ids),
        boundary_is_hole=np.concatenate(all_boundary_holes),
        component_id_per_triangle=np.concatenate(all_triangle_components),
        component_policy=component_policy,
        meshed_polygons=meshed_polygons,
        boundary_modification=boundary_modification,
    )


def triangle_quality(
    nodes_xy_um: np.ndarray, triangles: np.ndarray
) -> dict[str, np.ndarray]:
    points = nodes_xy_um[triangles]
    first = points[:, 1] - points[:, 0]
    second = points[:, 2] - points[:, 0]
    signed_area = 0.5 * (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    inverted = signed_area < 0
    if np.any(inverted):
        triangles[inverted, 1], triangles[inverted, 2] = (
            triangles[inverted, 2].copy(),
            triangles[inverted, 1].copy(),
        )
        points = nodes_xy_um[triangles]
        first = points[:, 1] - points[:, 0]
        second = points[:, 2] - points[:, 0]
        signed_area = 0.5 * (
            first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        )

    edge_01 = np.linalg.norm(points[:, 1] - points[:, 0], axis=1)
    edge_12 = np.linalg.norm(points[:, 2] - points[:, 1], axis=1)
    edge_20 = np.linalg.norm(points[:, 0] - points[:, 2], axis=1)
    edges = np.column_stack((edge_01, edge_12, edge_20))
    area = np.abs(signed_area)

    def angle(opposite: np.ndarray, adjacent_a: np.ndarray, adjacent_b: np.ndarray) -> np.ndarray:
        denominator = 2.0 * adjacent_a * adjacent_b
        cosine = np.divide(
            adjacent_a**2 + adjacent_b**2 - opposite**2,
            denominator,
            out=np.ones_like(opposite),
            where=denominator > 0,
        )
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    angles = np.column_stack(
        (
            angle(edge_12, edge_01, edge_20),
            angle(edge_20, edge_01, edge_12),
            angle(edge_01, edge_12, edge_20),
        )
    )
    perimeter = np.sum(edges, axis=1)
    inradius = np.divide(2.0 * area, perimeter, out=np.zeros_like(area), where=perimeter > 0)
    aspect_ratio = np.divide(
        np.max(edges, axis=1),
        2.0 * math.sqrt(3.0) * inradius,
        out=np.full_like(area, np.inf),
        where=inradius > 0,
    )
    normalized_quality = np.divide(
        4.0 * math.sqrt(3.0) * area,
        np.sum(edges**2, axis=1),
        out=np.zeros_like(area),
        where=np.sum(edges**2, axis=1) > 0,
    )
    return {
        "signed_area_um2": signed_area,
        "area_um2": area,
        "centroid_xy_um": np.mean(points, axis=1),
        "minimum_angle_degrees": np.min(angles, axis=1),
        "maximum_angle_degrees": np.max(angles, axis=1),
        "aspect_ratio": aspect_ratio,
        "edge_length_min_um": np.min(edges, axis=1),
        "edge_length_max_um": np.max(edges, axis=1),
        "edge_length_mean_um": np.mean(edges, axis=1),
        "quality_metric": normalized_quality,
        "orientation_was_corrected": inverted,
    }


def unique_mesh_edges(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.vstack(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])
    )
    edges.sort(axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique, counts


def distribution(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {key: None for key in ("min", "p01", "p05", "median", "p95", "p99", "max")}
    quantiles = np.percentile(finite, [0, 1, 5, 50, 95, 99, 100])
    return dict(zip(("min", "p01", "p05", "median", "p95", "p99", "max"), map(float, quantiles), strict=True))
