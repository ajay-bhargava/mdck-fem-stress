"""Independent validation for persistent Stage 4 mesh artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import meshio
import numpy as np
import pandas as pd
import pyvista as pv
import shapely
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union


REQUIRED_FILES = (
    "mesh.npz",
    "mesh.vtu",
    "boundary.npz",
    "mesh_summary.json",
    "mesh_quality.csv",
    "mesh_config.yaml",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _stage3_domain(position_dir: Path) -> Polygon | MultiPolygon:
    data = _read_json(position_dir / "geometry/vector/domain.geojson")
    return shape(data["features"][0]["geometry"])


def _selected_domain(
    domain: Polygon | MultiPolygon, component_policy: list[dict[str, Any]]
) -> Polygon | MultiPolygon:
    components = [domain] if isinstance(domain, Polygon) else list(domain.geoms)
    components = sorted(components, key=lambda geometry: geometry.area, reverse=True)
    selected = [
        component
        for component, policy in zip(components, component_policy, strict=True)
        if policy["meshed"]
    ]
    return unary_union(selected)


def validate_mesh_artifacts(
    position_dir: Path,
    *,
    mesh_dir_override: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    mesh_dir = mesh_dir_override or position_dir / "fem/mesh"
    issues: list[str] = []
    for filename in REQUIRED_FILES:
        if not (mesh_dir / filename).is_file():
            issues.append(f"missing {filename}")
    if issues:
        return issues, {}
    try:
        summary = _read_json(mesh_dir / "mesh_summary.json")
    except Exception as exc:
        return [f"cannot read mesh summary: {exc}"], {}
    if summary.get("position") != position_dir.name:
        issues.append("mesh summary position mismatch")

    try:
        with np.load(mesh_dir / "mesh.npz", allow_pickle=False) as data:
            if any(data[name].dtype == object for name in data.files):
                issues.append("mesh NPZ contains object arrays")
            nodes = data["nodes_xy_um"]
            triangles = data["triangles"]
            areas = data["triangle_area_um2"]
            centroids = data["triangle_centroid_xy_um"]
            boundary_nodes = data["boundary_node_indices"]
            component_ids = data["component_id_per_triangle"]
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            issues.append(f"invalid node shape {nodes.shape}")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            issues.append(f"invalid triangle shape {triangles.shape}")
        if len(areas) != len(triangles) or centroids.shape != (len(triangles), 2):
            issues.append("triangle metric shape mismatch")
        if len(component_ids) != len(triangles):
            issues.append("component ID count mismatch")
        if len(triangles) and (triangles.min() < 0 or triangles.max() >= len(nodes)):
            issues.append("triangle node index out of range")
        points = nodes[triangles]
        signed = 0.5 * (
            (points[:, 1, 0] - points[:, 0, 0])
            * (points[:, 2, 1] - points[:, 0, 1])
            - (points[:, 1, 1] - points[:, 0, 1])
            * (points[:, 2, 0] - points[:, 0, 0])
        )
        if np.any(signed <= 0):
            issues.append(f"{int(np.sum(signed <= 0))} zero/negative or inverted elements")
        if not np.allclose(signed, areas, atol=1e-10, rtol=1e-10):
            issues.append("stored triangle areas disagree with coordinates")
        if len(np.unique(boundary_nodes)) != summary["mesh"]["boundary_node_count"]:
            issues.append("boundary node count disagrees with summary")
        if len(nodes) != summary["mesh"]["node_count"]:
            issues.append("node count disagrees with summary")
        if len(triangles) != summary["mesh"]["element_count"]:
            issues.append("element count disagrees with summary")
    except Exception as exc:
        issues.append(f"mesh NPZ validation failed: {exc}")
        return issues, summary

    try:
        with np.load(mesh_dir / "boundary.npz", allow_pickle=False) as data:
            boundary_edges = data["boundary_edges"]
            boundary_components = data["boundary_component_id"]
            boundary_rings = data["boundary_ring_id"]
            boundary_holes = data["boundary_is_hole"]
            boundary_lengths = data["boundary_edge_length_um"]
        expected = len(boundary_edges)
        if boundary_edges.shape != (expected, 2):
            issues.append("invalid boundary edge shape")
        if any(
            len(values) != expected
            for values in (boundary_components, boundary_rings, boundary_holes, boundary_lengths)
        ):
            issues.append("boundary metadata length mismatch")
        if expected and (boundary_edges.min() < 0 or boundary_edges.max() >= len(nodes)):
            issues.append("boundary node index out of range")
        actual_lengths = np.linalg.norm(
            nodes[boundary_edges[:, 1]] - nodes[boundary_edges[:, 0]], axis=1
        )
        if not np.allclose(actual_lengths, boundary_lengths, atol=1e-12, rtol=1e-12):
            issues.append("stored boundary lengths disagree with coordinates")
    except Exception as exc:
        issues.append(f"boundary NPZ validation failed: {exc}")

    try:
        quality = pd.read_csv(mesh_dir / "mesh_quality.csv")
        required = {
            "element_id",
            "component_id",
            "area_um2",
            "minimum_angle_degrees",
            "maximum_angle_degrees",
            "aspect_ratio",
            "edge_length_min_um",
            "edge_length_max_um",
            "edge_length_mean_um",
            "quality_metric",
        }
        if not required.issubset(quality.columns):
            issues.append("mesh quality CSV is missing required columns")
        if len(quality) != len(triangles):
            issues.append("mesh quality row count mismatch")
        if (quality["area_um2"] <= 0).any():
            issues.append("mesh quality reports non-positive areas")
    except Exception as exc:
        issues.append(f"mesh quality validation failed: {exc}")

    try:
        meshio_mesh = meshio.read(mesh_dir / "mesh.vtu")
        meshio_triangles = meshio_mesh.get_cells_type("triangle")
        if len(meshio_mesh.points) != len(nodes) or len(meshio_triangles) != len(triangles):
            issues.append("meshio VTU counts disagree with NPZ")
        pyvista_mesh = pv.read(mesh_dir / "mesh.vtu")
        if pyvista_mesh.n_points != len(nodes) or pyvista_mesh.n_cells != len(triangles):
            issues.append("PyVista VTU counts disagree with NPZ")
        for name in ("node_id", "is_boundary"):
            if name not in pyvista_mesh.point_data:
                issues.append(f"VTU missing point field {name}")
        for name in ("element_id", "area_um2", "component_id", "quality_metric"):
            if name not in pyvista_mesh.cell_data:
                issues.append(f"VTU missing cell field {name}")
    except Exception as exc:
        issues.append(f"VTU validation failed: {exc}")

    try:
        domain = _stage3_domain(position_dir)
        selected = _selected_domain(domain, summary["component_policy"]["components"])
        triangle_polygons = shapely.polygons(nodes[triangles])
        centroid_points = shapely.points(centroids)
        centroid_outside = int(np.sum(~shapely.covers(selected, centroid_points)))
        crossing = int(np.sum(~shapely.covers(selected.buffer(1.0e-9), triangle_polygons)))
        if centroid_outside:
            issues.append(f"{centroid_outside} triangle centroids outside Stage 3 domain")
        if crossing:
            issues.append(f"{crossing} triangles cross outside Stage 3 domain")
        intended_area = float(selected.area)
        relative_error = abs(float(np.sum(areas)) - intended_area) / intended_area
        stored_error = summary["geometry_preservation"]["relative_area_error"]
        if not np.isclose(relative_error, stored_error, atol=1e-10, rtol=1e-8):
            issues.append("recomputed domain/mesh area error disagrees with summary")
    except Exception as exc:
        issues.append(f"domain containment validation failed: {exc}")

    if summary.get("containment", {}).get("invalid_element_count", 1) != 0:
        issues.append("mesh summary reports invalid elements")
    if summary.get("nuclear_label_constraint", {}).get("used_as_mesh_constraints") is not False:
        issues.append("nuclear labels were not explicitly excluded as mesh constraints")
    return issues, summary
