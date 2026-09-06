#!/usr/bin/env python3
"""Build canonical Stage 4 constrained continuum meshes."""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import meshio
import numpy as np
import pandas as pd
import shapely
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, shape
from shapely.ops import unary_union

from dryad.mesh import (
    ConstrainedMesh,
    distribution,
    mesh_domain,
    polygon_components,
    triangle_quality,
    unique_mesh_edges,
)
from dryad.mesh_validate import validate_mesh_artifacts


LOGGER = logging.getLogger("dryad.build_mesh")
DEFAULT_TRACTION_SPACING_FACTOR = 0.5
DEFAULT_MINIMUM_COMPONENT_AREA_UM2 = 1.0
DEFAULT_BOUNDARY_EDGE_FACTOR = 0.5
DEFAULT_MINIMUM_ANGLE_DEGREES = 25.0
SMALL_ANGLE_WARNING_DEGREES = 20.0
EXTREME_ASPECT_RATIO = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--positions", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--mesh-size-um", type=float)
    parser.add_argument(
        "--traction-spacing-factor",
        type=float,
        default=DEFAULT_TRACTION_SPACING_FACTOR,
        help="target edge length / measured traction-grid spacing (default: 0.5)",
    )
    parser.add_argument(
        "--minimum-component-area-um2",
        type=float,
        default=DEFAULT_MINIMUM_COMPONENT_AREA_UM2,
        help="explicit disconnected-component meshing threshold (default: 1.0)",
    )
    parser.add_argument(
        "--boundary-simplify-um",
        type=float,
        help="boundary simplification tolerance; default 0 preserves Stage 3 exactly",
    )
    parser.add_argument(
        "--boundary-edge-factor",
        type=float,
        default=DEFAULT_BOUNDARY_EDGE_FACTOR,
        help="maximum boundary segment / interior target edge length (default: 0.5)",
    )
    parser.add_argument(
        "--minimum-angle-deg",
        type=float,
        default=DEFAULT_MINIMUM_ANGLE_DEGREES,
    )
    return parser.parse_args()


def discover_positions(data_root: Path, requested: list[str] | None) -> list[Path]:
    available = {
        path.name: path
        for path in data_root.iterdir()
        if path.is_dir() and path.name.startswith("pos") and path.name[3:].isdigit()
    }
    names = requested or list(available)
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"positions not found: {', '.join(missing)}")
    return [available[name] for name in sorted(set(names), key=lambda value: int(value[3:]))]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def measured_traction_spacing_um(position_dir: Path, pixel_size_um: float) -> dict[str, float]:
    with np.load(position_dir / "diagnostics/grid_alignment.npz", allow_pickle=False) as data:
        x = data["traction_x"].astype(np.float64)
        y = data["traction_y"].astype(np.float64)
    dx_px = float(np.median(np.diff(x, axis=1)))
    dy_px = float(np.median(np.diff(y, axis=0)))
    if dx_px <= 0 or dy_px <= 0:
        raise ValueError("traction grid spacing must be positive")
    return {
        "dx_pixels": dx_px,
        "dy_pixels": dy_px,
        "dx_um": dx_px * pixel_size_um,
        "dy_um": dy_px * pixel_size_um,
        "representative_um": math.sqrt(dx_px * dy_px) * pixel_size_um,
    }


def load_fem_domain(position_dir: Path) -> Polygon | MultiPolygon:
    domain = read_json(position_dir / "geometry/vector/domain.geojson")
    if domain.get("metadata", {}).get("coordinate_space") != "FEM":
        raise ValueError("Stage 3 domain is not in FEM coordinate space")
    if domain.get("metadata", {}).get("units") != "micrometers":
        raise ValueError("Stage 3 domain units are not micrometers")
    features = domain.get("features", [])
    if len(features) != 1:
        raise ValueError("Stage 3 domain GeoJSON must contain one feature")
    geometry = shape(features[0]["geometry"])
    if not isinstance(geometry, (Polygon, MultiPolygon)) or not geometry.is_valid:
        raise ValueError("Stage 3 FEM domain is not valid polygonal geometry")
    return geometry


def mesh_connectivity(node_count: int, triangles: np.ndarray) -> tuple[int, int]:
    edges, _ = unique_mesh_edges(triangles)
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(node_count, node_count))
    count, _ = connected_components(graph, directed=False)
    return int(count), int(len(edges))


def boundary_comparison(
    mesh: ConstrainedMesh,
    original_target_domain: Polygon | MultiPolygon,
) -> dict[str, float]:
    mesh_lines = MultiLineString(
        [
            LineString(mesh.nodes_xy_um[edge])
            for edge in mesh.boundary_edges
        ]
    )
    original_boundary = original_target_domain.boundary
    hausdorff = float(original_boundary.hausdorff_distance(mesh_lines))
    boundary_nodes = np.unique(mesh.boundary_edges)
    mesh_points = shapely.points(mesh.nodes_xy_um[boundary_nodes])
    mesh_to_original = shapely.distance(mesh_points, original_boundary)
    original_coordinates = []
    for component in polygon_components(original_target_domain):
        original_coordinates.append(np.asarray(component.exterior.coords))
        original_coordinates.extend(np.asarray(interior.coords) for interior in component.interiors)
    original_points = shapely.points(np.concatenate(original_coordinates))
    original_to_mesh = shapely.distance(original_points, mesh_lines)
    combined = np.concatenate((mesh_to_original, original_to_mesh))
    return {
        "hausdorff_distance_um": hausdorff,
        "maximum_boundary_deviation_um": float(np.max(combined)),
        "mean_boundary_deviation_um": float(np.mean(combined)),
    }


def containment_counts(
    mesh: ConstrainedMesh,
) -> dict[str, int | float]:
    meshing_domain = unary_union(mesh.meshed_polygons)
    triangle_coordinates = mesh.nodes_xy_um[mesh.triangles]
    triangle_polygons = shapely.polygons(triangle_coordinates)
    centroids = shapely.points(np.mean(triangle_coordinates, axis=1))
    tolerance_um = 1.0e-10
    # Triangle boundary coordinates are constrained to the PSLG. A tiny absolute
    # tolerance avoids classifying floating-point boundary coincidence as crossing.
    containment_domain = meshing_domain.buffer(tolerance_um)
    centroid_inside = shapely.covers(containment_domain, centroids)
    polygon_inside = shapely.covers(containment_domain, triangle_polygons)
    return {
        "containment_tolerance_um": tolerance_um,
        "centroids_outside_domain": int(np.sum(~centroid_inside)),
        "triangles_crossing_outside_domain": int(np.sum(~polygon_inside)),
    }


def quality_frame(
    quality: dict[str, np.ndarray], component_ids: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "element_id": np.arange(len(component_ids), dtype=np.int64),
            "component_id": component_ids,
            "area_um2": quality["area_um2"],
            "minimum_angle_degrees": quality["minimum_angle_degrees"],
            "maximum_angle_degrees": quality["maximum_angle_degrees"],
            "aspect_ratio": quality["aspect_ratio"],
            "edge_length_min_um": quality["edge_length_min_um"],
            "edge_length_max_um": quality["edge_length_max_um"],
            "edge_length_mean_um": quality["edge_length_mean_um"],
            "quality_metric": quality["quality_metric"],
            "orientation_was_corrected": quality["orientation_was_corrected"],
        }
    )


def convergence_case(
    domain: Polygon | MultiPolygon,
    *,
    factor: float,
    traction_spacing_um: float,
    minimum_component_area_um2: float,
    boundary_simplify_tolerance_um: float,
    minimum_angle_degrees: float,
) -> dict[str, float | int]:
    target = traction_spacing_um * factor
    mesh = mesh_domain(
        domain,
        target_edge_length_um=target,
        minimum_component_area_um2=minimum_component_area_um2,
        boundary_simplify_tolerance_um=boundary_simplify_tolerance_um,
        boundary_target_edge_length_um=target * DEFAULT_BOUNDARY_EDGE_FACTOR,
        minimum_angle_degrees=minimum_angle_degrees,
    )
    quality = triangle_quality(mesh.nodes_xy_um, mesh.triangles)
    original_area = sum(
        record["area_um2"] for record in mesh.component_policy if record["meshed"]
    )
    mesh_area = float(np.sum(quality["area_um2"]))
    all_edges, _ = unique_mesh_edges(mesh.triangles)
    edge_lengths = np.linalg.norm(
        mesh.nodes_xy_um[all_edges[:, 1]] - mesh.nodes_xy_um[all_edges[:, 0]], axis=1
    )
    return {
        "traction_spacing_factor": factor,
        "target_edge_length_um": target,
        "node_count": len(mesh.nodes_xy_um),
        "element_count": len(mesh.triangles),
        "median_edge_length_um": float(np.median(edge_lengths)),
        "minimum_angle_degrees": float(np.min(quality["minimum_angle_degrees"])),
        "median_quality_metric": float(np.median(quality["quality_metric"])),
        "relative_area_error": abs(mesh_area - original_area) / original_area,
    }


def create_summary(
    *,
    position: str,
    mesh: ConstrainedMesh,
    quality: dict[str, np.ndarray],
    original_domain: Polygon | MultiPolygon,
    traction_spacing: dict[str, float],
    target_edge_length_um: float,
    containment: dict[str, int],
    boundary: dict[str, float],
    config: dict[str, Any],
    convergence_preview: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    original_total_area = float(original_domain.area)
    meshed_original_area = float(
        sum(item["area_um2"] for item in mesh.component_policy if item["meshed"])
    )
    excluded_area = original_total_area - meshed_original_area
    mesh_area = float(np.sum(quality["area_um2"]))
    absolute_error = abs(mesh_area - meshed_original_area)
    relative_error = absolute_error / meshed_original_area
    all_edges, _ = unique_mesh_edges(mesh.triangles)
    edge_lengths = np.linalg.norm(
        mesh.nodes_xy_um[all_edges[:, 1]] - mesh.nodes_xy_um[all_edges[:, 0]], axis=1
    )
    boundary_nodes = np.unique(mesh.boundary_edges)
    connected_count, unique_edge_count = mesh_connectivity(len(mesh.nodes_xy_um), mesh.triangles)
    zero_or_negative = int(np.sum(quality["signed_area_um2"] <= 0))
    small_angles = int(np.sum(quality["minimum_angle_degrees"] < SMALL_ANGLE_WARNING_DEGREES))
    extreme_aspect = int(np.sum(quality["aspect_ratio"] > EXTREME_ASPECT_RATIO))
    median_edge = float(np.median(edge_lengths))
    triangle_area_median = float(np.median(quality["area_um2"]))
    representative_spacing = traction_spacing["representative_um"]
    return {
        "schema_version": 1,
        "position": position,
        "status": "success",
        "canonical_representation": "mesh.npz; mesh.vtu is the canonical VTK visualization exchange",
        "coordinate_space": {"name": "FEM", "units": "micrometers", "x": "right", "y": "up"},
        "meshing_backend": {
            "name": "MeshPy Triangle constrained PSLG",
            "meshpy_version": version("meshpy"),
        },
        "mesh": {
            "node_count": len(mesh.nodes_xy_um),
            "element_count": len(mesh.triangles),
            "boundary_node_count": len(boundary_nodes),
            "boundary_edge_count": len(mesh.boundary_edges),
            "unique_edge_count": unique_edge_count,
            "connected_component_count": connected_count,
            "expected_meshed_component_count": sum(item["meshed"] for item in mesh.component_policy),
            "euler_v_minus_e_plus_f": len(mesh.nodes_xy_um) - unique_edge_count + len(mesh.triangles),
        },
        "component_policy": {
            "minimum_component_area_um2": config["minimum_component_area_um2"],
            "components": mesh.component_policy,
            "excluded_component_count": sum(not item["meshed"] for item in mesh.component_policy),
            "excluded_area_um2": excluded_area,
        },
        "boundary_processing": {
            "method": "topology-preserving simplification followed by segment densification",
            "stage3_geometry_modified": False,
            "meshing_copy_simplified": config["boundary_simplify_tolerance_um"] > 0,
            "simplify_tolerance_um": config["boundary_simplify_tolerance_um"],
            "maximum_boundary_segment_um": config["boundary_target_edge_length_um"],
            "components": mesh.boundary_modification,
        },
        "geometry_preservation": {
            "stage3_total_domain_area_um2": original_total_area,
            "intended_meshed_domain_area_um2": meshed_original_area,
            "mesh_area_um2": mesh_area,
            "excluded_area_um2": excluded_area,
            "absolute_area_error_um2": absolute_error,
            "relative_area_error": relative_error,
            **boundary,
        },
        "containment": {
            **containment,
            "invalid_element_count": zero_or_negative
            + containment["triangles_crossing_outside_domain"],
        },
        "quality": {
            "area_um2": distribution(quality["area_um2"]),
            "minimum_angle_degrees": distribution(quality["minimum_angle_degrees"]),
            "maximum_angle_degrees": distribution(quality["maximum_angle_degrees"]),
            "aspect_ratio": distribution(quality["aspect_ratio"]),
            "edge_length_min_um": distribution(quality["edge_length_min_um"]),
            "edge_length_max_um": distribution(quality["edge_length_max_um"]),
            "edge_length_mean_um": distribution(quality["edge_length_mean_um"]),
            "all_unique_edge_length_um": distribution(edge_lengths),
            "quality_metric": distribution(quality["quality_metric"]),
            "zero_or_negative_area_elements": zero_or_negative,
            "orientation_corrections": int(np.sum(quality["orientation_was_corrected"])),
            "elements_below_20_degree_minimum_angle": small_angles,
            "elements_above_aspect_ratio_5": extreme_aspect,
            "aspect_ratio_definition": "longest_edge / (2*sqrt(3)*inradius); equilateral=1",
            "quality_metric_definition": "4*sqrt(3)*area/sum(edge_length_squared); equilateral=1",
        },
        "traction_resolution": {
            **traction_spacing,
            "target_edge_length_um": target_edge_length_um,
            "target_edge_to_traction_spacing_ratio": target_edge_length_um / representative_spacing,
            "actual_median_edge_length_um": median_edge,
            "actual_median_edge_to_traction_spacing_ratio": median_edge / representative_spacing,
            "median_triangle_area_um2": triangle_area_median,
            "estimated_elements_per_traction_grid_cell": representative_spacing**2 / triangle_area_median,
            "traction_interpolation_performed": False,
        },
        "convergence_preview": convergence_preview,
        "nuclear_label_constraint": {
            "used_as_mesh_constraints": False,
            "statement": "StarDist polygons are nuclear-label geometry, not epithelial cell boundaries or FEM subdomains.",
        },
        "stage5_readiness": {
            "safe_for_traction_mapping": bool(
                zero_or_negative == 0
                and containment["centroids_outside_domain"] == 0
                and containment["triangles_crossing_outside_domain"] == 0
                and relative_error <= 1.0e-3
                and connected_count == sum(item["meshed"] for item in mesh.component_policy)
            ),
            "scope": "continuous monolayer only; no cell-resolved material partition",
        },
    }


def build_position(
    position_dir: Path,
    *,
    mesh_size_um: float | None,
    traction_spacing_factor: float,
    minimum_component_area_um2: float,
    boundary_simplify_um: float | None,
    boundary_edge_factor: float,
    minimum_angle_degrees: float,
    overwrite: bool,
) -> dict[str, Any]:
    position = position_dir.name
    mesh_dir = position_dir / "fem/mesh"
    if mesh_dir.exists() and not overwrite:
        issues, summary = validate_mesh_artifacts(position_dir)
        if issues:
            raise ValueError("; ".join(issues))
        LOGGER.info("%s: mesh exists; validated without overwrite", position)
        return summary

    metadata = read_yaml(position_dir / "metadata.yaml")
    pixel_size_um = float(metadata["imaging"]["pixel_size_um"])
    traction_spacing = measured_traction_spacing_um(position_dir, pixel_size_um)
    representative_spacing = traction_spacing["representative_um"]
    target_edge_length_um = (
        mesh_size_um if mesh_size_um is not None else representative_spacing * traction_spacing_factor
    )
    actual_spacing_factor = target_edge_length_um / representative_spacing
    simplify_tolerance = boundary_simplify_um if boundary_simplify_um is not None else 0.0
    boundary_target = target_edge_length_um * boundary_edge_factor
    domain = load_fem_domain(position_dir)
    mesh = mesh_domain(
        domain,
        target_edge_length_um=target_edge_length_um,
        minimum_component_area_um2=minimum_component_area_um2,
        boundary_simplify_tolerance_um=simplify_tolerance,
        boundary_target_edge_length_um=boundary_target,
        minimum_angle_degrees=minimum_angle_degrees,
    )
    quality = triangle_quality(mesh.nodes_xy_um, mesh.triangles)
    containment = containment_counts(mesh)
    original_target = unary_union(
        [
            component
            for component, policy in zip(
                polygon_components(domain), mesh.component_policy, strict=True
            )
            if policy["meshed"]
        ]
    )
    boundary = boundary_comparison(mesh, original_target)
    config = {
        "meshing_backend": "MeshPy Triangle constrained PSLG",
        "backend_version": version("meshpy"),
        "target_edge_length_um": target_edge_length_um,
        "traction_grid_spacing_um": representative_spacing,
        "traction_grid_dx_um": traction_spacing["dx_um"],
        "traction_grid_dy_um": traction_spacing["dy_um"],
        "traction_spacing_factor": actual_spacing_factor,
        "minimum_component_area_um2": minimum_component_area_um2,
        "boundary_refinement": {
            "enabled": True,
            "method": "densify constrained rings without moving the simplified ring",
            "maximum_segment_factor": boundary_edge_factor,
        },
        "boundary_simplify_tolerance_um": simplify_tolerance,
        "boundary_target_edge_length_um": boundary_target,
        "minimum_angle_degrees": minimum_angle_degrees,
        "maximum_triangle_area_um2": math.sqrt(3.0) * target_edge_length_um**2 / 4.0,
        "coordinate_units": "micrometers",
        "coordinate_system": "persisted Stage 3 FEM space; x right, y up",
        "nuclear_labels_used_as_constraints": False,
    }
    convergence = None
    if position == "pos01":
        convergence = [
            convergence_case(
                domain,
                factor=factor,
                traction_spacing_um=representative_spacing,
                minimum_component_area_um2=minimum_component_area_um2,
                boundary_simplify_tolerance_um=simplify_tolerance,
                minimum_angle_degrees=minimum_angle_degrees,
            )
            for factor in (1.0, 0.5, 0.25)
        ]
    summary = create_summary(
        position=position,
        mesh=mesh,
        quality=quality,
        original_domain=domain,
        traction_spacing=traction_spacing,
        target_edge_length_um=target_edge_length_um,
        containment=containment,
        boundary=boundary,
        config=config,
        convergence_preview=convergence,
    )

    temporary_parent = position_dir / "fem"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{position}.mesh.", dir=temporary_parent))
    try:
        with (temporary_dir / "mesh.npz").open("wb") as handle:
            np.savez(
                handle,
                nodes_xy_um=mesh.nodes_xy_um,
                triangles=mesh.triangles,
                triangle_area_um2=quality["area_um2"],
                triangle_centroid_xy_um=quality["centroid_xy_um"],
                boundary_node_indices=np.unique(mesh.boundary_edges),
                component_id_per_triangle=mesh.component_id_per_triangle,
            )
        with (temporary_dir / "boundary.npz").open("wb") as handle:
            np.savez(
                handle,
                boundary_edges=mesh.boundary_edges,
                boundary_component_id=mesh.boundary_component_id,
                boundary_ring_id=mesh.boundary_ring_id,
                boundary_is_hole=mesh.boundary_is_hole,
                boundary_edge_length_um=np.linalg.norm(
                    mesh.nodes_xy_um[mesh.boundary_edges[:, 1]]
                    - mesh.nodes_xy_um[mesh.boundary_edges[:, 0]],
                    axis=1,
                ),
            )
        quality_frame(quality, mesh.component_id_per_triangle).to_csv(
            temporary_dir / "mesh_quality.csv", index=False
        )
        write_json(temporary_dir / "mesh_summary.json", summary)
        with (temporary_dir / "mesh_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        points_3d = np.column_stack((mesh.nodes_xy_um, np.zeros(len(mesh.nodes_xy_um))))
        point_boundary = np.zeros(len(mesh.nodes_xy_um), dtype=np.uint8)
        point_boundary[np.unique(mesh.boundary_edges)] = 1
        vtu = meshio.Mesh(
            points=points_3d,
            cells=[("triangle", mesh.triangles)],
            point_data={
                "node_id": np.arange(len(mesh.nodes_xy_um), dtype=np.int64),
                "is_boundary": point_boundary,
            },
            cell_data={
                "element_id": [np.arange(len(mesh.triangles), dtype=np.int64)],
                "area_um2": [quality["area_um2"]],
                "component_id": [mesh.component_id_per_triangle],
                "quality_metric": [quality["quality_metric"]],
                "minimum_angle_degrees": [quality["minimum_angle_degrees"]],
                "aspect_ratio": [quality["aspect_ratio"]],
            },
        )
        vtu.write(temporary_dir / "mesh.vtu", binary=True)
        issues, _ = validate_mesh_artifacts(position_dir, mesh_dir_override=temporary_dir)
        if issues:
            raise ValueError("generated mesh failed validation: " + "; ".join(issues))
        if mesh_dir.exists():
            shutil.rmtree(mesh_dir)
        temporary_dir.replace(mesh_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return summary


def main() -> int:
    args = parse_args()
    if args.validate_only and args.overwrite:
        raise SystemExit("--validate-only and --overwrite cannot be combined")
    numeric_positive = {
        "mesh size": args.mesh_size_um,
        "traction spacing factor": args.traction_spacing_factor,
        "minimum component area": args.minimum_component_area_um2,
        "boundary edge factor": args.boundary_edge_factor,
        "minimum angle": args.minimum_angle_deg,
    }
    for name, value in numeric_positive.items():
        if value is not None and value <= 0:
            raise SystemExit(f"{name} must be positive")
    if args.boundary_simplify_um is not None and args.boundary_simplify_um < 0:
        raise SystemExit("boundary simplification tolerance cannot be negative")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        positions = discover_positions(args.data.resolve(), args.positions)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    successful = 0
    failed = 0
    for position_dir in positions:
        try:
            if args.validate_only:
                issues, summary = validate_mesh_artifacts(position_dir)
                if issues:
                    raise ValueError("; ".join(issues))
                LOGGER.info(
                    "%s: valid nodes=%s elements=%s",
                    position_dir.name,
                    summary["mesh"]["node_count"],
                    summary["mesh"]["element_count"],
                )
            else:
                summary = build_position(
                    position_dir,
                    mesh_size_um=args.mesh_size_um,
                    traction_spacing_factor=args.traction_spacing_factor,
                    minimum_component_area_um2=args.minimum_component_area_um2,
                    boundary_simplify_um=args.boundary_simplify_um,
                    boundary_edge_factor=args.boundary_edge_factor,
                    minimum_angle_degrees=args.minimum_angle_deg,
                    overwrite=args.overwrite,
                )
                LOGGER.info(
                    "%s: nodes=%s elements=%s area_error=%s",
                    position_dir.name,
                    summary["mesh"]["node_count"],
                    summary["mesh"]["element_count"],
                    summary["geometry_preservation"]["relative_area_error"],
                )
            successful += 1
        except Exception:
            LOGGER.exception("%s: mesh processing failed", position_dir.name)
            failed += 1
    print(f"Mesh positions: {len(positions)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
