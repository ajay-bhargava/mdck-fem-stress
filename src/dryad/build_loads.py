#!/usr/bin/env python3
"""Build Stage 5 conservative published-traction FEM load artifacts."""

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
import scipy
import shapely
import yaml
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union

from dryad.load_validate import validate_load_artifacts
from dryad.mesh import mesh_domain
from dryad.traction_mapping import (
    MappingResult,
    map_traction_frames,
    quadrature_rule,
    source_grid_to_element_index,
    transform_source_grid_to_fem_m,
    transform_source_vectors_to_fem,
)


LOGGER = logging.getLogger("dryad.build_loads")
DEFAULT_RULE = "degree5"
REPRESENTATIVE_FRAMES = (0, 48, 96)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--positions", nargs="+")
    parser.add_argument("--frames", nargs="+", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
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


def stage3_selected_domain(
    position_dir: Path, mesh_summary: dict[str, Any]
) -> Polygon | MultiPolygon:
    geojson = read_json(position_dir / "geometry/vector/domain.geojson")
    geometry = shape(geojson["features"][0]["geometry"])
    components = [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)
    components = sorted(components, key=lambda item: item.area, reverse=True)
    selected = [
        component
        for component, policy in zip(
            components, mesh_summary["component_policy"]["components"], strict=True
        )
        if policy["meshed"]
    ]
    return unary_union(selected)


def _series(values: list[float | None]) -> dict[str, float | None]:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(array):
        return {"median": None, "maximum": None}
    return {"median": float(np.median(array)), "maximum": float(np.max(array))}


def _metric_by_frame(result: MappingResult) -> dict[int, dict[str, Any]]:
    return {int(metric["frame"]): metric for metric in result.metrics}


def quadrature_convergence(
    *,
    nodes_xy_um: np.ndarray,
    triangles: np.ndarray,
    x_grid_fem_m: np.ndarray,
    y_grid_fem_m: np.ndarray,
    tx_fem_pa: np.ndarray,
    ty_fem_pa: np.ndarray,
    moment_reference_xy_m: np.ndarray,
    frames: np.ndarray,
) -> list[dict[str, Any]]:
    results: dict[str, MappingResult] = {}
    for name in ("centroid", "degree2", "degree5", "degree7"):
        results[name] = map_traction_frames(
            nodes_xy_um=nodes_xy_um,
            triangles=triangles.copy(),
            x_grid_fem_m=x_grid_fem_m,
            y_grid_fem_m=y_grid_fem_m,
            tx_fem_pa=tx_fem_pa,
            ty_fem_pa=ty_fem_pa,
            frame_indices=frames,
            moment_reference_xy_m=moment_reference_xy_m,
            rule=quadrature_rule(name),
        )
    reference = results["degree7"]
    reference_metrics = _metric_by_frame(reference)
    output = []
    labels = {
        "centroid": "centroid_sampling",
        "degree2": "low_order",
        "degree5": "default",
        "degree7": "higher_order_reference",
    }
    for name, result in results.items():
        frame_rows = []
        metrics = _metric_by_frame(result)
        for local_index, frame in enumerate(frames):
            current = metrics[int(frame)]
            high = reference_metrics[int(frame)]
            frame_rows.append(
                {
                    "frame": int(frame),
                    "source_fx_N": current["source_fx_N"],
                    "source_fy_N": current["source_fy_N"],
                    "source_moment_Nm": current["source_moment_Nm"],
                    "force_difference_vs_degree7_N": float(
                        np.hypot(
                            current["source_fx_N"] - high["source_fx_N"],
                            current["source_fy_N"] - high["source_fy_N"],
                        )
                    ),
                    "moment_difference_vs_degree7_Nm": abs(
                        current["source_moment_Nm"] - high["source_moment_Nm"]
                    ),
                    "element_average_vector_rms_difference_vs_degree7_Pa": float(
                        np.sqrt(
                            np.mean(
                                (result.tx_element_pa[local_index] - reference.tx_element_pa[local_index]) ** 2
                                + (result.ty_element_pa[local_index] - reference.ty_element_pa[local_index]) ** 2
                            )
                        )
                    ),
                    "nodal_load_l2_difference_vs_degree7_N": float(
                        np.sqrt(
                            np.sum(
                                (result.fx_node_n[local_index] - reference.fx_node_n[local_index]) ** 2
                                + (result.fy_node_n[local_index] - reference.fy_node_n[local_index]) ** 2
                            )
                        )
                    ),
                }
            )
        output.append(
            {
                "case": labels[name],
                "rule": quadrature_rule(name).name,
                "polynomial_order": quadrature_rule(name).polynomial_order,
                "points_per_element": len(quadrature_rule(name).weights),
                "runtime_seconds": result.runtime_seconds,
                "frames": frame_rows,
            }
        )
    return output


def mesh_resolution_sensitivity(
    *,
    domain: Polygon | MultiPolygon,
    canonical_nodes_xy_um: np.ndarray,
    canonical_triangles: np.ndarray,
    mesh_config: dict[str, Any],
    x_grid_fem_m: np.ndarray,
    y_grid_fem_m: np.ndarray,
    tx_fem_pa: np.ndarray,
    ty_fem_pa: np.ndarray,
    moment_reference_xy_m: np.ndarray,
    frames: np.ndarray,
) -> list[dict[str, Any]]:
    spacing = float(mesh_config["traction_grid_spacing_um"])
    mapped_results: dict[float, MappingResult] = {}
    mesh_counts: dict[float, tuple[int, int]] = {}
    for factor in (1.0, 0.5, 0.25):
        if math.isclose(factor, float(mesh_config["traction_spacing_factor"])):
            nodes = canonical_nodes_xy_um
            triangles = canonical_triangles
        else:
            generated = mesh_domain(
                domain,
                target_edge_length_um=spacing * factor,
                minimum_component_area_um2=float(mesh_config["minimum_component_area_um2"]),
                boundary_simplify_tolerance_um=float(mesh_config["boundary_simplify_tolerance_um"]),
                boundary_target_edge_length_um=(
                    spacing * factor * float(mesh_config["boundary_refinement"]["maximum_segment_factor"])
                ),
                minimum_angle_degrees=float(mesh_config["minimum_angle_degrees"]),
            )
            nodes = generated.nodes_xy_um
            triangles = generated.triangles
        mesh_counts[factor] = (len(nodes), len(triangles))
        mapped_results[factor] = map_traction_frames(
            nodes_xy_um=nodes,
            triangles=triangles.copy(),
            x_grid_fem_m=x_grid_fem_m,
            y_grid_fem_m=y_grid_fem_m,
            tx_fem_pa=tx_fem_pa,
            ty_fem_pa=ty_fem_pa,
            frame_indices=frames,
            moment_reference_xy_m=moment_reference_xy_m,
            rule=quadrature_rule(DEFAULT_RULE),
        )
    default_metrics = _metric_by_frame(mapped_results[0.5])
    output = []
    for factor in (1.0, 0.5, 0.25):
        result = mapped_results[factor]
        metrics = _metric_by_frame(result)
        frame_rows = []
        for frame in frames:
            current = metrics[int(frame)]
            default = default_metrics[int(frame)]
            frame_rows.append(
                {
                    "frame": int(frame),
                    "source_fx_N": current["source_fx_N"],
                    "source_fy_N": current["source_fy_N"],
                    "mapped_fx_N": current["fem_fx_N"],
                    "mapped_fy_N": current["fem_fy_N"],
                    "source_moment_Nm": current["source_moment_Nm"],
                    "mapped_moment_Nm": current["fem_moment_Nm"],
                    "integrated_traction_magnitude_N": current[
                        "source_integrated_traction_magnitude_N"
                    ],
                    "mapping_force_error_N": current["absolute_force_mapping_error_N"],
                    "mapping_moment_error_Nm": current["absolute_moment_mapping_error_Nm"],
                    "source_force_difference_vs_default_N": float(
                        np.hypot(
                            current["source_fx_N"] - default["source_fx_N"],
                            current["source_fy_N"] - default["source_fy_N"],
                        )
                    ),
                    "source_moment_difference_vs_default_Nm": abs(
                        current["source_moment_Nm"] - default["source_moment_Nm"]
                    ),
                }
            )
        output.append(
            {
                "traction_spacing_factor": factor,
                "target_edge_length_um": spacing * factor,
                "node_count": mesh_counts[factor][0],
                "element_count": mesh_counts[factor][1],
                "runtime_seconds": result.runtime_seconds,
                "frames": frame_rows,
            }
        )
    return output


def build_position(
    position_dir: Path,
    *,
    requested_frames: list[int] | None,
    overwrite: bool,
) -> dict[str, Any]:
    position = position_dir.name
    output_dir = position_dir / "fem/loads"
    if output_dir.exists() and not overwrite:
        issues, summary = validate_load_artifacts(position_dir, require_complete=requested_frames is None)
        if issues:
            raise ValueError("; ".join(issues))
        LOGGER.info("%s: loads exist; validated without overwrite", position)
        return summary

    metadata = read_yaml(position_dir / "metadata.yaml")
    transform = read_json(position_dir / "diagnostics/coordinate_transform.json")
    mesh_summary = read_json(position_dir / "fem/mesh/mesh_summary.json")
    mesh_config = read_yaml(position_dir / "fem/mesh/mesh_config.yaml")
    with np.load(position_dir / "fem/mesh/mesh.npz", allow_pickle=False) as data:
        nodes_xy_um = data["nodes_xy_um"]
        triangles = data["triangles"]
        triangle_area_um2 = data["triangle_area_um2"]
        component_ids = data["component_id_per_triangle"]
    with np.load(position_dir / "traction/published.npz", allow_pickle=False) as data:
        x_source = data["x"]
        y_source = data["y"]
        tx_source = data["tx"]
        ty_source = data["ty"]
    if tx_source.shape != ty_source.shape or tx_source.shape[:2] != x_source.shape or x_source.shape != y_source.shape:
        raise ValueError("published traction x/y/tx/ty dimensions are inconsistent")
    frame_count = tx_source.shape[-1]
    expected_frames = int(metadata["imaging"]["n_frames"])
    if frame_count != expected_frames:
        raise ValueError(f"published traction has {frame_count} frames; expected {expected_frames}")
    if requested_frames is None:
        frame_indices = np.arange(frame_count, dtype=np.int64)
    else:
        invalid = [frame for frame in requested_frames if frame < 0 or frame >= frame_count]
        if invalid:
            raise ValueError(f"frames outside 0..{frame_count - 1}: {invalid}")
        frame_indices = np.asarray(sorted(set(requested_frames)), dtype=np.int64)
    complete = np.array_equal(frame_indices, np.arange(frame_count))

    pixel_size_m = float(metadata["imaging"]["pixel_size_m"])
    image_height = int(transform["image_coordinates"]["row_range_zero_based"][1]) + 1
    x_grid_fem_m, y_grid_fem_m = transform_source_grid_to_fem_m(
        x_source, y_source, pixel_size_m=pixel_size_m, image_height=image_height
    )
    # Semantic action/reaction sign is unresolved. +1 preserves the published
    # vector direction after transforming its coordinate basis.
    tx_fem_pa, ty_fem_pa = transform_source_vectors_to_fem(
        tx_source, ty_source, semantic_sign=1.0
    )
    domain = stage3_selected_domain(position_dir, mesh_summary)
    moment_reference_xy_m = np.asarray(domain.centroid.coords[0]) * 1.0e-6
    rule = quadrature_rule(DEFAULT_RULE)
    result = map_traction_frames(
        nodes_xy_um=nodes_xy_um,
        triangles=triangles.copy(),
        x_grid_fem_m=x_grid_fem_m,
        y_grid_fem_m=y_grid_fem_m,
        tx_fem_pa=tx_fem_pa,
        ty_fem_pa=ty_fem_pa,
        frame_indices=frame_indices,
        moment_reference_xy_m=moment_reference_xy_m,
        rule=rule,
    )
    metrics = pd.DataFrame(result.metrics)
    metrics.insert(0, "position", position)

    source_grid_xy_m = np.column_stack((x_grid_fem_m.ravel(), y_grid_fem_m.ravel()))
    grid_element_index = source_grid_to_element_index(
        nodes_xy_um, triangles, source_grid_xy_m
    )
    representative = np.asarray(
        [frame for frame in REPRESENTATIVE_FRAMES if frame in set(frame_indices.tolist())],
        dtype=np.int64,
    )
    quadrature_test = None
    mesh_sensitivity = None
    if position == "pos01" and len(representative):
        quadrature_test = quadrature_convergence(
            nodes_xy_um=nodes_xy_um,
            triangles=triangles,
            x_grid_fem_m=x_grid_fem_m,
            y_grid_fem_m=y_grid_fem_m,
            tx_fem_pa=tx_fem_pa,
            ty_fem_pa=ty_fem_pa,
            moment_reference_xy_m=moment_reference_xy_m,
            frames=representative,
        )
        mesh_sensitivity = mesh_resolution_sensitivity(
            domain=domain,
            canonical_nodes_xy_um=nodes_xy_um,
            canonical_triangles=triangles,
            mesh_config=mesh_config,
            x_grid_fem_m=x_grid_fem_m,
            y_grid_fem_m=y_grid_fem_m,
            tx_fem_pa=tx_fem_pa,
            ty_fem_pa=ty_fem_pa,
            moment_reference_xy_m=moment_reference_xy_m,
            frames=representative,
        )

    mapping_config = {
        "source_traction_units": "Pa",
        "source_traction_units_status": (
            "inferred_from_SI_substrate_parameters_and_stress_dimension; not explicitly stated in README"
        ),
        "internal_length_units": "m",
        "internal_traction_units": "Pa = N/m^2",
        "internal_force_units": "N",
        "internal_moment_units": "N*m",
        "visualization_length_units": "um",
        "coordinate_transform": {
            "source_x_to_fem_m": "(x_source - 1) * pixel_size_m",
            "source_y_to_fem_m": "(image_height - y_source) * pixel_size_m",
            "source_coordinate_origin": "MATLAB one-based, top-left",
            "fem_coordinate_origin": "bottom-left pixel center",
        },
        "vector_transform": {
            "tx_fem": "+tx_source",
            "ty_fem": "-ty_source",
            "reason": "reflection of the y coordinate basis",
        },
        "published_traction_semantics": "cell-substrate tractions; action direction unresolved",
        "fem_load_semantics": (
            "coordinate-transformed published vector direction; action/reaction sign not resolved"
        ),
        "traction_to_fem_sign": None,
        "stored_semantic_sign": 1,
        "alternative_substrate_on_cell_sign": -1,
        "interpolation_method": "bilinear on regular published traction grid; no extrapolation",
        "element_traction_definition": "quadrature area average over each P1 triangle",
        "quadrature_rule": rule.name,
        "quadrature_order": rule.polynomial_order,
        "quadrature_points_per_element": len(rule.weights),
        "moment_reference": {
            "definition": "area centroid of Stage 4 selected continuous-monolayer domain",
            "x_m": float(moment_reference_xy_m[0]),
            "y_m": float(moment_reference_xy_m[1]),
        },
        "dof_ordering": "interleaved conceptual ordering [fx1,fy1,fx2,fy2,...]; stored as separate frame-by-node arrays",
        "mesh_source": "../mesh/mesh.npz",
        "expected_frame_count": frame_count,
        "processed_frame_indices": frame_indices.tolist(),
        "complete_canonical_time_series": bool(complete),
        "software_versions": {
            "numpy": version("numpy"),
            "scipy": scipy.__version__,
            "shapely": shapely.__version__,
            "meshio": version("meshio"),
        },
        "equilibrium_correction_applied": False,
        "mechanical_constraints_applied": False,
    }

    force_errors = metrics["force_mapping_error_over_integrated_traction"].tolist()
    moment_errors = metrics["moment_mapping_error_over_integrated_abs_moment"].tolist()
    force_fractions = metrics["source_net_force_fraction"].tolist()
    moment_fractions = metrics["source_net_moment_fraction"].tolist()
    summary = {
        "schema_version": 1,
        "position": position,
        "status": "success",
        "complete_canonical_time_series": bool(complete),
        "expected_frame_count": frame_count,
        "processed_frame_count": len(frame_indices),
        "processed_frame_indices": frame_indices.tolist(),
        "traction_shape": list(tx_source.shape),
        "mesh_node_count": len(nodes_xy_um),
        "mesh_element_count": len(triangles),
        "units_and_semantics": {
            "source_traction_units": mapping_config["source_traction_units"],
            "source_traction_units_status": mapping_config["source_traction_units_status"],
            "published_traction_semantics": mapping_config["published_traction_semantics"],
            "fem_load_semantics": mapping_config["fem_load_semantics"],
            "traction_to_fem_sign": None,
            "stored_semantic_sign": 1,
            "alternative_interpretation": "multiply all stored traction/load vectors by -1",
        },
        "coordinate_and_vector_transform": {
            **mapping_config["coordinate_transform"],
            **mapping_config["vector_transform"],
        },
        "mapping": {
            "interpolation_method": mapping_config["interpolation_method"],
            "quadrature_rule": rule.name,
            "quadrature_order": rule.polynomial_order,
            "quadrature_points_per_element": len(rule.weights),
            "quadrature_points_total_per_frame": result.quadrature_points_total,
            "quadrature_points_outside_traction_support": result.quadrature_points_outside,
            "fraction_outside_traction_support": (
                result.quadrature_points_outside / result.quadrature_points_total
            ),
            "runtime_seconds": result.runtime_seconds,
            "estimated_persistent_array_bytes": int(
                result.tx_element_pa.nbytes
                + result.ty_element_pa.nbytes
                + result.fx_node_n.nbytes
                + result.fy_node_n.nbytes
            ),
        },
        "moment_reference": mapping_config["moment_reference"],
        "mapping_conservation": {
            "force_error_over_integrated_traction": _series(force_errors),
            "moment_error_over_integrated_abs_moment": _series(moment_errors),
            "absolute_force_error_N": _series(
                metrics["absolute_force_mapping_error_N"].tolist()
            ),
            "absolute_moment_error_Nm": _series(
                metrics["absolute_moment_mapping_error_Nm"].tolist()
            ),
            "worst_force_mapping_frame": int(
                metrics.loc[
                    metrics["force_mapping_error_over_integrated_traction"].idxmax(), "frame"
                ]
            ),
            "worst_moment_mapping_frame": int(
                metrics.loc[
                    metrics["moment_mapping_error_over_integrated_abs_moment"].idxmax(), "frame"
                ]
            ),
        },
        "experimental_equilibrium": {
            "net_force_fraction": _series(force_fractions),
            "net_moment_fraction": _series(moment_fractions),
            "worst_force_balance_frame": int(
                metrics.loc[metrics["source_net_force_fraction"].idxmax(), "frame"]
            ),
            "worst_moment_balance_frame": int(
                metrics.loc[metrics["source_net_moment_fraction"].idxmax(), "frame"]
            ),
            "correction_applied": False,
        },
        "quadrature_convergence_pos01": quadrature_test,
        "mesh_resolution_sensitivity_pos01": mesh_sensitivity,
        "vtu_representative_frame": int(frame_indices[0]),
        "warnings": [
            "Published traction action direction is not documented; stored sign preserves the published vector after coordinate-basis transformation.",
            "Traction units are treated as Pa from dimensional reconstruction context but are not explicitly stated in the source README.",
            "Interpolation represents the measured bilinear field on a finer mesh and does not increase experimental spatial resolution.",
        ],
        "stage6_input": {
            "numerically_conservative": True,
            "raw_loads_equilibrium_correction_applied": False,
            "semantic_sign_resolved": False,
            "source_units_explicitly_documented": False,
        },
    }

    output_parent = position_dir / "fem"
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{position}.loads.", dir=output_parent))
    try:
        with (temporary_dir / "traction_on_elements.npz").open("wb") as handle:
            np.savez(
                handle,
                tx_element=result.tx_element_pa,
                ty_element=result.ty_element_pa,
                traction_magnitude_element=np.hypot(
                    result.tx_element_pa, result.ty_element_pa
                ).astype(np.float32),
                frame_indices=frame_indices,
                element_area_m2=triangle_area_um2 * 1.0e-12,
                source_grid_element_index=grid_element_index.reshape(x_source.shape),
                source_grid_x_fem_m=x_grid_fem_m,
                source_grid_y_fem_m=y_grid_fem_m,
                coordinate_components=np.asarray("FEM_x_right_y_up"),
                units=np.asarray("Pa"),
                definition=np.asarray("quadrature_area_average"),
            )
        with (temporary_dir / "nodal_forces.npz").open("wb") as handle:
            np.savez(
                handle,
                fx_node=result.fx_node_n,
                fy_node=result.fy_node_n,
                frame_indices=frame_indices,
                nodes_xy_m=nodes_xy_um * 1.0e-6,
                dof_ordering=np.asarray("separate_frame_by_node; conceptual_interleaved_xy"),
                units=np.asarray("N"),
                constraints_applied=np.asarray(False),
            )
        metrics.to_csv(temporary_dir / "frame_metrics.csv", index=False)
        write_json(temporary_dir / "load_summary.json", summary)
        with (temporary_dir / "mapping_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(mapping_config, handle, sort_keys=False)

        representative_index = 0
        tx_vtu = result.tx_element_pa[representative_index].astype(np.float64)
        ty_vtu = result.ty_element_pa[representative_index].astype(np.float64)
        fx_vtu = result.fx_node_n[representative_index]
        fy_vtu = result.fy_node_n[representative_index]
        points_3d = np.column_stack((nodes_xy_um, np.zeros(len(nodes_xy_um))))
        mapped_vtu = meshio.Mesh(
            points_3d,
            [("triangle", triangles)],
            point_data={
                "fx_N": fx_vtu,
                "fy_N": fy_vtu,
                "nodal_force_magnitude_N": np.hypot(fx_vtu, fy_vtu),
            },
            cell_data={
                "tx_Pa": [tx_vtu],
                "ty_Pa": [ty_vtu],
                "traction_magnitude_Pa": [np.hypot(tx_vtu, ty_vtu)],
                "component_id": [component_ids],
            },
        )
        mapped_vtu.write(temporary_dir / "traction_mapped.vtu", binary=True)
        issues, _ = validate_load_artifacts(
            position_dir,
            loads_dir_override=temporary_dir,
            require_complete=complete,
        )
        if issues:
            raise ValueError("generated load mapping failed validation: " + "; ".join(issues))
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return summary


def main() -> int:
    args = parse_args()
    if args.validate_only and args.overwrite:
        raise SystemExit("--validate-only and --overwrite cannot be combined")
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
                issues, summary = validate_load_artifacts(
                    position_dir, require_complete=args.frames is None
                )
                if issues:
                    raise ValueError("; ".join(issues))
                LOGGER.info(
                    "%s: valid frames=%s", position_dir.name, summary["processed_frame_count"]
                )
            else:
                summary = build_position(
                    position_dir,
                    requested_frames=args.frames,
                    overwrite=args.overwrite,
                )
                LOGGER.info(
                    "%s: frames=%s force_error_max=%s",
                    position_dir.name,
                    summary["processed_frame_count"],
                    summary["mapping_conservation"][
                        "force_error_over_integrated_traction"
                    ]["maximum"],
                )
            successful += 1
        except Exception:
            LOGGER.exception("%s: load mapping failed", position_dir.name)
            failed += 1
    print(f"Load positions: {len(positions)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
