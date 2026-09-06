"""Independent validation of Stage 5 conservative load-mapping artifacts."""

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
import yaml


REQUIRED_FILES = (
    "traction_on_elements.npz",
    "nodal_forces.npz",
    "load_summary.json",
    "frame_metrics.csv",
    "mapping_config.yaml",
    "traction_mapped.vtu",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    return value


def validate_load_artifacts(
    position_dir: Path,
    *,
    loads_dir_override: Path | None = None,
    require_complete: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    loads_dir = loads_dir_override or position_dir / "fem/loads"
    issues: list[str] = []
    for filename in REQUIRED_FILES:
        if not (loads_dir / filename).is_file():
            issues.append(f"missing {filename}")
    if issues:
        return issues, {}
    try:
        summary = _read_json(loads_dir / "load_summary.json")
        config = _read_yaml(loads_dir / "mapping_config.yaml")
    except Exception as exc:
        return [f"cannot read summary/config: {exc}"], {}
    if summary.get("position") != position_dir.name:
        issues.append("load summary position mismatch")
    if require_complete and not summary.get("complete_canonical_time_series", False):
        issues.append("load artifact is partial, not a complete canonical time series")
    if config.get("traction_to_fem_sign", "missing") is not None:
        issues.append("unresolved traction semantic sign was not persisted as null")
    if config.get("equilibrium_correction_applied") is not False:
        issues.append("raw load artifact reports an equilibrium correction")
    if config.get("mechanical_constraints_applied") is not False:
        issues.append("raw load artifact reports mechanical constraints")

    try:
        with np.load(loads_dir / "traction_on_elements.npz", allow_pickle=False) as data:
            if any(data[name].dtype == object for name in data.files):
                issues.append("element traction NPZ contains object arrays")
            tx_element = data["tx_element"]
            ty_element = data["ty_element"]
            magnitude = data["traction_magnitude_element"]
            frame_indices = data["frame_indices"]
            element_area = data["element_area_m2"]
            grid_element = data["source_grid_element_index"]
            if str(data["units"]) != "Pa":
                issues.append("element traction units are not Pa")
        if tx_element.shape != ty_element.shape or tx_element.shape != magnitude.shape:
            issues.append("element traction array shapes differ")
        if tx_element.shape[0] != len(frame_indices):
            issues.append("element traction frame count mismatch")
        if tx_element.shape[1] != len(element_area):
            issues.append("element traction element count mismatch")
        if not np.all(np.isfinite(tx_element)) or not np.all(np.isfinite(ty_element)):
            issues.append("element traction arrays contain non-finite values")
        if not np.allclose(magnitude, np.hypot(tx_element, ty_element), atol=1e-5, rtol=1e-5):
            issues.append("stored element traction magnitude is inconsistent")
        if grid_element.ndim != 2:
            issues.append("source-grid element lookup is not 2-D")
    except Exception as exc:
        issues.append(f"element traction NPZ validation failed: {exc}")
        return issues, summary

    try:
        with np.load(loads_dir / "nodal_forces.npz", allow_pickle=False) as data:
            if any(data[name].dtype == object for name in data.files):
                issues.append("nodal force NPZ contains object arrays")
            fx_node = data["fx_node"]
            fy_node = data["fy_node"]
            force_frames = data["frame_indices"]
            nodes_xy_m = data["nodes_xy_m"]
            constraints_applied = bool(data["constraints_applied"])
            if str(data["units"]) != "N":
                issues.append("nodal force units are not N")
        if constraints_applied:
            issues.append("nodal forces report applied constraints")
        if fx_node.shape != fy_node.shape:
            issues.append("nodal force x/y shapes differ")
        if fx_node.shape[0] != len(force_frames) or fx_node.shape[1] != len(nodes_xy_m):
            issues.append("nodal force dimensions are inconsistent")
        if not np.array_equal(frame_indices, force_frames):
            issues.append("element and nodal frame indices differ")
        if not np.all(np.isfinite(fx_node)) or not np.all(np.isfinite(fy_node)):
            issues.append("nodal force arrays contain non-finite values")
    except Exception as exc:
        issues.append(f"nodal force NPZ validation failed: {exc}")
        return issues, summary

    try:
        metrics = pd.read_csv(loads_dir / "frame_metrics.csv")
        required_columns = {
            "position",
            "frame",
            "source_fx_N",
            "source_fy_N",
            "source_integrated_traction_magnitude_N",
            "source_net_force_fraction",
            "source_moment_Nm",
            "source_integrated_abs_moment_Nm",
            "source_net_moment_fraction",
            "fem_fx_N",
            "fem_fy_N",
            "fem_moment_Nm",
            "absolute_force_mapping_error_N",
            "force_mapping_error_over_integrated_traction",
            "absolute_moment_mapping_error_Nm",
            "moment_mapping_error_over_integrated_abs_moment",
            "quadrature_outside_support_fraction",
        }
        if not required_columns.issubset(metrics.columns):
            issues.append("frame metrics CSV is missing required columns")
        if len(metrics) != len(frame_indices):
            issues.append("frame metrics row count mismatch")
        if not np.array_equal(metrics["frame"].to_numpy(), frame_indices):
            issues.append("frame metrics indices differ from NPZ")
        fem_fx = np.sum(fx_node, axis=1)
        fem_fy = np.sum(fy_node, axis=1)
        if not np.allclose(fem_fx, metrics["fem_fx_N"], atol=1e-18, rtol=1e-10):
            issues.append("summed FEM fx disagrees with frame metrics")
        if not np.allclose(fem_fy, metrics["fem_fy_N"], atol=1e-18, rtol=1e-10):
            issues.append("summed FEM fy disagrees with frame metrics")
        reference = config["moment_reference"]
        fem_moment = np.sum(
            (nodes_xy_m[:, 0] - float(reference["x_m"])) * fy_node
            - (nodes_xy_m[:, 1] - float(reference["y_m"])) * fx_node,
            axis=1,
        )
        if not np.allclose(fem_moment, metrics["fem_moment_Nm"], atol=1e-24, rtol=1e-9):
            issues.append("recomputed FEM moment disagrees with frame metrics")
        force_error = np.hypot(
            fem_fx - metrics["source_fx_N"].to_numpy(),
            fem_fy - metrics["source_fy_N"].to_numpy(),
        )
        if not np.allclose(
            force_error,
            metrics["absolute_force_mapping_error_N"],
            atol=1e-20,
            rtol=1e-8,
        ):
            issues.append("recomputed force conservation error disagrees with metrics")
        element_fx = np.sum(tx_element * element_area[None, :], axis=1)
        element_fy = np.sum(ty_element * element_area[None, :], axis=1)
        element_force_error = np.hypot(
            element_fx - metrics["source_fx_N"].to_numpy(),
            element_fy - metrics["source_fy_N"].to_numpy(),
        ) / metrics["source_integrated_traction_magnitude_N"].to_numpy()
        if np.max(element_force_error) > 1.0e-7:
            issues.append("float32 element averages do not reproduce the source integral")
        if metrics["force_mapping_error_over_integrated_traction"].max() > 1.0e-10:
            issues.append("consistent nodal assembly fails force conservation tolerance")
        if metrics["moment_mapping_error_over_integrated_abs_moment"].max() > 1.0e-10:
            issues.append("consistent nodal assembly fails moment conservation tolerance")
        if (metrics["quadrature_outside_support_fraction"] != 0).any():
            issues.append("quadrature points outside traction support")
    except Exception as exc:
        issues.append(f"frame metrics validation failed: {exc}")

    try:
        with np.load(position_dir / "fem/mesh/mesh.npz", allow_pickle=False) as mesh_data:
            mesh_nodes = mesh_data["nodes_xy_um"]
            mesh_triangles = mesh_data["triangles"]
            mesh_area_m2 = mesh_data["triangle_area_um2"] * 1.0e-12
        if len(mesh_nodes) != fx_node.shape[1] or len(mesh_triangles) != tx_element.shape[1]:
            issues.append("load arrays do not match canonical mesh dimensions")
        if not np.allclose(nodes_xy_m, mesh_nodes * 1.0e-6, atol=1e-15, rtol=0):
            issues.append("nodal-force coordinates differ from the canonical mesh")
        if not np.allclose(element_area, mesh_area_m2, atol=1e-24, rtol=1e-12):
            issues.append("element areas differ from the canonical mesh")
        meshio_data = meshio.read(loads_dir / "traction_mapped.vtu")
        pyvista_data = pv.read(loads_dir / "traction_mapped.vtu")
        if len(meshio_data.points) != len(mesh_nodes):
            issues.append("mapped VTU point count mismatch")
        if len(meshio_data.get_cells_type("triangle")) != len(mesh_triangles):
            issues.append("mapped VTU element count mismatch")
        if pyvista_data.n_points != len(mesh_nodes) or pyvista_data.n_cells != len(mesh_triangles):
            issues.append("PyVista mapped VTU counts mismatch")
        for name in ("fx_N", "fy_N", "nodal_force_magnitude_N"):
            if name not in pyvista_data.point_data:
                issues.append(f"mapped VTU missing point field {name}")
        for name in ("tx_Pa", "ty_Pa", "traction_magnitude_Pa"):
            if name not in pyvista_data.cell_data:
                issues.append(f"mapped VTU missing cell field {name}")
    except Exception as exc:
        issues.append(f"mapped VTU validation failed: {exc}")

    expected_count = int(summary.get("expected_frame_count", 0))
    if require_complete and (
        len(frame_indices) != expected_count
        or not np.array_equal(frame_indices, np.arange(expected_count))
    ):
        issues.append("canonical frame set is incomplete or non-contiguous")
    if summary.get("mapping", {}).get("quadrature_points_outside_traction_support", 1) != 0:
        issues.append("summary reports quadrature outside traction support")
    return issues, summary
