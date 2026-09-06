"""Validation checks for source and standardized Dryad position data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import tifffile


REQUIRED_SOURCE_FILES = (
    "Centroids_to_Voronoi.mat",
    "DIC_c1_64w0_16d0.mat",
    "KNN_results_JN_refine.mat",
    "tractions_DIC_c1_64w0.mat",
    "ExperimentalSettings.txt",
    "Label_Image.tif",
    "domain.tif",
    "c1.tif",
    "c1_ref.tif",
    "c2.tif",
    "nuc.tif",
)


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str


def validate_required_sources(source_dir: Path) -> list[ValidationIssue]:
    return [
        ValidationIssue("error", "missing_source", f"missing required source file: {name}")
        for name in REQUIRED_SOURCE_FILES
        if not (source_dir / name).is_file()
    ]


def validate_dic_arrays(
    arrays: Mapping[str, np.ndarray], expected_frames: int | None
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        u, v, x, y = (np.asarray(arrays[name]) for name in ("u", "v", "x", "y"))
    except KeyError as exc:
        return [ValidationIssue("error", "dic_missing_array", f"DIC missing {exc.args[0]}")]

    if u.shape != v.shape:
        issues.append(ValidationIssue("error", "dic_uv_shape", f"u {u.shape} != v {v.shape}"))
    if u.ndim != 3:
        issues.append(ValidationIssue("error", "dic_dimensions", f"u must be 3-D, got {u.shape}"))
    else:
        if expected_frames is not None and u.shape[-1] != expected_frames:
            issues.append(
                ValidationIssue(
                    "error",
                    "dic_frame_count",
                    f"DIC has {u.shape[-1]} frames; expected {expected_frames}",
                )
            )
        spatial_shape = u.shape[:2]
        if x.shape != spatial_shape or y.shape != spatial_shape:
            issues.append(
                ValidationIssue(
                    "error",
                    "dic_grid_shape",
                    f"x/y shapes {x.shape}/{y.shape} do not match {spatial_shape}",
                )
            )
    c_peak = arrays.get("c_peak")
    if c_peak is not None and u.ndim == 3 and np.asarray(c_peak).shape != u.shape[:2]:
        issues.append(
            ValidationIssue(
                "error",
                "dic_c_peak_shape",
                f"c_peak {np.asarray(c_peak).shape} does not match {u.shape[:2]}",
            )
        )
    return issues


def validate_traction_arrays(
    arrays: Mapping[str, np.ndarray], expected_frames: int | None
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        u, v, tx, ty, x, y = (
            np.asarray(arrays[name]) for name in ("u", "v", "tx", "ty", "x", "y")
        )
    except KeyError as exc:
        return [
            ValidationIssue("error", "traction_missing_array", f"traction missing {exc.args[0]}")
        ]

    field_shapes = {u.shape, v.shape, tx.shape, ty.shape}
    if len(field_shapes) != 1:
        issues.append(
            ValidationIssue(
                "error",
                "traction_field_shape",
                f"u/v/tx/ty shapes differ: {u.shape}, {v.shape}, {tx.shape}, {ty.shape}",
            )
        )
    if u.ndim != 3:
        issues.append(
            ValidationIssue("error", "traction_dimensions", f"fields must be 3-D, got {u.shape}")
        )
    else:
        if expected_frames is not None and u.shape[-1] != expected_frames:
            issues.append(
                ValidationIssue(
                    "error",
                    "traction_frame_count",
                    f"traction has {u.shape[-1]} frames; expected {expected_frames}",
                )
            )
        spatial_shape = u.shape[:2]
        if x.shape != spatial_shape or y.shape != spatial_shape:
            issues.append(
                ValidationIssue(
                    "error",
                    "traction_grid_shape",
                    f"x/y shapes {x.shape}/{y.shape} do not match {spatial_shape}",
                )
            )
    return issues


def validate_tiff(path: Path, label: str) -> tuple[list[ValidationIssue], tuple[int, ...] | None]:
    try:
        with tifffile.TiffFile(path) as tif:
            if not tif.series:
                raise ValueError("contains no TIFF series")
            shape = tuple(int(value) for value in tif.series[0].shape)
        return [], shape
    except Exception as exc:  # corrupted TIFFs can raise several library exceptions
        return [ValidationIssue("error", "tiff_open", f"cannot open {label}: {exc}")], None


def validate_output_position(
    output_dir: Path, expected_frames: int | None
) -> tuple[list[ValidationIssue], dict[str, object]]:
    """Validate an existing standardized position and return manifest facts."""
    issues: list[ValidationIssue] = []
    facts: dict[str, object] = {
        "dic_shape_y": None,
        "dic_shape_x": None,
        "traction_shape_y": None,
        "traction_shape_x": None,
        "has_domain": False,
        "has_labels": False,
        "has_dic": False,
        "has_traction": False,
        "has_centroids": False,
        "has_trajectories": False,
    }

    dic_path = output_dir / "displacement" / "published_dic.npz"
    if dic_path.is_file():
        try:
            with np.load(dic_path, allow_pickle=False) as data:
                arrays = {name: data[name] for name in ("u", "v", "x", "y", "c_peak")}
                issues.extend(validate_dic_arrays(arrays, expected_frames))
                if arrays["u"].ndim >= 2:
                    facts["dic_shape_y"], facts["dic_shape_x"] = arrays["u"].shape[:2]
            facts["has_dic"] = True
        except Exception as exc:
            issues.append(ValidationIssue("error", "dic_npz", f"cannot validate {dic_path}: {exc}"))

    traction_path = output_dir / "traction" / "published.npz"
    if traction_path.is_file():
        try:
            with np.load(traction_path, allow_pickle=False) as data:
                arrays = {name: data[name] for name in ("u", "v", "tx", "ty", "x", "y")}
                issues.extend(validate_traction_arrays(arrays, expected_frames))
                if arrays["u"].ndim >= 2:
                    facts["traction_shape_y"], facts["traction_shape_x"] = arrays["u"].shape[:2]
            facts["has_traction"] = True
        except Exception as exc:
            issues.append(
                ValidationIssue("error", "traction_npz", f"cannot validate {traction_path}: {exc}")
            )

    for key, relative in (
        ("has_domain", Path("geometry/domain.tif")),
        ("has_labels", Path("geometry/labels.tif")),
    ):
        path = output_dir / relative
        if path.is_file():
            tiff_issues, _ = validate_tiff(path, str(relative))
            issues.extend(tiff_issues)
            facts[key] = not tiff_issues

    centroid_path = output_dir / "geometry/centroids.npz"
    if centroid_path.is_file():
        try:
            encoded_cells = (
                "centroids",
                "areas",
                "perimeters",
                "shape_index",
                "count_vertices",
                "Tsx",
                "Tsy",
                "NNZ_shape_index",
            )
            direct_fields = (
                "Avg_shape_index",
                "Avg_amp_T_in",
                "Avg_amp_T_in_by_area",
                "startFrame",
                "endFrame",
                "pix_size",
            )
            with np.load(centroid_path, allow_pickle=False) as data:
                required_keys = set(direct_fields)
                for name in encoded_cells:
                    required_keys.update(
                        {
                            f"{name}_values",
                            f"{name}_offsets",
                            f"{name}_shapes",
                            f"{name}_ndims",
                            f"{name}_cell_shape",
                        }
                    )
                missing = sorted(required_keys - set(data.files))
                if missing:
                    raise ValueError(f"missing arrays: {', '.join(missing)}")
            facts["has_centroids"] = True
        except Exception as exc:
            issues.append(
                ValidationIssue("error", "centroid_npz", f"cannot validate {centroid_path}: {exc}")
            )

    trajectory_path = output_dir / "geometry/trajectories.npz"
    if trajectory_path.is_file():
        try:
            required_keys = {
                "traj_x_retained",
                "traj_y_retained",
                "traj_x_retained_atleastT",
                "traj_y_retained_atleastT",
                "min_surviv_hr",
                "start_frame",
                "end_frame",
            }
            with np.load(trajectory_path, allow_pickle=False) as data:
                missing = sorted(required_keys - set(data.files))
                if missing:
                    raise ValueError(f"missing arrays: {', '.join(missing)}")
                if data["traj_x_retained"].shape != data["traj_y_retained"].shape:
                    raise ValueError("retained trajectory x/y shapes differ")
                if (
                    data["traj_x_retained_atleastT"].shape
                    != data["traj_y_retained_atleastT"].shape
                ):
                    raise ValueError("atleastT trajectory x/y shapes differ")
            facts["has_trajectories"] = True
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    "error", "trajectory_npz", f"cannot validate {trajectory_path}: {exc}"
                )
            )

    required_outputs = (
        "metadata.yaml",
        "source_index.yaml",
        "geometry/domain.tif",
        "geometry/labels.tif",
        "geometry/centroids.npz",
        "geometry/trajectories.npz",
        "displacement/published_dic.npz",
        "traction/published.npz",
        "images/source_paths.yaml",
    )
    for relative in required_outputs:
        if not (output_dir / relative).is_file():
            issues.append(
                ValidationIssue("error", "missing_output", f"missing standardized output: {relative}")
            )
    return issues, facts
