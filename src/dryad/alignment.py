"""Numerically align published DIC, traction, and image coordinate systems.

This module creates diagnostics only. It does not mesh, solve, invert tractions,
or calculate optical flow.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import yaml
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table


LOGGER = logging.getLogger("dryad.alignment")


def _scalar(array: np.ndarray) -> int | float:
    value = np.asarray(array).reshape(-1)[0].item()
    return value


def _median_spacing(grid: np.ndarray, axis: int) -> float | None:
    differences = np.diff(np.asarray(grid, dtype=np.float64), axis=axis)
    finite = differences[np.isfinite(differences)]
    return float(np.median(finite)) if finite.size else None


def _finite_float(value: float | np.floating | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _summary(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
    }


def _frame_stats(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if reference.shape != candidate.shape or reference.ndim != 3:
        raise ValueError(
            f"displacement comparison requires matching 3-D arrays, got "
            f"{reference.shape} and {candidate.shape}"
        )
    rmse = np.full(reference.shape[-1], np.nan, dtype=np.float64)
    correlation = np.full(reference.shape[-1], np.nan, dtype=np.float64)
    for frame in range(reference.shape[-1]):
        first = reference[..., frame].ravel()
        second = candidate[..., frame].ravel()
        valid = np.isfinite(first) & np.isfinite(second)
        if not np.any(valid):
            continue
        first = first[valid]
        second = second[valid]
        rmse[frame] = np.sqrt(np.mean(np.square(first - second)))
        if first.size >= 2 and np.std(first) > 0 and np.std(second) > 0:
            correlation[frame] = np.corrcoef(first, second)[0, 1]
        elif np.array_equal(first, second):
            correlation[frame] = 1.0
    return rmse, correlation


def _detect_subset(
    dic_x: np.ndarray,
    dic_y: np.ndarray,
    traction_x: np.ndarray,
    traction_y: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    target_shape = traction_x.shape
    if traction_y.shape != target_shape or dic_x.shape != dic_y.shape:
        return {"traction_is_dic_subset": False, "reason": "x/y grid shape mismatch"}
    if target_shape[0] > dic_x.shape[0] or target_shape[1] > dic_x.shape[1]:
        return {"traction_is_dic_subset": False, "reason": "traction grid is larger than DIC"}

    candidates = np.argwhere(
        np.isclose(dic_x, traction_x[0, 0], atol=tolerance, rtol=0)
        & np.isclose(dic_y, traction_y[0, 0], atol=tolerance, rtol=0)
    )
    best_error: float | None = None
    best_slice: tuple[int, int] | None = None
    for row_start, col_start in candidates:
        row_stop = int(row_start + target_shape[0])
        col_stop = int(col_start + target_shape[1])
        if row_stop > dic_x.shape[0] or col_stop > dic_x.shape[1]:
            continue
        x_slice = dic_x[row_start:row_stop, col_start:col_stop]
        y_slice = dic_y[row_start:row_stop, col_start:col_stop]
        error = float(
            max(
                np.max(np.abs(x_slice.astype(np.float64) - traction_x)),
                np.max(np.abs(y_slice.astype(np.float64) - traction_y)),
            )
        )
        if best_error is None or error < best_error:
            best_error = error
            best_slice = (int(row_start), int(col_start))

    if best_slice is not None and best_error is not None and best_error <= tolerance:
        row_start, col_start = best_slice
        return {
            "traction_is_dic_subset": True,
            "dic_row_start": row_start,
            "dic_row_stop": row_start + target_shape[0],
            "dic_col_start": col_start,
            "dic_col_stop": col_start + target_shape[1],
            "removed_rows_before": row_start,
            "removed_rows_after": dic_x.shape[0] - (row_start + target_shape[0]),
            "removed_cols_before": col_start,
            "removed_cols_after": dic_x.shape[1] - (col_start + target_shape[1]),
            "max_coordinate_error": best_error,
            "comparison_tolerance": tolerance,
        }

    dic_points = np.column_stack((dic_x.ravel(), dic_y.ravel())).astype(np.float64)
    traction_points = np.column_stack((traction_x.ravel(), traction_y.ravel())).astype(
        np.float64
    )
    distances, _ = cKDTree(dic_points).query(traction_points)
    dic_dx = _median_spacing(dic_x, 1)
    dic_dy = _median_spacing(dic_y, 0)
    traction_dx = _median_spacing(traction_x, 1)
    traction_dy = _median_spacing(traction_y, 0)
    same_spacing = (
        dic_dx is not None
        and dic_dy is not None
        and traction_dx is not None
        and traction_dy is not None
        and abs(dic_dx - traction_dx) <= tolerance
        and abs(dic_dy - traction_dy) <= tolerance
    )
    return {
        "traction_is_dic_subset": False,
        "classification": (
            "translated_or_cropped_and_shifted" if same_spacing else "differently_spaced_or_interpolated"
        ),
        "nearest_dic_coordinate_error": _summary(distances),
        "comparison_tolerance": tolerance,
    }


def _rectilinear(grid_x: np.ndarray, grid_y: np.ndarray, tolerance: float) -> bool:
    expected_x = np.broadcast_to(grid_x[0:1, :], grid_x.shape)
    expected_y = np.broadcast_to(grid_y[:, 0:1], grid_y.shape)
    return bool(
        np.allclose(grid_x, expected_x, atol=tolerance, rtol=0)
        and np.allclose(grid_y, expected_y, atol=tolerance, rtol=0)
    )


def _interpolate_dic_to_traction(
    dic_x: np.ndarray,
    dic_y: np.ndarray,
    field: np.ndarray,
    traction_x: np.ndarray,
    traction_y: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    if not _rectilinear(dic_x, dic_y, tolerance):
        raise ValueError("DIC grid is not rectilinear; interpolation was not attempted")
    points = np.column_stack((traction_y.ravel(), traction_x.ravel()))
    output = np.empty((*traction_x.shape, field.shape[-1]), dtype=np.float64)
    for frame in range(field.shape[-1]):
        interpolator = RegularGridInterpolator(
            (dic_y[:, 0], dic_x[0, :]),
            field[..., frame],
            bounds_error=False,
            fill_value=np.nan,
        )
        output[..., frame] = interpolator(points).reshape(traction_x.shape)
    return output


def _centroid_label_evidence(
    position_dir: Path, pixel_size_um: float | None
) -> dict[str, Any]:
    centroid_path = position_dir / "geometry/centroids.npz"
    labels_path = position_dir / "geometry/labels.tif"
    if pixel_size_um is None or not centroid_path.is_file() or not labels_path.is_file():
        return {"status": "unavailable"}
    try:
        with np.load(centroid_path, allow_pickle=False) as data:
            offsets = data["centroids_offsets"]
            shapes = data["centroids_shapes"]
            ndims = data["centroids_ndims"]
            shape = tuple(int(value) for value in shapes[0, : int(ndims[0])])
            centroids = data["centroids_values"][offsets[0] : offsets[1]].reshape(shape)
            source_pix_size = _scalar(data["pix_size"])
        labels = tifffile.imread(labels_path, key=0)
        properties = regionprops_table(labels, properties=("centroid",))
        image_centroids_zero_based = np.column_stack(
            (properties["centroid-1"], properties["centroid-0"])
        )
        source_in_pixel_units = centroids / float(source_pix_size)
        expected_one_based = image_centroids_zero_based + 1.0
        distances, _ = cKDTree(expected_one_based).query(source_in_pixel_units)
        maximum_error = float(np.max(distances)) if distances.size else None
        verified = (
            distances.size == centroids.shape[0]
            and maximum_error is not None
            and maximum_error <= 1.0e-8
            and math.isclose(float(source_pix_size), pixel_size_um, rel_tol=0, abs_tol=1e-12)
        )
        return {
            "status": "verified" if verified else "not_verified",
            "frame_tested": 0,
            "centroids_tested": int(centroids.shape[0]),
            "label_regions_available": int(image_centroids_zero_based.shape[0]),
            "centroid_pix_size": float(source_pix_size),
            "normalized_pixel_size_um": float(pixel_size_um),
            "max_centroid_coordinate_error_pixels": maximum_error,
            "observed_relation": (
                "centroid_um / pixel_size_um = (image_col + 1, image_row + 1)"
            ),
            "conclusion": (
                "dataset image coordinates use MATLAB-style one-based pixel coordinates"
                if verified
                else "image-coordinate offset remains unresolved"
            ),
        }
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}


def _load_metadata(position_dir: Path) -> dict[str, Any]:
    with (position_dir / "metadata.yaml").open(encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("metadata.yaml is not a mapping")
    return metadata


def _coordinate_transform(
    position: str,
    image_shape: tuple[int, ...],
    pixel_size_m: float | None,
    subset: dict[str, Any],
    centroid_evidence: dict[str, Any],
) -> dict[str, Any]:
    _, height, width = image_shape
    one_based_verified = centroid_evidence.get("status") == "verified"
    return {
        "schema_version": 1,
        "position": position,
        "image_coordinates": {
            "array_index_origin": "top_left",
            "column_range_zero_based": [0, width - 1],
            "row_range_zero_based": [0, height - 1],
            "x_direction": "right",
            "y_direction": "down",
        },
        "source_mechanical_coordinates": {
            "origin_convention": (
                "MATLAB_one_based_pixel_coordinates" if one_based_verified else "unverified"
            ),
            "x_direction": "right",
            "y_direction": "down",
            "to_image_array_index": {
                "scale_x": 1.0,
                "scale_y": 1.0,
                "offset_x": -1.0 if one_based_verified else None,
                "offset_y": -1.0 if one_based_verified else None,
                "flip_y": False,
                "status": "verified" if one_based_verified else "unverified",
            },
            "evidence": centroid_evidence,
        },
        "dic_to_traction": {
            "coordinate_transform": {
                "scale_x": 1.0,
                "scale_y": 1.0,
                "offset_x": 0.0,
                "offset_y": 0.0,
                "flip_y": False,
            },
            "index_mapping": subset,
        },
        "source_to_physical": {
            "pixel_size_m": pixel_size_m,
            "relative_distance_scale_x": pixel_size_m,
            "relative_distance_scale_y": pixel_size_m,
            "rotation_or_shear": "none_observed",
        },
        "image_array_index_to_fem_target": {
            "target_origin": "bottom_left_pixel_center",
            "x_m_formula": "column_index * pixel_size_m",
            "y_m_formula": "(image_height - 1 - row_index) * pixel_size_m",
            "scale_x": pixel_size_m,
            "scale_y": -pixel_size_m if pixel_size_m is not None else None,
            "offset_x": 0.0,
            "offset_y": ((height - 1) * pixel_size_m if pixel_size_m is not None else None),
            "flip_y": True,
            "applied_in_stage_2": False,
        },
        "display_transform": {
            "description": "Image, DIC, and traction are displayed in top-left image coordinates",
            "dic_or_traction_to_display": {
                "x": "source_x - 1",
                "y": "source_y - 1",
                "requires_reversed_screen_y_axis": True,
                "status": "verified" if one_based_verified else "provisional",
            },
        },
    }


def analyze_position(position_dir: Path) -> dict[str, Any]:
    position = position_dir.name
    metadata = _load_metadata(position_dir)
    expected_frames = metadata.get("imaging", {}).get("n_frames")
    pixel_size_m = metadata.get("imaging", {}).get("pixel_size_m")
    pixel_size_um = metadata.get("imaging", {}).get("pixel_size_um")

    with np.load(position_dir / "displacement/published_dic.npz", allow_pickle=False) as data:
        dic = {name: data[name] for name in ("x", "y", "u", "v", "w0", "d0", "inc")}
    with np.load(position_dir / "traction/published.npz", allow_pickle=False) as data:
        traction = {
            name: data[name]
            for name in ("x", "y", "u", "v", "tx", "ty", "w0", "d0", "inc")
        }

    dic_x = dic["x"]
    dic_y = dic["y"]
    traction_x = traction["x"]
    traction_y = traction["y"]
    spacings = [
        value
        for value in (
            _median_spacing(dic_x, 1),
            _median_spacing(dic_y, 0),
            _median_spacing(traction_x, 1),
            _median_spacing(traction_y, 0),
        )
        if value is not None
    ]
    tolerance = max(1.0e-9, max(map(abs, spacings), default=1.0) * 1.0e-9)
    subset = _detect_subset(dic_x, dic_y, traction_x, traction_y, tolerance)

    if subset.get("traction_is_dic_subset"):
        rows = np.arange(subset["dic_row_start"], subset["dic_row_stop"], dtype=np.int64)
        cols = np.arange(subset["dic_col_start"], subset["dic_col_stop"], dtype=np.int64)
        reference_u = dic["u"][np.ix_(rows, cols, np.arange(dic["u"].shape[-1]))]
        reference_v = dic["v"][np.ix_(rows, cols, np.arange(dic["v"].shape[-1]))]
        comparison_method = "direct_DIC_index_subset"
    else:
        rows = np.empty(0, dtype=np.int64)
        cols = np.empty(0, dtype=np.int64)
        reference_u = _interpolate_dic_to_traction(
            dic_x, dic_y, dic["u"], traction_x, traction_y, tolerance
        )
        reference_v = _interpolate_dic_to_traction(
            dic_x, dic_y, dic["v"], traction_x, traction_y, tolerance
        )
        comparison_method = "linear_interpolation_of_DIC_to_traction_coordinates"

    u_rmse, u_correlation = _frame_stats(reference_u, traction["u"])
    v_rmse, v_correlation = _frame_stats(reference_v, traction["v"])
    global_u_rmse = float(np.sqrt(np.nanmean(np.square(reference_u - traction["u"]))))
    global_v_rmse = float(np.sqrt(np.nanmean(np.square(reference_v - traction["v"]))))

    domain_path = position_dir / "geometry/domain.tif"
    labels_path = position_dir / "geometry/labels.tif"
    with tifffile.TiffFile(domain_path) as tif:
        domain_shape = tuple(int(value) for value in tif.series[0].shape)
        domain_dtype = str(tif.series[0].dtype)
    with tifffile.TiffFile(labels_path) as tif:
        labels_shape = tuple(int(value) for value in tif.series[0].shape)
        labels_dtype = str(tif.series[0].dtype)
    if len(domain_shape) != 3:
        raise ValueError(f"expected a 3-D domain TIFF, got {domain_shape}")

    centroid_evidence = _centroid_label_evidence(position_dir, pixel_size_um)
    coordinate_transform = _coordinate_transform(
        position, domain_shape, pixel_size_m, subset, centroid_evidence
    )

    diagnostics_dir = position_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    npz_path = diagnostics_dir / "grid_alignment.npz"
    temporary_npz = diagnostics_dir / ".grid_alignment.npz.tmp"
    with temporary_npz.open("wb") as handle:
        np.savez(
            handle,
            dic_x=dic_x,
            dic_y=dic_y,
            traction_x=traction_x,
            traction_y=traction_y,
            dic_row_indices=rows,
            dic_col_indices=cols,
            u_rmse_by_frame=u_rmse,
            v_rmse_by_frame=v_rmse,
            u_correlation_by_frame=u_correlation,
            v_correlation_by_frame=v_correlation,
        )
    temporary_npz.replace(npz_path)

    d0 = _scalar(dic["d0"])
    w0 = _scalar(dic["w0"])
    crop_spacing_relation = None
    if subset.get("traction_is_dic_subset") and d0:
        crop_spacing_relation = {
            "w0_divided_by_d0": float(w0 / d0),
            "removed_grid_samples_each_edge": {
                "top": subset["removed_rows_before"],
                "bottom": subset["removed_rows_after"],
                "left": subset["removed_cols_before"],
                "right": subset["removed_cols_after"],
            },
            "numerically_equal": bool(
                all(
                    count == w0 / d0
                    for count in (
                        subset["removed_rows_before"],
                        subset["removed_rows_after"],
                        subset["removed_cols_before"],
                        subset["removed_cols_after"],
                    )
                )
            ),
            "historical_causation": "unverified",
        }

    diagnostics = {
        "schema_version": 1,
        "position": position,
        "status": "success",
        "expected_frames": expected_frames,
        "image": {
            "domain_shape": list(domain_shape),
            "domain_dtype": domain_dtype,
            "labels_shape": list(labels_shape),
            "labels_dtype": labels_dtype,
        },
        "dic": {
            "spatial_shape": list(dic_x.shape),
            "field_shape": list(dic["u"].shape),
            "x_min": float(np.min(dic_x)),
            "x_max": float(np.max(dic_x)),
            "y_min": float(np.min(dic_y)),
            "y_max": float(np.max(dic_y)),
            "median_dx": _median_spacing(dic_x, 1),
            "median_dy": _median_spacing(dic_y, 0),
            "rectilinear": _rectilinear(dic_x, dic_y, tolerance),
            "w0": _scalar(dic["w0"]),
            "d0": _scalar(dic["d0"]),
            "inc": _scalar(dic["inc"]),
        },
        "traction": {
            "spatial_shape": list(traction_x.shape),
            "field_shape": list(traction["tx"].shape),
            "x_min": float(np.min(traction_x)),
            "x_max": float(np.max(traction_x)),
            "y_min": float(np.min(traction_y)),
            "y_max": float(np.max(traction_y)),
            "median_dx": _median_spacing(traction_x, 1),
            "median_dy": _median_spacing(traction_y, 0),
            "rectilinear": _rectilinear(traction_x, traction_y, tolerance),
            "w0": _scalar(traction["w0"]),
            "d0": _scalar(traction["d0"]),
            "inc": _scalar(traction["inc"]),
        },
        "grid_mapping": subset,
        "crop_parameter_relation": crop_spacing_relation,
        "displacement_agreement": {
            "comparison_method": comparison_method,
            "u_rmse_global": global_u_rmse,
            "v_rmse_global": global_v_rmse,
            "u_rmse_by_frame_summary": _summary(u_rmse),
            "v_rmse_by_frame_summary": _summary(v_rmse),
            "u_correlation_by_frame_summary": _summary(u_correlation),
            "v_correlation_by_frame_summary": _summary(v_correlation),
            "per_frame_values": "grid_alignment.npz",
            "fields_are_exactly_equal": bool(
                np.array_equal(reference_u, traction["u"], equal_nan=True)
                and np.array_equal(reference_v, traction["v"], equal_nan=True)
            ),
        },
        "coordinate_transform": "coordinate_transform.json",
        "observed_numerical_mapping": (
            "traction coordinates are an exact contiguous subset of DIC coordinates"
            if subset.get("traction_is_dic_subset")
            else subset.get("classification", "not an exact subset")
        ),
        "historical_processing_interpretation": {
            "status": "unverified",
            "statement": (
                "The numerical crop equals w0/d0 grid samples on every edge, but no local "
                "historical processing source code was found to prove why it was applied."
            ),
        },
        "warnings": [],
    }
    _write_json(diagnostics_dir / "coordinate_transform.json", coordinate_transform)
    _write_json(diagnostics_dir / "alignment.json", diagnostics)
    return diagnostics


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def discover_positions(data_root: Path, requested: list[str] | None) -> list[Path]:
    available = {
        path.name: path
        for path in data_root.iterdir()
        if path.is_dir() and path.name.startswith("pos") and path.name[3:].isdigit()
    }
    if requested:
        missing = [position for position in requested if position not in available]
        if missing:
            raise ValueError(f"positions not found: {', '.join(missing)}")
        names = requested
    else:
        names = list(available)
    return [available[name] for name in sorted(set(names), key=lambda value: int(value[3:]))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--positions", nargs="+")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data_root = args.data.resolve()
    try:
        positions = discover_positions(data_root, args.positions)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    successful = 0
    failed = 0
    for position_dir in positions:
        try:
            result = analyze_position(position_dir)
            mapping = result["grid_mapping"]
            LOGGER.info(
                "%s: subset=%s rows=%s:%s cols=%s:%s max_error=%s",
                position_dir.name,
                mapping.get("traction_is_dic_subset"),
                mapping.get("dic_row_start"),
                mapping.get("dic_row_stop"),
                mapping.get("dic_col_start"),
                mapping.get("dic_col_stop"),
                mapping.get("max_coordinate_error"),
            )
            successful += 1
        except Exception:
            LOGGER.exception("%s: alignment failed", position_dir.name)
            failed += 1
    print(f"Alignment positions: {len(positions)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
