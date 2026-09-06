"""Validation of persistent Stage 3 vector-geometry artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from shapely.geometry import shape


REQUIRED_FILES = (
    "domain.geojson",
    "cells.geojson",
    "adjacency.csv",
    "geometry_metrics.csv",
    "geometry_summary.json",
    "geometry.npz",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _validate_offsets(
    data: np.lib.npyio.NpzFile,
    prefix: str,
    item_count: int,
    issues: list[str],
) -> None:
    item_offsets = data[f"{prefix}_part_offsets"]
    part_offsets = data[f"{prefix}_part_ring_offsets"]
    ring_offsets = data[f"{prefix}_ring_vertex_offsets"]
    holes = data[f"{prefix}_ring_is_hole"]
    vertices = data[f"{prefix}_vertex_xy_image_px"]
    physical = data[f"{prefix}_vertex_xy_physical_um"]
    fem = data[f"{prefix}_vertex_xy_fem_um"]
    if len(item_offsets) != item_count + 1:
        issues.append(f"{prefix}: item/part offset count mismatch")
    for name, offsets in (
        ("item_part", item_offsets),
        ("part_ring", part_offsets),
        ("ring_vertex", ring_offsets),
    ):
        if offsets.ndim != 1 or not len(offsets) or offsets[0] != 0:
            issues.append(f"{prefix}: invalid {name} offsets")
        elif np.any(np.diff(offsets) < 0):
            issues.append(f"{prefix}: non-monotonic {name} offsets")
    if len(item_offsets) and item_offsets[-1] != len(part_offsets) - 1:
        issues.append(f"{prefix}: final part offset mismatch")
    if len(part_offsets) and part_offsets[-1] != len(ring_offsets) - 1:
        issues.append(f"{prefix}: final ring offset mismatch")
    if len(ring_offsets) and ring_offsets[-1] != len(vertices):
        issues.append(f"{prefix}: final vertex offset mismatch")
    if len(holes) != len(ring_offsets) - 1:
        issues.append(f"{prefix}: ring-hole flag count mismatch")
    if vertices.shape != physical.shape or vertices.shape != fem.shape:
        issues.append(f"{prefix}: coordinate representation shapes differ")


def validate_geometry_artifacts(
    position_dir: Path,
    *,
    vector_dir_override: Path | None = None,
    expected_frame: int | None = None,
) -> tuple[list[str], dict[str, Any]]:
    vector_dir = vector_dir_override or position_dir / "geometry/vector"
    issues: list[str] = []
    for filename in REQUIRED_FILES:
        if not (vector_dir / filename).is_file():
            issues.append(f"missing {filename}")
    if issues:
        return issues, {}

    try:
        summary = _read_json(vector_dir / "geometry_summary.json")
    except Exception as exc:
        return [f"cannot read geometry_summary.json: {exc}"], {}
    if expected_frame is not None and summary.get("geometry_frame") != expected_frame:
        issues.append(
            f"geometry frame {summary.get('geometry_frame')} != expected {expected_frame}"
        )
    if summary.get("position") != position_dir.name:
        issues.append("summary position mismatch")

    try:
        domain_geojson = _read_json(vector_dir / "domain.geojson")
        cells_geojson = _read_json(vector_dir / "cells.geojson")
        if domain_geojson.get("metadata", {}).get("units") != "micrometers":
            issues.append("domain GeoJSON units are not micrometers")
        if cells_geojson.get("metadata", {}).get("units") != "micrometers":
            issues.append("cell GeoJSON units are not micrometers")
        if len(domain_geojson.get("features", [])) != 1:
            issues.append("domain GeoJSON must contain one domain feature")
        else:
            domain_geometry = shape(domain_geojson["features"][0]["geometry"])
            if domain_geometry.is_empty or not domain_geometry.is_valid:
                issues.append("domain GeoJSON geometry is empty or invalid")
        expected_cells = summary["cells"]["extracted_polygon_count"]
        if len(cells_geojson.get("features", [])) != expected_cells:
            issues.append("cell GeoJSON feature count mismatch")
        feature_ids = [feature.get("properties", {}).get("cell_id") for feature in cells_geojson["features"]]
        if len(feature_ids) != len(set(feature_ids)):
            issues.append("cell GeoJSON contains duplicate cell IDs")
        invalid_geojson_cells = 0
        empty_geojson_cells = 0
        for feature in cells_geojson.get("features", []):
            geometry = shape(feature["geometry"])
            invalid_geojson_cells += not geometry.is_valid
            empty_geojson_cells += geometry.is_empty
        if invalid_geojson_cells:
            issues.append(f"{invalid_geojson_cells} cell GeoJSON geometries are invalid")
        if empty_geojson_cells:
            issues.append(f"{empty_geojson_cells} cell GeoJSON geometries are empty")
    except Exception as exc:
        issues.append(f"GeoJSON validation failed: {exc}")

    try:
        metrics = pd.read_csv(vector_dir / "geometry_metrics.csv")
        adjacency = pd.read_csv(vector_dir / "adjacency.csv")
        required_metrics = {
            "position",
            "cell_id",
            "area_pixels",
            "area_um2",
            "perimeter_um",
            "centroid_x_um",
            "centroid_y_um",
            "neighbor_count",
            "shared_boundary_total_um",
            "touches_domain_boundary",
            "is_valid",
            "relative_area_change_after_cleanup",
        }
        missing_metrics = sorted(required_metrics - set(metrics.columns))
        if missing_metrics:
            issues.append("metrics missing columns: " + ", ".join(missing_metrics))
        if len(metrics) != summary["cells"]["extracted_polygon_count"]:
            issues.append("geometry metrics row count mismatch")
        if metrics["cell_id"].duplicated().any():
            issues.append("geometry metrics contain duplicate cell IDs")
        required_adjacency = {"cell_id_a", "cell_id_b", "shared_boundary_um"}
        if not required_adjacency.issubset(adjacency.columns):
            issues.append("adjacency CSV is missing required columns")
        elif len(adjacency):
            if (adjacency["cell_id_a"] >= adjacency["cell_id_b"]).any():
                issues.append("adjacency pairs are not uniquely ordered")
            if (adjacency["shared_boundary_um"] <= 0).any():
                issues.append("adjacency contains non-positive shared boundaries")
            if adjacency.duplicated(["cell_id_a", "cell_id_b"]).any():
                issues.append("adjacency contains duplicate pairs")
    except Exception as exc:
        issues.append(f"CSV validation failed: {exc}")

    try:
        with np.load(vector_dir / "geometry.npz", allow_pickle=False) as data:
            if any(data[name].dtype == object for name in data.files):
                issues.append("geometry NPZ contains object arrays")
            cell_ids = data["cell_ids"]
            if len(cell_ids) != summary["cells"]["extracted_polygon_count"]:
                issues.append("geometry NPZ cell count mismatch")
            if int(data["geometry_frame"]) != summary["geometry_frame"]:
                issues.append("geometry NPZ frame mismatch")
            _validate_offsets(data, "cell", len(cell_ids), issues)
            _validate_offsets(data, "domain", 1, issues)
    except Exception as exc:
        issues.append(f"NPZ validation failed: {exc}")

    try:
        frame = int(summary["geometry_frame"])
        labels = tifffile.imread(position_dir / "geometry/labels.tif", key=frame)
        label_ids = np.unique(labels)
        label_ids = label_ids[label_ids != 0]
        raster_area = int(np.count_nonzero(labels))
        if len(label_ids) != summary["cells"]["raster_label_count"]:
            issues.append("summary raster label count disagrees with labels.tif")
        if raster_area != summary["cells"]["raster_labeled_area_pixels"]:
            issues.append("summary raster label area disagrees with labels.tif")
    except Exception as exc:
        issues.append(f"raster cross-check failed: {exc}")

    cells = summary.get("cells", {})
    overlap = summary.get("overlap", {})
    if cells.get("invalid_polygon_count", 0):
        issues.append(f"{cells['invalid_polygon_count']} invalid polygons remain")
    if cells.get("empty_polygon_count", 0):
        issues.append(f"{cells['empty_polygon_count']} empty polygons remain")
    if overlap.get("overlapping_cell_pairs", 0):
        issues.append(f"{overlap['overlapping_cell_pairs']} cell overlaps remain")
    maximum_error = cells.get("maximum_per_cell_relative_area_error")
    threshold = summary.get("parameters", {}).get("large_area_change_threshold", 0.01)
    if maximum_error is not None and maximum_error > threshold:
        issues.append(
            f"maximum per-cell raster/vector area error {maximum_error} exceeds {threshold}"
        )
    if not summary.get("domain", {}).get("is_valid", False):
        issues.append("domain geometry is invalid")
    return issues, summary
