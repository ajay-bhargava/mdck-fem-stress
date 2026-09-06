#!/usr/bin/env python3
"""Build persistent Stage 3 vector geometry from standardized raster inputs."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import yaml
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.prepared import prep
from skimage.measure import regionprops

from dryad.geometry import (
    Polygonal,
    adjacency_and_overlap,
    clean_polygon,
    geometry_rings,
    image_to_fem_um,
    image_to_physical_um,
    polygonal_parts,
    raster_mask_to_polygon,
)
from dryad.geometry_validate import validate_geometry_artifacts


LOGGER = logging.getLogger("dryad.build_geometry")
LABEL_SEMANTICS = (
    "nuclear_objects_from_StarDist_Label_Image; not verified epithelial cell boundaries"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--positions", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="reference raster frame to vectorize (default: 0)",
    )
    parser.add_argument(
        "--simplify-tolerance-px",
        type=float,
        default=0.0,
        help="topology-preserving simplification tolerance; default preserves pixel edges",
    )
    parser.add_argument(
        "--minimum-shared-boundary-px",
        type=float,
        default=0.5,
        help="minimum shared polygon-boundary length for adjacency",
    )
    parser.add_argument(
        "--large-area-change-threshold",
        type=float,
        default=0.01,
        help="relative cleanup area change that triggers a warning",
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


def _feature_collection(features: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "metadata": metadata,
        "features": features,
    }


def _fem_transform_parameters(transform: dict[str, Any]) -> dict[str, float]:
    parameters = transform["image_array_index_to_fem_target"]
    required = ("scale_x", "scale_y", "offset_x", "offset_y")
    if any(parameters.get(name) is None for name in required):
        raise ValueError("persisted Stage 2 FEM affine transform is incomplete")
    if parameters.get("flip_y") is not True:
        raise ValueError("persisted Stage 2 transform does not contain the required y flip")
    return {name: float(parameters[name]) for name in required}


def _pack_polygon_set(
    geometries: list[Polygonal],
    *,
    prefix: str,
    pixel_size_um: float,
    fem: dict[str, float],
) -> dict[str, np.ndarray]:
    part_offsets = [0]
    part_ring_offsets = [0]
    ring_vertex_offsets = [0]
    ring_is_hole: list[bool] = []
    vertices: list[np.ndarray] = []
    for geometry in geometries:
        parts = polygonal_parts(geometry)
        part_offsets.append(part_offsets[-1] + len(parts))
        for part in parts:
            rings = [(np.asarray(part.exterior.coords), False)] + [
                (np.asarray(interior.coords), True) for interior in part.interiors
            ]
            part_ring_offsets.append(part_ring_offsets[-1] + len(rings))
            for coordinates, is_hole in rings:
                coordinates = coordinates.astype(np.float64, copy=False)
                vertices.append(coordinates)
                ring_vertex_offsets.append(ring_vertex_offsets[-1] + len(coordinates))
                ring_is_hole.append(is_hole)
    image_xy = np.concatenate(vertices) if vertices else np.empty((0, 2), dtype=np.float64)
    physical_xy = image_xy * pixel_size_um
    fem_xy = np.empty_like(image_xy)
    fem_xy[:, 0] = (image_xy[:, 0] * fem["scale_x"] + fem["offset_x"]) * 1.0e6
    fem_xy[:, 1] = (image_xy[:, 1] * fem["scale_y"] + fem["offset_y"]) * 1.0e6
    return {
        f"{prefix}_part_offsets": np.asarray(part_offsets, dtype=np.int64),
        f"{prefix}_part_ring_offsets": np.asarray(part_ring_offsets, dtype=np.int64),
        f"{prefix}_ring_vertex_offsets": np.asarray(ring_vertex_offsets, dtype=np.int64),
        f"{prefix}_ring_is_hole": np.asarray(ring_is_hole, dtype=bool),
        f"{prefix}_vertex_xy_image_px": image_xy,
        f"{prefix}_vertex_xy_physical_um": physical_xy,
        f"{prefix}_vertex_xy_fem_um": fem_xy,
    }


def _load_reference_centroids(
    position_dir: Path, frame: int
) -> tuple[np.ndarray | None, float | None]:
    path = position_dir / "geometry/centroids.npz"
    if not path.is_file():
        return None, None
    with np.load(path, allow_pickle=False) as data:
        offsets = data["centroids_offsets"]
        if frame + 1 >= len(offsets):
            return None, None
        shape = tuple(
            int(value)
            for value in data["centroids_shapes"][
                frame, : int(data["centroids_ndims"][frame])
            ]
        )
        values = data["centroids_values"][offsets[frame] : offsets[frame + 1]]
        centroids = values.reshape(shape)
        pix_size = float(np.asarray(data["pix_size"]).reshape(-1)[0])
    return centroids, pix_size


def _centroid_agreement(
    geometries: list[Polygonal],
    cell_ids: np.ndarray,
    source_centroids: np.ndarray | None,
    source_pix_size_um: float | None,
    *,
    tolerance_px: float = 1.0e-6,
) -> tuple[dict[str, Any], set[int]]:
    if source_centroids is None or source_pix_size_um is None or not geometries:
        return {"status": "unavailable", "semantic_identity": "not_proven"}, set()
    vector_centroids = np.asarray(
        [(geometry.centroid.x, geometry.centroid.y) for geometry in geometries],
        dtype=np.float64,
    )
    # Stage 2 established source/pix_size = zero-based image centroid + 1.
    reference_image = np.asarray(source_centroids, dtype=np.float64) / source_pix_size_um - 1.0
    distances, indices = cKDTree(vector_centroids).query(reference_image)
    matched = distances <= tolerance_px
    matched_indices = indices[matched]
    unique_matches = len(np.unique(matched_indices))
    one_to_one = unique_matches == int(np.sum(matched))
    matched_ids = {int(cell_ids[index]) for index in matched_indices} if one_to_one else set()
    finite = distances[np.isfinite(distances)]
    return {
        "status": "spatially_matched" if np.any(matched) else "no_match",
        "matching_method": "nearest vector centroid after persisted MATLAB one-based offset",
        "semantic_identity": (
            "label correspondence established by unique centroid location; no trajectory-ID correspondence inferred"
            if one_to_one
            else "ambiguous"
        ),
        "reference_centroid_count": int(len(reference_image)),
        "matched_centroid_count": int(np.sum(matched)),
        "unique_one_to_one": one_to_one,
        "median_mismatch_pixels": float(np.median(finite)) if finite.size else None,
        "maximum_mismatch_pixels": float(np.max(finite)) if finite.size else None,
        "match_tolerance_pixels": tolerance_px,
    }, matched_ids


def build_position(
    position_dir: Path,
    *,
    frame: int,
    simplify_tolerance_px: float,
    minimum_shared_boundary_px: float,
    large_area_change_threshold: float,
    overwrite: bool,
) -> dict[str, Any]:
    position = position_dir.name
    vector_dir = position_dir / "geometry/vector"
    if vector_dir.exists() and not overwrite:
        issues, summary = validate_geometry_artifacts(position_dir, expected_frame=frame)
        if issues:
            raise ValueError("; ".join(issues))
        LOGGER.info("%s: vector geometry exists; validated without overwrite", position)
        return summary

    metadata = read_yaml(position_dir / "metadata.yaml")
    transform = read_json(position_dir / "diagnostics/coordinate_transform.json")
    if transform.get("position") != position:
        raise ValueError("coordinate transform position mismatch")
    fem = _fem_transform_parameters(transform)
    pixel_size_um = float(metadata["imaging"]["pixel_size_um"])
    persisted_pixel_size_m = float(transform["source_to_physical"]["pixel_size_m"])
    if not np.isclose(pixel_size_um, persisted_pixel_size_m * 1.0e6, atol=1e-12, rtol=0):
        raise ValueError("metadata pixel size disagrees with persisted Stage 2 transform")

    domain_path = position_dir / "geometry/domain.tif"
    labels_path = position_dir / "geometry/labels.tif"
    with tifffile.TiffFile(domain_path) as tif:
        image_shape = tuple(int(value) for value in tif.series[0].shape)
    if len(image_shape) != 3 or not 0 <= frame < image_shape[0]:
        raise ValueError(f"frame {frame} invalid for domain shape {image_shape}")
    domain_raster = tifffile.imread(domain_path, key=frame) > 0
    labels = tifffile.imread(labels_path, key=frame)
    if labels.shape != domain_raster.shape:
        raise ValueError(f"domain {domain_raster.shape} and labels {labels.shape} differ")
    image_height, image_width = labels.shape
    expected_offset_y = (image_height - 1) * persisted_pixel_size_m
    if not np.isclose(fem["offset_y"], expected_offset_y, atol=1e-15, rtol=0):
        raise ValueError("persisted FEM y-flip offset disagrees with raster image height")

    domain_original = raster_mask_to_polygon(domain_raster)
    domain_cleanup = clean_polygon(
        domain_original, simplify_tolerance_px=simplify_tolerance_px
    )
    domain_geometry = domain_cleanup.geometry
    if domain_geometry.is_empty or not domain_geometry.is_valid:
        raise ValueError("domain could not be converted to valid polygonal geometry")

    properties = regionprops(labels, cache=False)
    cell_ids = np.asarray([int(prop.label) for prop in properties], dtype=np.int64)
    geometries: list[Polygonal] = []
    records: list[dict[str, Any]] = []
    for index, prop in enumerate(properties):
        min_row, min_col, _, _ = prop.bbox
        original = raster_mask_to_polygon(
            prop.image,
            row_offset=int(min_row),
            col_offset=int(min_col),
        )
        cleaned = clean_polygon(original, simplify_tolerance_px=simplify_tolerance_px)
        geometry = cleaned.geometry
        geometries.append(geometry)
        centroid = geometry.centroid if not geometry.is_empty else None
        records.append(
            {
                "position": position,
                "geometry_frame": frame,
                "cell_id": int(prop.label),
                "area_pixels": float(prop.area),
                "vector_area_pixels": float(geometry.area),
                "original_polygon_area_pixels": cleaned.original_area,
                "cleaned_polygon_area_pixels": cleaned.cleaned_area,
                "area_um2": float(geometry.area * pixel_size_um**2),
                "perimeter_pixels": float(geometry.length),
                "perimeter_um": float(geometry.length * pixel_size_um),
                "centroid_x_pixel": float(centroid.x) if centroid else None,
                "centroid_y_pixel": float(centroid.y) if centroid else None,
                "centroid_x_image_um": float(centroid.x * pixel_size_um) if centroid else None,
                "centroid_y_image_um": float(centroid.y * pixel_size_um) if centroid else None,
                "centroid_x_um": (
                    float((centroid.x * fem["scale_x"] + fem["offset_x"]) * 1.0e6)
                    if centroid
                    else None
                ),
                "centroid_y_um": (
                    float((centroid.y * fem["scale_y"] + fem["offset_y"]) * 1.0e6)
                    if centroid
                    else None
                ),
                "was_valid_before_cleanup": cleaned.was_valid,
                "is_valid": cleaned.is_valid,
                "is_empty": cleaned.is_empty,
                "is_multipolygon": cleaned.multipolygon,
                "polygon_component_count": len(polygonal_parts(geometry)),
                "cleanup_operations": ";".join(cleaned.cleanup_operations),
                "relative_area_change_after_cleanup": cleaned.relative_area_change,
                "relative_raster_vector_area_error": (
                    abs(float(geometry.area) - float(prop.area)) / float(prop.area)
                    if prop.area
                    else None
                ),
                "large_cleanup_area_change": (
                    cleaned.relative_area_change > large_area_change_threshold
                ),
            }
        )
        if (index + 1) % 1000 == 0:
            LOGGER.info("%s: vectorized %d/%d labels", position, index + 1, len(properties))

    edges, overlap = adjacency_and_overlap(
        geometries,
        cell_ids,
        minimum_shared_boundary_px=minimum_shared_boundary_px,
    )
    id_to_index = {int(cell_id): index for index, cell_id in enumerate(cell_ids)}
    neighbor_count = np.zeros(len(cell_ids), dtype=np.int64)
    shared_total_px = np.zeros(len(cell_ids), dtype=np.float64)
    for edge in edges:
        first = id_to_index[int(edge["cell_id_a"])]
        second = id_to_index[int(edge["cell_id_b"])]
        shared = float(edge["shared_boundary_pixels"])
        neighbor_count[first] += 1
        neighbor_count[second] += 1
        shared_total_px[first] += shared
        shared_total_px[second] += shared
        edge["shared_boundary_um"] = shared * pixel_size_um

    prepared_domain = prep(domain_geometry)
    boundary_status_counts: dict[str, int] = {}
    for index, (geometry, record) in enumerate(zip(geometries, records, strict=True)):
        if geometry.is_empty or not geometry.is_valid:
            status = "invalid"
        elif prepared_domain.covers(geometry):
            shared_with_domain = float(
                geometry.boundary.intersection(domain_geometry.boundary).length
            )
            status = "boundary-touching" if shared_with_domain > 1.0e-9 else "inside"
        elif prepared_domain.intersects(geometry):
            status = "partially outside"
        else:
            status = "outside"
        boundary_status_counts[status] = boundary_status_counts.get(status, 0) + 1
        record["domain_status"] = status
        record["touches_domain_boundary"] = status in {
            "boundary-touching",
            "partially outside",
        }
        record["neighbor_count"] = int(neighbor_count[index])
        record["shared_boundary_total_um"] = float(shared_total_px[index] * pixel_size_um)

    source_centroids, source_pix_size_um = _load_reference_centroids(position_dir, frame)
    centroid_summary, centroid_matched_ids = _centroid_agreement(
        geometries, cell_ids, source_centroids, source_pix_size_um
    )
    for record in records:
        record["matched_existing_centroid"] = record["cell_id"] in centroid_matched_ids

    domain_components = sorted(
        polygonal_parts(domain_geometry), key=lambda component: component.area, reverse=True
    )
    domain_holes = sum(len(component.interiors) for component in domain_components)
    component_areas = [float(component.area) for component in domain_components]
    component_records = [
        {
            "component_index": index,
            "area_pixels": float(component.area),
            "area_um2": float(component.area * pixel_size_um**2),
            "classification": "primary_largest" if index == 0 else "disconnected_secondary",
            "removed": False,
        }
        for index, component in enumerate(domain_components)
    ]
    hole_records = [
        {
            "component_index": component_index,
            "hole_index": hole_index,
            "area_pixels": float(Polygon(interior).area),
            "area_um2": float(Polygon(interior).area * pixel_size_um**2),
            "removed": False,
        }
        for component_index, component in enumerate(domain_components)
        for hole_index, interior in enumerate(component.interiors)
    ]
    total_raster_area = float(np.count_nonzero(labels))
    total_vector_area = float(sum(geometry.area for geometry in geometries))
    global_area_error = (
        abs(total_vector_area - total_raster_area) / total_raster_area
        if total_raster_area
        else None
    )
    per_cell_errors = [
        record["relative_raster_vector_area_error"]
        for record in records
        if record["relative_raster_vector_area_error"] is not None
    ]
    areas_um2 = np.asarray([record["area_um2"] for record in records], dtype=np.float64)
    invalid_count = sum(not record["is_valid"] for record in records)
    empty_count = sum(record["is_empty"] for record in records)
    multipolygon_count = sum(record["is_multipolygon"] for record in records)
    large_change_count = sum(record["large_cleanup_area_change"] for record in records)

    adjacency_frame = pd.DataFrame(
        edges,
        columns=(
            "cell_id_a",
            "cell_id_b",
            "shared_boundary_pixels",
            "shared_boundary_um",
        ),
    )
    metrics_frame = pd.DataFrame(records)

    warnings: list[str] = [
        "Label_Image.tif is documented as StarDist nuclear segmentation; polygon labels are not verified epithelial cell boundaries."
    ]
    if boundary_status_counts.get("partially outside", 0) or boundary_status_counts.get("outside", 0):
        warnings.append(
            "Some label polygons extend beyond or lie outside the domain; they were flagged and not clipped."
        )
    if large_change_count:
        warnings.append(f"{large_change_count} labels exceeded the cleanup area-change threshold.")

    transform_metadata = {
        "source_artifact": "../../diagnostics/coordinate_transform.json",
        "image_space": {
            "origin": "upper_left",
            "x": "column",
            "y": "row",
            "units": "pixels",
            "pixel_model": "unit squares centered on zero-based pixel indices",
        },
        "physical_microscope_space": {
            "units": "micrometers",
            "scale_x_um_per_pixel": pixel_size_um,
            "scale_y_um_per_pixel": pixel_size_um,
            "y_direction": "down",
        },
        "fem_space": {
            "units": "micrometers",
            "y_direction": "up",
            "scale_x_m": fem["scale_x"],
            "scale_y_m": fem["scale_y"],
            "offset_x_m": fem["offset_x"],
            "offset_y_m": fem["offset_y"],
            "applied": True,
        },
    }
    domain_fem = image_to_fem_um(
        domain_geometry,
        scale_x_m=fem["scale_x"],
        scale_y_m=fem["scale_y"],
        offset_x_m=fem["offset_x"],
        offset_y_m=fem["offset_y"],
    )
    cells_fem = [
        image_to_fem_um(
            geometry,
            scale_x_m=fem["scale_x"],
            scale_y_m=fem["scale_y"],
            offset_x_m=fem["offset_x"],
            offset_y_m=fem["offset_y"],
        )
        for geometry in geometries
    ]

    domain_geojson = _feature_collection(
        [
            {
                "type": "Feature",
                "id": position,
                "properties": {
                    "position": position,
                    "geometry_frame": frame,
                    "area_um2": float(domain_fem.area),
                    "valid": bool(domain_fem.is_valid),
                    "component_count": len(domain_components),
                    "hole_count": domain_holes,
                },
                "geometry": mapping(domain_fem),
            }
        ],
        {
            "coordinate_space": "FEM",
            "units": "micrometers",
            "transform": transform_metadata,
        },
    )
    cell_features = []
    for geometry, record in zip(cells_fem, records, strict=True):
        properties_for_feature = {
            key: record[key]
            for key in (
                "cell_id",
                "area_um2",
                "perimeter_um",
                "centroid_x_um",
                "centroid_y_um",
                "neighbor_count",
                "domain_status",
                "touches_domain_boundary",
                "is_valid",
                "is_multipolygon",
                "relative_area_change_after_cleanup",
            )
        }
        properties_for_feature["valid"] = properties_for_feature.pop("is_valid")
        properties_for_feature["label_semantics"] = LABEL_SEMANTICS
        cell_features.append(
            {
                "type": "Feature",
                "id": int(record["cell_id"]),
                "properties": properties_for_feature,
                "geometry": mapping(geometry),
            }
        )
    cells_geojson = _feature_collection(
        cell_features,
        {
            "coordinate_space": "FEM",
            "units": "micrometers",
            "geometry_frame": frame,
            "label_semantics": LABEL_SEMANTICS,
            "transform": transform_metadata,
        },
    )

    domain_raster_area = float(np.count_nonzero(domain_raster))
    summary = {
        "schema_version": 1,
        "position": position,
        "status": "success",
        "geometry_frame": frame,
        "label_semantics": LABEL_SEMANTICS,
        "parameters": {
            "simplify_tolerance_pixels": simplify_tolerance_px,
            "minimum_shared_boundary_pixels": minimum_shared_boundary_px,
            "large_area_change_threshold": large_area_change_threshold,
            "components_removed": 0,
            "cells_clipped_to_domain": 0,
        },
        "coordinate_transform": transform_metadata,
        "domain": {
            "is_valid": bool(domain_geometry.is_valid),
            "raster_area_pixels": domain_raster_area,
            "vector_area_pixels": float(domain_geometry.area),
            "area_um2": float(domain_geometry.area * pixel_size_um**2),
            "relative_area_error": (
                abs(float(domain_geometry.area) - domain_raster_area) / domain_raster_area
                if domain_raster_area
                else None
            ),
            "component_count": len(domain_components),
            "hole_count": domain_holes,
            "component_areas_pixels": component_areas,
            "components": component_records,
            "holes": hole_records,
            "cleanup_relative_area_change": domain_cleanup.relative_area_change,
        },
        "cells": {
            "raster_label_count": int(len(cell_ids)),
            "extracted_polygon_count": len(geometries),
            "invalid_polygon_count": invalid_count,
            "empty_polygon_count": empty_count,
            "multipolygon_count": multipolygon_count,
            "large_cleanup_area_change_count": large_change_count,
            "raster_labeled_area_pixels": total_raster_area,
            "vector_area_pixels": total_vector_area,
            "global_relative_area_error": global_area_error,
            "maximum_per_cell_relative_area_error": max(per_cell_errors, default=None),
            "mean_area_um2": float(np.mean(areas_um2)) if len(areas_um2) else None,
            "median_area_um2": float(np.median(areas_um2)) if len(areas_um2) else None,
            "boundary_status_counts": boundary_status_counts,
        },
        "adjacency": {
            "edge_count": len(edges),
            "minimum_shared_boundary_pixels": minimum_shared_boundary_px,
            "mean_neighbor_count": float(np.mean(neighbor_count)) if len(neighbor_count) else None,
            "median_neighbor_count": float(np.median(neighbor_count)) if len(neighbor_count) else None,
            "maximum_neighbor_count": int(np.max(neighbor_count)) if len(neighbor_count) else None,
        },
        "overlap": overlap,
        "centroid_agreement": centroid_summary,
        "warnings": warnings,
        "stage4_readiness": {
            "domain_geometry_safe_for_constrained_meshing": bool(
                domain_geometry.is_valid
                and not domain_geometry.is_empty
                and summary_safe_error(
                    abs(float(domain_geometry.area) - domain_raster_area) / domain_raster_area
                    if domain_raster_area
                    else None
                )
            ),
            "label_polygons_safe_as_epithelial_cell_boundaries": False,
            "reason": (
                "Raster/vector geometry is valid and area-conserving, but source documentation "
                "identifies labels as nuclei rather than epithelial cell boundaries."
            ),
        },
        "npz_encoding": {
            "description": "ragged components/rings/vertices without object arrays",
            "topology": (
                "item_part_offsets -> part_ring_offsets -> ring_vertex_offsets; "
                "ring_is_hole marks interior rings"
            ),
        },
    }

    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{position}.vector.", dir=position_dir / "geometry")
    )
    try:
        write_json(temporary_dir / "domain.geojson", domain_geojson)
        write_json(temporary_dir / "cells.geojson", cells_geojson)
        adjacency_frame.to_csv(temporary_dir / "adjacency.csv", index=False)
        metrics_frame.to_csv(temporary_dir / "geometry_metrics.csv", index=False)
        write_json(temporary_dir / "geometry_summary.json", summary)

        payload: dict[str, np.ndarray] = {
            "schema_version": np.asarray("1.0"),
            "geometry_frame": np.asarray(frame, dtype=np.int64),
            "cell_ids": cell_ids,
            "pixel_size_um": np.asarray(pixel_size_um, dtype=np.float64),
            "image_shape": np.asarray(labels.shape, dtype=np.int64),
            "cell_centroid_xy_image_px": np.asarray(
                [[record["centroid_x_pixel"], record["centroid_y_pixel"]] for record in records],
                dtype=np.float64,
            ),
            "cell_area_pixels": np.asarray(
                [record["vector_area_pixels"] for record in records], dtype=np.float64
            ),
        }
        payload.update(
            _pack_polygon_set(
                geometries,
                prefix="cell",
                pixel_size_um=pixel_size_um,
                fem=fem,
            )
        )
        payload.update(
            _pack_polygon_set(
                [domain_geometry],
                prefix="domain",
                pixel_size_um=pixel_size_um,
                fem=fem,
            )
        )
        with (temporary_dir / "geometry.npz").open("wb") as handle:
            np.savez(handle, **payload)

        issues, _ = validate_geometry_artifacts(
            position_dir,
            vector_dir_override=temporary_dir,
            expected_frame=frame,
        )
        if issues:
            raise ValueError("generated geometry failed validation: " + "; ".join(issues))
        if vector_dir.exists():
            shutil.rmtree(vector_dir)
        temporary_dir.replace(vector_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return summary


def summary_safe_error(value: float | None, tolerance: float = 1.0e-9) -> bool:
    return value is not None and value <= tolerance


def main() -> int:
    args = parse_args()
    if args.validate_only and args.overwrite:
        raise SystemExit("--validate-only and --overwrite cannot be combined")
    if args.frame < 0:
        raise SystemExit("--frame must be non-negative")
    if args.simplify_tolerance_px < 0 or args.minimum_shared_boundary_px < 0:
        raise SystemExit("geometry tolerances must be non-negative")
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
            if args.validate_only:
                issues, summary = validate_geometry_artifacts(
                    position_dir, expected_frame=args.frame
                )
                if issues:
                    raise ValueError("; ".join(issues))
                LOGGER.info(
                    "%s: valid, labels=%s, adjacency=%s",
                    position_dir.name,
                    summary["cells"]["raster_label_count"],
                    summary["adjacency"]["edge_count"],
                )
            else:
                summary = build_position(
                    position_dir,
                    frame=args.frame,
                    simplify_tolerance_px=args.simplify_tolerance_px,
                    minimum_shared_boundary_px=args.minimum_shared_boundary_px,
                    large_area_change_threshold=args.large_area_change_threshold,
                    overwrite=args.overwrite,
                )
                LOGGER.info(
                    "%s: labels=%s, adjacency=%s, area_error=%s",
                    position_dir.name,
                    summary["cells"]["raster_label_count"],
                    summary["adjacency"]["edge_count"],
                    summary["cells"]["global_relative_area_error"],
                )
            successful += 1
        except Exception:
            LOGGER.exception("%s: geometry processing failed", position_dir.name)
            failed += 1
    print(f"Geometry positions: {len(positions)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
