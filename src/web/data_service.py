"""Read-only service layer for standardized data and Stage 2 diagnostics."""

from __future__ import annotations

import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import yaml
from PIL import Image
from shapely.geometry import LineString

from src.web.plotting import (
    domain_outline_payload,
    grid_comparison_payload,
    image_frame_payload,
    json_array,
    mechanical_frame_payload,
)


class PositionNotFound(KeyError):
    pass


class FrameOutOfRange(ValueError):
    pass


@lru_cache(maxsize=6)
def _load_npz(path_string: str) -> dict[str, np.ndarray]:
    with np.load(path_string, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


@lru_cache(maxsize=64)
def _load_yaml(path_string: str) -> dict[str, Any]:
    with Path(path_string).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path_string}")
    return value


@lru_cache(maxsize=64)
def _load_json(path_string: str) -> dict[str, Any]:
    with Path(path_string).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path_string}")
    return value


@lru_cache(maxsize=16)
def _load_csv(path_string: str) -> pd.DataFrame:
    return pd.read_csv(path_string)


@lru_cache(maxsize=32)
def _render_grayscale_jpeg(path_string: str, frame: int, max_width: int = 1280) -> bytes:
    image = tifffile.imread(path_string, key=frame).astype(np.float32)
    finite = image[np.isfinite(image)]
    if not len(finite):
        normalized = np.zeros(image.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, (0.5, 99.8))
        if high <= low:
            high = low + 1.0
        normalized = np.clip((image - low) / (high - low), 0.0, 1.0)
        normalized = np.round(255.0 * np.power(normalized, 0.82)).astype(np.uint8)
    rendered = Image.fromarray(normalized, mode="L")
    if rendered.width > max_width:
        height = max(1, round(rendered.height * max_width / rendered.width))
        rendered = rendered.resize((max_width, height), Image.Resampling.LANCZOS)
    output = BytesIO()
    rendered.save(output, format="JPEG", quality=84, optimize=True)
    return output.getvalue()


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _rings_for_item(
    arrays: dict[str, np.ndarray],
    prefix: str,
    item_index: int,
    coordinate_space: str,
    simplify_px: float = 0.0,
) -> list[dict[str, Any]]:
    part_offsets = arrays[f"{prefix}_part_offsets"]
    part_ring_offsets = arrays[f"{prefix}_part_ring_offsets"]
    vertex_offsets = arrays[f"{prefix}_ring_vertex_offsets"]
    holes = arrays[f"{prefix}_ring_is_hole"]
    vertices = arrays[f"{prefix}_vertex_xy_{coordinate_space}"]
    rings: list[dict[str, Any]] = []
    for part in range(int(part_offsets[item_index]), int(part_offsets[item_index + 1])):
        for ring in range(int(part_ring_offsets[part]), int(part_ring_offsets[part + 1])):
            coordinates = vertices[vertex_offsets[ring] : vertex_offsets[ring + 1]]
            if simplify_px > 0 and coordinate_space == "image_px" and len(coordinates) > 4:
                simplified = np.asarray(LineString(coordinates).simplify(simplify_px).coords)
                if len(simplified) >= 3:
                    if not np.array_equal(simplified[0], simplified[-1]):
                        simplified = np.vstack((simplified, simplified[0]))
                    coordinates = simplified
            rings.append(
                {
                    "part_index": part - int(part_offsets[item_index]),
                    "is_hole": bool(holes[ring]),
                    "xy": coordinates.tolist(),
                }
            )
    return rings


class DataService:
    def __init__(self, data_root: Path):
        self.data_root = data_root.resolve()

    def positions(self) -> list[str]:
        if not self.data_root.is_dir():
            return []
        positions = [
            path.name
            for path in self.data_root.iterdir()
            if path.is_dir()
            and path.name.startswith("pos")
            and path.name[3:].isdigit()
            and (path / "metadata.yaml").is_file()
        ]
        return sorted(positions, key=lambda name: int(name[3:]))

    def position_dir(self, position: str) -> Path:
        if position not in self.positions():
            raise PositionNotFound(position)
        return self.data_root / position

    def metadata(self, position: str) -> dict[str, Any]:
        directory = self.position_dir(position)
        metadata = dict(_load_yaml(str(directory / "metadata.yaml")))
        metadata["available_layers"] = self.available_layers(position)
        return metadata

    def alignment(self, position: str) -> dict[str, Any]:
        directory = self.position_dir(position)
        path = directory / "diagnostics/alignment.json"
        if not path.is_file():
            raise FileNotFoundError(f"alignment artifact is missing for {position}")
        return _load_json(str(path))

    def coordinate_transform(self, position: str) -> dict[str, Any]:
        directory = self.position_dir(position)
        path = directory / "diagnostics/coordinate_transform.json"
        if not path.is_file():
            raise FileNotFoundError(f"coordinate transform is missing for {position}")
        return _load_json(str(path))

    def available_layers(self, position: str) -> list[str]:
        directory = self.position_dir(position)
        layers = []
        if (directory / "fem/stress/time_series/summary.json").is_file() or (
            directory / "fem/stress/pilot_frame000_sign_plus/stress.npz"
        ).is_file():
            layers.append("predicted_stress")
        source_paths = directory / "images/source_paths.yaml"
        if source_paths.is_file() and _load_yaml(str(source_paths)).get("nuclei"):
            layers.append("nuclei")
        if (directory / "geometry/domain.tif").is_file():
            layers.append("domain")
        if (directory / "geometry/labels.tif").is_file():
            layers.append("labels")
        if (directory / "displacement/published_dic.npz").is_file():
            layers.extend(("dic_magnitude", "dic_vectors", "dic_quality"))
        if (directory / "traction/published.npz").is_file():
            layers.extend(("traction_magnitude", "traction_vectors"))
        if (directory / "diagnostics/alignment.json").is_file():
            layers.append("grid_comparison")
        if "domain" in layers and "traction_magnitude" in layers:
            layers.append("traction_domain")
        vector_dir = directory / "geometry/vector"
        if (vector_dir / "geometry.npz").is_file():
            layers.extend(("cell_polygons", "cell_adjacency"))
            if "traction_magnitude" in layers:
                layers.append("cell_geometry_traction")
        mesh_dir = directory / "fem/mesh"
        if (mesh_dir / "mesh.npz").is_file():
            layers.extend(
                (
                    "fem_mesh",
                    "fem_mesh_nuclear",
                    "mesh_quality_area",
                    "mesh_quality_minimum_angle",
                    "mesh_quality_aspect_ratio",
                    "mesh_quality_metric",
                    "mesh_traction_grid",
                )
            )
        loads_dir = directory / "fem/loads"
        if (loads_dir / "load_summary.json").is_file():
            layers.extend(
                (
                    "fem_story",
                    "force_balance_story",
                    "published_traction",
                    "mapped_fem_traction",
                    "published_mapped_traction",
                    "traction_mapping_difference",
                    "fem_nodal_forces",
                    "force_balance",
                )
            )
        return layers

    def dic_frame(self, position: str, frame: int) -> dict[str, Any]:
        directory = self.position_dir(position)
        arrays = _load_npz(str(directory / "displacement/published_dic.npz"))
        self._validate_frame(position, frame, arrays["u"].shape[-1])
        metadata = self.metadata(position)
        return mechanical_frame_payload(
            position=position,
            frame=frame,
            x=arrays["x"],
            y=arrays["y"],
            first=arrays["u"][..., frame],
            second=arrays["v"][..., frame],
            first_name="u",
            second_name="v",
            pixel_size_m=metadata["imaging"].get("pixel_size_m"),
            extra_scalar=("c_peak", arrays["c_peak"]),
        )

    def traction_frame(self, position: str, frame: int) -> dict[str, Any]:
        directory = self.position_dir(position)
        arrays = _load_npz(str(directory / "traction/published.npz"))
        self._validate_frame(position, frame, arrays["tx"].shape[-1])
        metadata = self.metadata(position)
        return mechanical_frame_payload(
            position=position,
            frame=frame,
            x=arrays["x"],
            y=arrays["y"],
            first=arrays["tx"][..., frame],
            second=arrays["ty"][..., frame],
            first_name="tx",
            second_name="ty",
            pixel_size_m=metadata["imaging"].get("pixel_size_m"),
        )

    def source_image_info(self, position: str, layer: str) -> dict[str, Any]:
        directory = self.position_dir(position)
        source_paths = _load_yaml(str(directory / "images/source_paths.yaml"))
        if layer not in {"nuclei", "cell_body"}:
            raise ValueError(f"unsupported source image layer: {layer}")
        raw_path = source_paths.get(layer)
        if not raw_path:
            raise FileNotFoundError(f"source image {layer} is missing for {position}")
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.data_root.parent / path
        path = path.resolve()
        with tifffile.TiffFile(path) as tif:
            shape = tif.series[0].shape
        if len(shape) != 3:
            raise ValueError(f"expected a time-series image for {position}/{layer}")
        return {
            "position": position,
            "layer": layer,
            "frame_count": int(shape[0]),
            "height_px": int(shape[1]),
            "width_px": int(shape[2]),
        }

    def source_image_jpeg(self, position: str, frame: int, layer: str) -> bytes:
        info = self.source_image_info(position, layer)
        self._validate_frame(position, frame, info["frame_count"])
        directory = self.position_dir(position)
        source_paths = _load_yaml(str(directory / "images/source_paths.yaml"))
        path = Path(source_paths[layer])
        if not path.is_absolute():
            path = self.data_root.parent / path
        return _render_grayscale_jpeg(str(path.resolve()), frame)

    @lru_cache(maxsize=24)
    def nuclear_overlay_png(self, position: str, frame: int) -> bytes:
        """Frame-specific nuclear label boundaries; never cell/FEM boundaries."""
        from skimage.segmentation import find_boundaries

        directory = self.position_dir(position)
        self._validate_frame(position, frame, self.metadata(position)["imaging"]["n_frames"])
        labels = tifffile.imread(directory / "geometry/labels.tif", key=frame)
        edges = find_boundaries(labels, mode="thick")
        rgba = np.zeros((*edges.shape, 4), dtype=np.uint8)
        rgba[edges] = [97, 230, 206, 235]
        output = BytesIO()
        Image.fromarray(rgba).save(output, format="PNG")
        return output.getvalue()

    def image_frame(self, position: str, frame: int, layer: str) -> dict[str, Any]:
        directory = self.position_dir(position)
        paths = {
            "domain": directory / "geometry/domain.tif",
            "labels": directory / "geometry/labels.tif",
        }
        if layer not in paths:
            raise ValueError(f"unsupported image layer: {layer}")
        path = paths[layer]
        with tifffile.TiffFile(path) as tif:
            shape = tif.series[0].shape
            n_frames = int(shape[0]) if len(shape) == 3 else 1
        self._validate_frame(position, frame, n_frames)
        image = tifffile.imread(path, key=frame if n_frames > 1 else 0)
        return image_frame_payload(image, position=position, frame=frame, layer=layer)

    def domain_outline(self, position: str, frame: int) -> dict[str, Any]:
        directory = self.position_dir(position)
        path = directory / "geometry/domain.tif"
        with tifffile.TiffFile(path) as tif:
            shape = tif.series[0].shape
            n_frames = int(shape[0]) if len(shape) == 3 else 1
        self._validate_frame(position, frame, n_frames)
        domain = tifffile.imread(path, key=frame if n_frames > 1 else 0)
        return domain_outline_payload(domain, position=position, frame=frame)

    def grid_comparison(self, position: str) -> dict[str, Any]:
        directory = self.position_dir(position)
        arrays = _load_npz(str(directory / "diagnostics/grid_alignment.npz"))
        return grid_comparison_payload(
            position=position,
            dic_x=arrays["dic_x"],
            dic_y=arrays["dic_y"],
            traction_x=arrays["traction_x"],
            traction_y=arrays["traction_y"],
            alignment=self.alignment(position),
        )

    def geometry_summary(self, position: str) -> dict[str, Any]:
        directory = self.position_dir(position)
        path = directory / "geometry/vector/geometry_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"geometry summary is missing for {position}")
        return _load_json(str(path))

    def geometry_domain(self, position: str, simplify_px: float = 0.75) -> dict[str, Any]:
        directory = self.position_dir(position)
        arrays = _load_npz(str(directory / "geometry/vector/geometry.npz"))
        return {
            "position": position,
            "geometry_frame": int(arrays["geometry_frame"]),
            "coordinate_space": "zero_based_image_pixel_centers",
            "rings": _rings_for_item(arrays, "domain", 0, "image_px", simplify_px),
        }

    def geometry_cells(self, position: str, simplify_px: float = 0.75) -> dict[str, Any]:
        directory = self.position_dir(position)
        arrays = _load_npz(str(directory / "geometry/vector/geometry.npz"))
        metrics = _load_csv(str(directory / "geometry/vector/geometry_metrics.csv"))
        boundary_x: list[float | None] = []
        boundary_y: list[float | None] = []
        for index in range(len(arrays["cell_ids"])):
            for ring in _rings_for_item(arrays, "cell", index, "image_px", simplify_px):
                coordinates = np.asarray(ring["xy"])
                boundary_x.extend(coordinates[:, 0].tolist())
                boundary_y.extend(coordinates[:, 1].tolist())
                boundary_x.append(None)
                boundary_y.append(None)
        fields = (
            "cell_id",
            "area_um2",
            "perimeter_um",
            "neighbor_count",
            "domain_status",
            "centroid_x_pixel",
            "centroid_y_pixel",
            "is_multipolygon",
        )
        cells = [
            {field: _json_scalar(value) for field, value in row.items()}
            for row in metrics[list(fields)].to_dict(orient="records")
        ]
        return {
            "position": position,
            "geometry_frame": int(arrays["geometry_frame"]),
            "coordinate_space": "zero_based_image_pixel_centers",
            "simplify_tolerance_pixels": simplify_px,
            "label_semantics": self.geometry_summary(position)["label_semantics"],
            "boundary_x": boundary_x,
            "boundary_y": boundary_y,
            "cells": cells,
        }

    def geometry_cell(self, position: str, cell_id: int) -> dict[str, Any]:
        directory = self.position_dir(position)
        arrays = _load_npz(str(directory / "geometry/vector/geometry.npz"))
        matches = np.flatnonzero(arrays["cell_ids"] == cell_id)
        if not len(matches):
            raise ValueError(f"unknown cell ID {cell_id} for {position}")
        index = int(matches[0])
        metrics = _load_csv(str(directory / "geometry/vector/geometry_metrics.csv"))
        record = metrics.loc[metrics["cell_id"] == cell_id].iloc[0].to_dict()
        record = {key: _json_scalar(value) for key, value in record.items()}
        return {
            "position": position,
            "cell_id": cell_id,
            "properties": record,
            "rings_image_px": _rings_for_item(arrays, "cell", index, "image_px"),
            "rings_physical_um": _rings_for_item(arrays, "cell", index, "physical_um"),
            "rings_fem_um": _rings_for_item(arrays, "cell", index, "fem_um"),
        }

    def geometry_adjacency(self, position: str) -> dict[str, Any]:
        directory = self.position_dir(position)
        adjacency = _load_csv(str(directory / "geometry/vector/adjacency.csv"))
        metrics = _load_csv(str(directory / "geometry/vector/geometry_metrics.csv"))
        centroids = metrics.set_index("cell_id")[["centroid_x_pixel", "centroid_y_pixel"]]
        line_x: list[float | None] = []
        line_y: list[float | None] = []
        edges = []
        for row in adjacency.itertuples(index=False):
            first = centroids.loc[row.cell_id_a]
            second = centroids.loc[row.cell_id_b]
            line_x.extend((float(first.iloc[0]), float(second.iloc[0]), None))
            line_y.extend((float(first.iloc[1]), float(second.iloc[1]), None))
            edges.append(
                {
                    "cell_id_a": int(row.cell_id_a),
                    "cell_id_b": int(row.cell_id_b),
                    "shared_boundary_um": float(row.shared_boundary_um),
                }
            )
        return {
            "position": position,
            "coordinate_space": "zero_based_image_pixel_centers",
            "label_semantics": self.geometry_summary(position)["label_semantics"],
            "line_x": line_x,
            "line_y": line_y,
            "edges": edges,
        }

    def mesh_summary(self, position: str) -> dict[str, Any]:
        directory = self.position_dir(position)
        path = directory / "fem/mesh/mesh_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"mesh summary is missing for {position}")
        return _load_json(str(path))

    def _mesh_selection(
        self, position: str, max_elements: int
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        directory = self.position_dir(position)
        arrays = _load_npz(str(directory / "fem/mesh/mesh.npz"))
        triangles = arrays["triangles"]
        stride = max(1, int(np.ceil(len(triangles) / max_elements)))
        element_ids = np.arange(0, len(triangles), stride, dtype=np.int64)
        selected = triangles[element_ids]
        used_nodes, inverse = np.unique(selected, return_inverse=True)
        local_triangles = inverse.reshape(selected.shape)
        return arrays, element_ids, used_nodes, local_triangles

    def _fem_nodes_to_image(self, position: str, nodes_um: np.ndarray) -> np.ndarray:
        transform = self.coordinate_transform(position)["image_array_index_to_fem_target"]
        scale_x = float(transform["scale_x"]) * 1.0e6
        scale_y = float(transform["scale_y"]) * 1.0e6
        offset_x = float(transform["offset_x"]) * 1.0e6
        offset_y = float(transform["offset_y"]) * 1.0e6
        image = np.empty_like(nodes_um, dtype=np.float64)
        image[:, 0] = (nodes_um[:, 0] - offset_x) / scale_x
        image[:, 1] = (nodes_um[:, 1] - offset_y) / scale_y
        return image

    def mesh(self, position: str, max_elements: int = 30_000) -> dict[str, Any]:
        arrays, element_ids, used_nodes, local_triangles = self._mesh_selection(
            position, max_elements
        )
        nodes_fem = arrays["nodes_xy_um"][used_nodes]
        nodes_image = self._fem_nodes_to_image(position, nodes_fem)
        return {
            "position": position,
            "coordinate_space": "zero_based_image_pixel_centers_for_display",
            "source_coordinate_space": "FEM_micrometers",
            "full_node_count": len(arrays["nodes_xy_um"]),
            "full_element_count": len(arrays["triangles"]),
            "display_element_count": len(element_ids),
            "element_stride": int(element_ids[1] - element_ids[0]) if len(element_ids) > 1 else 1,
            "element_ids": element_ids.tolist(),
            "nodes_xy_image_px": nodes_image.tolist(),
            "nodes_xy_fem_um": nodes_fem.tolist(),
            "triangles": local_triangles.tolist(),
        }

    def mesh_quality(
        self, position: str, metric: str, max_elements: int = 30_000
    ) -> dict[str, Any]:
        fields = {
            "area": "area_um2",
            "minimum_angle": "minimum_angle_degrees",
            "aspect_ratio": "aspect_ratio",
            "quality_metric": "quality_metric",
        }
        if metric not in fields:
            raise ValueError(f"unsupported mesh quality metric: {metric}")
        payload = self.mesh(position, max_elements)
        directory = self.position_dir(position)
        quality = _load_csv(str(directory / "fem/mesh/mesh_quality.csv"))
        element_ids = np.asarray(payload["element_ids"], dtype=np.int64)
        selected = quality.iloc[element_ids]
        payload.update(
            {
                "metric": metric,
                "metric_field": fields[metric],
                "metric_values": selected[fields[metric]].tolist(),
                "area_um2": selected["area_um2"].tolist(),
                "minimum_angle_degrees": selected["minimum_angle_degrees"].tolist(),
                "aspect_ratio": selected["aspect_ratio"].tolist(),
                "quality_metric": selected["quality_metric"].tolist(),
                "component_id": selected["component_id"].astype(int).tolist(),
            }
        )
        return payload

    def mesh_boundary(self, position: str) -> dict[str, Any]:
        directory = self.position_dir(position)
        mesh_arrays = _load_npz(str(directory / "fem/mesh/mesh.npz"))
        boundary = _load_npz(str(directory / "fem/mesh/boundary.npz"))
        nodes_fem = mesh_arrays["nodes_xy_um"]
        nodes_image = self._fem_nodes_to_image(position, nodes_fem)
        edges = boundary["boundary_edges"]
        line_x: list[float | None] = []
        line_y: list[float | None] = []
        for first, second in edges:
            line_x.extend((float(nodes_image[first, 0]), float(nodes_image[second, 0]), None))
            line_y.extend((float(nodes_image[first, 1]), float(nodes_image[second, 1]), None))
        return {
            "position": position,
            "coordinate_space": "zero_based_image_pixel_centers_for_display",
            "boundary_edge_count": len(edges),
            "line_x": line_x,
            "line_y": line_y,
            "boundary_edges": edges.tolist(),
            "boundary_component_id": boundary["boundary_component_id"].tolist(),
            "boundary_ring_id": boundary["boundary_ring_id"].tolist(),
            "boundary_is_hole": boundary["boundary_is_hole"].tolist(),
            "boundary_edge_length_um": boundary["boundary_edge_length_um"].tolist(),
        }

    def stress_pilot_summary(self, position: str) -> dict[str, Any]:
        directory = self.position_dir(position) / "fem/stress/pilot_frame000_sign_plus"
        provenance = _load_json(str(directory / "provenance.json"))
        validation = _load_json(str(directory / "validation.json"))
        arrays = _load_npz(str(directory / "stress.npz"))
        magnitude = np.max(np.abs(np.linalg.eigvalsh(arrays["stress_pa"])), axis=1)
        return {
            "position": position, "frame": int(arrays["frame"]),
            "time_since_first_frame_s": float(arrays["time_since_first_frame_s"]),
            "field": "maximum absolute principal stress", "units": "Pa",
            "color_max_pa": float(np.percentile(magnitude, 99)),
            "maximum_pa": float(magnitude.max()),
            "color_note": "Color saturates above the 99th element percentile; triangle values are not smoothed.",
            "sign_note": "Magnitude is unchanged under global traction-sign reversal; tension/compression remain unresolved.",
            "assumptions": provenance["assumptions"], "validation": validation,
            "status": "Conditional pilot reconstruction; frame 0 only, not a time series.",
        }

    def stress_series_summary(self, position: str) -> dict[str, Any]:
        summary = dict(_load_json(str(self.position_dir(position) / "fem/stress/time_series/summary.json")))
        summary["thickness_options_um"] = [2.5, 5.0, 10.0]
        summary["thickness_comparison_color_max_pa"] = summary["color_max_pa"] * (summary["assumptions"]["thickness_m"] * 1e6 / 2.5)
        summary["thickness_scaling"] = "Exact homogeneous free-sheet linear scaling: sigma(h)=sigma(h0)*h0/h; membrane resultant unchanged. Not a biological response prediction."
        return summary

    def stress_series_png(self, position: str, frame: int, thickness_um: float = 5.0) -> bytes:
        summary = self.stress_series_summary(position)
        if thickness_um not in summary["thickness_options_um"]:
            raise ValueError("Thickness must be 2.5, 5 or 10 micrometres")
        self._validate_frame(position, frame, summary["frame_count"])
        directory = self.position_dir(position) / "fem/stress/time_series"
        rendered = directory / "display_thickness_v1" / f"{thickness_um:g}um" / f"frame{frame:03d}.png"
        if rendered.is_file():
            return rendered.read_bytes()
        with np.load(directory / f"frame{frame:03d}/stress.npz", allow_pickle=False) as data:
            factor = summary["assumptions"]["thickness_m"] * 1e6 / thickness_um
            arrays = {"stress_pa": data["stress_pa"] * factor}
        arrays.update(_load_npz(str(directory / "geometry.npz")))
        return self._render_stress_png(position, arrays, summary["thickness_comparison_color_max_pa"])

    @lru_cache(maxsize=2)
    def stress_pilot_png(self, position: str) -> bytes:
        directory = self.position_dir(position) / "fem/stress/pilot_frame000_sign_plus"
        arrays = _load_npz(str(directory / "stress.npz"))
        summary = self.stress_pilot_summary(position)
        return self._render_stress_png(position, arrays, summary["color_max_pa"])

    def _render_stress_png(self, position: str, arrays: dict, color_max_pa: float) -> bytes:
        from PIL import ImageDraw
        info = self.source_image_info(position, "nuclei")
        scale = 1280 / info["width_px"]
        xy = self._fem_nodes_to_image(position, arrays["nodes_xy_m"] * 1e6) * scale
        magnitude = np.max(np.abs(np.linalg.eigvalsh(arrays["stress_pa"])), axis=1)
        normalized = np.clip(magnitude / max(color_max_pa, 1e-30), 0, 1)
        # Same sequential palette and clipping limits reported by the summary.
        stops = np.array([[22,19,45], [80,35,105], [175,54,94], [240,122,60], [252,237,164]])
        colors = np.column_stack([np.interp(normalized, np.linspace(0,1,5), stops[:,c]) for c in range(3)]).astype(int)
        image = Image.new("RGBA", (1280, round(info["height_px"]*scale)))
        draw = ImageDraw.Draw(image)
        for tri, color in zip(arrays["triangles"], colors):
            draw.polygon([tuple(point) for point in xy[tri]], fill=(*color.tolist(),255))
        output = BytesIO(); image.save(output, format="PNG")
        return output.getvalue()

    def load_summary(self, position: str) -> dict[str, Any]:
        directory = self.position_dir(position)
        path = directory / "fem/loads/load_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"load summary is missing for {position}")
        return _load_json(str(path))

    def load_frame(self, position: str, frame: int) -> dict[str, Any]:
        directory = self.position_dir(position)
        metrics = _load_csv(str(directory / "fem/loads/frame_metrics.csv"))
        selected = metrics.loc[metrics["frame"] == frame]
        if selected.empty:
            raise FrameOutOfRange(f"load frame {frame} is unavailable for {position}")
        return {
            "position": position,
            "frame": frame,
            "metrics": {
                key: _json_scalar(value)
                for key, value in selected.iloc[0].to_dict().items()
                if key != "position"
            },
            "units_and_semantics": self.load_summary(position)["units_and_semantics"],
            "moment_reference": self.load_summary(position)["moment_reference"],
        }

    def load_elements(
        self, position: str, frame: int, max_elements: int = 20_000
    ) -> dict[str, Any]:
        directory = self.position_dir(position)
        arrays = _load_npz(str(directory / "fem/loads/traction_on_elements.npz"))
        frame_matches = np.flatnonzero(arrays["frame_indices"] == frame)
        if not len(frame_matches):
            raise FrameOutOfRange(f"load frame {frame} is unavailable for {position}")
        frame_index = int(frame_matches[0])
        with np.load(directory / "fem/mesh/mesh.npz", allow_pickle=False) as mesh_data:
            centroids_um = mesh_data["triangle_centroid_xy_um"]
        element_count = len(centroids_um)
        stride = max(1, int(np.ceil(element_count / max_elements)))
        element_ids = np.arange(0, element_count, stride, dtype=np.int64)
        centroids_image = self._fem_nodes_to_image(position, centroids_um[element_ids])
        tx_fem = arrays["tx_element"][frame_index, element_ids]
        ty_fem = arrays["ty_element"][frame_index, element_ids]

        lookup = arrays["source_grid_element_index"]
        valid = lookup >= 0
        mapped_tx_fem = np.full(lookup.shape, np.nan, dtype=np.float64)
        mapped_ty_fem = np.full(lookup.shape, np.nan, dtype=np.float64)
        mapped_tx_fem[valid] = arrays["tx_element"][frame_index, lookup[valid]]
        mapped_ty_fem[valid] = arrays["ty_element"][frame_index, lookup[valid]]
        published = _load_npz(str(directory / "traction/published.npz"))
        published_tx_display = published["tx"][..., frame]
        published_ty_display = published["ty"][..., frame]
        mapped_tx_display = mapped_tx_fem
        mapped_ty_display = -mapped_ty_fem
        return {
            "position": position,
            "frame": frame,
            "full_element_count": element_count,
            "display_element_count": len(element_ids),
            "element_stride": stride,
            "element_ids": element_ids.tolist(),
            "centroid_xy_image_px": centroids_image.tolist(),
            "tx_fem_Pa": tx_fem.tolist(),
            "ty_fem_Pa": ty_fem.tolist(),
            "tx_display_Pa": tx_fem.tolist(),
            "ty_display_Pa": (-ty_fem).tolist(),
            "magnitude_Pa": np.hypot(tx_fem, ty_fem).tolist(),
            "definition": "quadrature area-average traction per FEM element",
            "difference_at_source_grid": {
                "definition": "containing-element area average minus published traction at each source-grid point inside the mesh",
                "x_image_px": (published["x"].astype(np.float64) - 1.0).tolist(),
                "y_image_px": (published["y"].astype(np.float64) - 1.0).tolist(),
                "inside_mesh": valid.tolist(),
                "delta_tx_Pa": json_array(mapped_tx_display - published_tx_display),
                "delta_ty_Pa": json_array(mapped_ty_display - published_ty_display),
                "delta_magnitude_Pa": json_array(
                    np.hypot(
                        mapped_tx_display - published_tx_display,
                        mapped_ty_display - published_ty_display,
                    )
                ),
            },
        }

    def load_nodes(
        self, position: str, frame: int, max_nodes: int = 2_000
    ) -> dict[str, Any]:
        directory = self.position_dir(position)
        arrays = _load_npz(str(directory / "fem/loads/nodal_forces.npz"))
        frame_matches = np.flatnonzero(arrays["frame_indices"] == frame)
        if not len(frame_matches):
            raise FrameOutOfRange(f"load frame {frame} is unavailable for {position}")
        index = int(frame_matches[0])
        magnitude = np.hypot(arrays["fx_node"][index], arrays["fy_node"][index])
        stride = max(1, int(np.ceil(len(magnitude) / max_nodes)))
        uniform = np.arange(0, len(magnitude), stride, dtype=np.int64)
        strongest_count = min(250, len(magnitude))
        strongest = np.argpartition(magnitude, -strongest_count)[-strongest_count:]
        node_ids = np.unique(np.concatenate((uniform, strongest)))
        nodes_um = arrays["nodes_xy_m"][node_ids] * 1.0e6
        nodes_image = self._fem_nodes_to_image(position, nodes_um)
        fx = arrays["fx_node"][index, node_ids]
        fy = arrays["fy_node"][index, node_ids]
        return {
            "position": position,
            "frame": frame,
            "label": "Consistent FEM nodal forces [N]",
            "full_node_count": len(magnitude),
            "display_node_count": len(node_ids),
            "node_ids": node_ids.tolist(),
            "xy_image_px": nodes_image.tolist(),
            "fx_fem_N": fx.tolist(),
            "fy_fem_N": fy.tolist(),
            "fx_display_N": fx.tolist(),
            "fy_display_N": (-fy).tolist(),
            "magnitude_N": np.hypot(fx, fy).tolist(),
        }

    def load_balance_map(
        self, position: str, frame: int, bins: int = 10
    ) -> dict[str, Any]:
        directory = self.position_dir(position)
        arrays = _load_npz(str(directory / "fem/loads/nodal_forces.npz"))
        frame_matches = np.flatnonzero(arrays["frame_indices"] == frame)
        if not len(frame_matches):
            raise FrameOutOfRange(f"load frame {frame} is unavailable for {position}")
        index = int(frame_matches[0])
        nodes_um = arrays["nodes_xy_m"] * 1.0e6
        nodes_image = self._fem_nodes_to_image(position, nodes_um)
        image_info = self.source_image_info(position, "nuclei")
        width = float(image_info["width_px"])
        height = float(image_info["height_px"])
        ix = np.clip((nodes_image[:, 0] / width * bins).astype(int), 0, bins - 1)
        iy = np.clip((nodes_image[:, 1] / height * bins).astype(int), 0, bins - 1)
        flat = iy * bins + ix
        fx = arrays["fx_node"][index]
        fy_display = -arrays["fy_node"][index]
        count = bins * bins
        sum_fx = np.bincount(flat, weights=fx, minlength=count)
        sum_fy = np.bincount(flat, weights=fy_display, minlength=count)
        sum_magnitude = np.bincount(flat, weights=np.hypot(fx, fy_display), minlength=count)
        occupied = sum_magnitude > 0
        ids = np.flatnonzero(occupied)
        centers = np.column_stack(
            (
                ((ids % bins) + 0.5) * width / bins,
                ((ids // bins) + 0.5) * height / bins,
            )
        )
        residual = np.hypot(sum_fx[ids], sum_fy[ids])
        local_fraction = np.divide(
            residual,
            sum_magnitude[ids],
            out=np.zeros_like(residual),
            where=sum_magnitude[ids] > 0,
        )
        frame_metrics = self.load_frame(position, frame)["metrics"]
        return {
            "position": position,
            "frame": frame,
            "definition": "spatial-bin sum of raw consistent nodal loads; an educational cancellation view, not stress",
            "bin_count_per_axis": bins,
            "center_xy_image_px": centers.tolist(),
            "fx_display_N": sum_fx[ids].tolist(),
            "fy_display_N": sum_fy[ids].tolist(),
            "residual_magnitude_N": residual.tolist(),
            "local_residual_fraction": local_fraction.tolist(),
            "metrics": frame_metrics,
        }

    def load_metrics(self, position: str) -> dict[str, Any]:
        directory = self.position_dir(position)
        metrics = _load_csv(str(directory / "fem/loads/frame_metrics.csv"))
        return {
            "position": position,
            "rows": [
                {key: _json_scalar(value) for key, value in row.items() if key != "position"}
                for row in metrics.to_dict(orient="records")
            ],
            "units_and_semantics": self.load_summary(position)["units_and_semantics"],
        }

    def diagnostics_report(self, position: str) -> dict[str, Any]:
        report = {
            "position": position,
            "metadata": self.metadata(position),
            "alignment": self.alignment(position),
            "coordinate_transform": self.coordinate_transform(position),
            "geometry": None,
            "mesh": None,
            "loads": None,
        }
        try:
            report["geometry"] = self.geometry_summary(position)
        except FileNotFoundError:
            pass
        try:
            report["mesh"] = self.mesh_summary(position)
        except FileNotFoundError:
            pass
        try:
            report["loads"] = self.load_summary(position)
        except FileNotFoundError:
            pass
        return report

    @staticmethod
    def _validate_frame(position: str, frame: int, frame_count: int) -> None:
        if frame < 0 or frame >= frame_count:
            raise FrameOutOfRange(
                f"frame {frame} is outside 0..{frame_count - 1} for {position}"
            )
