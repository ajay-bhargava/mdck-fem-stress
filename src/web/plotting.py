"""Server-side preparation of lightweight Plotly-ready representations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from skimage.measure import find_contours


def json_array(array: np.ndarray) -> list[Any]:
    """Convert an array to strict-JSON values, replacing NaN/Inf with null."""
    values = np.asarray(array)
    if np.issubdtype(values.dtype, np.floating) and not np.all(np.isfinite(values)):
        as_object = values.astype(object)
        as_object[~np.isfinite(values)] = None
        return as_object.tolist()
    return values.tolist()


def vector_stride(shape: tuple[int, int], target_arrows_per_axis: int = 20) -> int:
    return max(1, int(math.ceil(max(shape) / target_arrows_per_axis)))


def mechanical_frame_payload(
    *,
    position: str,
    frame: int,
    x: np.ndarray,
    y: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    first_name: str,
    second_name: str,
    pixel_size_m: float | None,
    extra_scalar: tuple[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Create full scalar fields plus independently decimated vectors."""
    magnitude = np.hypot(first, second)
    stride = vector_stride(x.shape)
    sparse = (slice(None, None, stride), slice(None, None, stride))
    payload: dict[str, Any] = {
        "position": position,
        "frame": frame,
        "shape": list(x.shape),
        "coordinates": {
            "source_origin": "MATLAB_one_based_top_left",
            "source_y_direction": "down",
            "source_x": json_array(x),
            "source_y": json_array(y),
            "display_x_zero_based": json_array(x.astype(np.float64) - 1.0),
            "display_y_zero_based": json_array(y.astype(np.float64) - 1.0),
            "pixel_size_m": pixel_size_m,
            "fem_y_flip_applied": False,
        },
        first_name: json_array(first),
        second_name: json_array(second),
        "magnitude": json_array(magnitude),
        "vectors": {
            "stride": stride,
            "source_x": json_array(x[sparse]),
            "source_y": json_array(y[sparse]),
            "display_x_zero_based": json_array(x[sparse].astype(np.float64) - 1.0),
            "display_y_zero_based": json_array(y[sparse].astype(np.float64) - 1.0),
            first_name: json_array(first[sparse]),
            second_name: json_array(second[sparse]),
        },
    }
    if extra_scalar is not None:
        name, values = extra_scalar
        payload[name] = json_array(values)
    return payload


def image_frame_payload(
    image: np.ndarray,
    *,
    position: str,
    frame: int,
    layer: str,
    max_dimension: int = 512,
) -> dict[str, Any]:
    """Downsample an image for JSON display, retaining native index coordinates."""
    height, width = image.shape
    stride = max(1, int(math.ceil(max(height, width) / max_dimension)))
    sampled = image[::stride, ::stride]
    return {
        "position": position,
        "frame": frame,
        "layer": layer,
        "native_shape": [height, width],
        "display_shape": list(sampled.shape),
        "downsample_stride": stride,
        "x_zero_based": np.arange(0, width, stride, dtype=np.int64).tolist(),
        "y_zero_based": np.arange(0, height, stride, dtype=np.int64).tolist(),
        "values": json_array(sampled),
        "rendering": "categorical" if layer == "labels" else "binary",
        "origin": "top_left",
        "y_direction": "down",
    }


def domain_outline_payload(
    domain: np.ndarray,
    *,
    position: str,
    frame: int,
    max_points_per_contour: int = 500,
) -> dict[str, Any]:
    contours_payload = []
    for contour in find_contours(np.asarray(domain) > 0, level=0.5):
        stride = max(1, int(math.ceil(len(contour) / max_points_per_contour)))
        sampled = contour[::stride]
        contours_payload.append(
            {
                "x_zero_based": sampled[:, 1].tolist(),
                "y_zero_based": sampled[:, 0].tolist(),
                "native_point_count": int(len(contour)),
                "display_stride": stride,
            }
        )
    return {
        "position": position,
        "frame": frame,
        "origin": "top_left",
        "y_direction": "down",
        "contours": contours_payload,
    }


def _grid_boundary(x: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
    rows = [0, 0, -1, -1, 0]
    cols = [0, -1, -1, 0, 0]
    return {
        "source_x": [float(x[row, col]) for row, col in zip(rows, cols, strict=True)],
        "source_y": [float(y[row, col]) for row, col in zip(rows, cols, strict=True)],
        "display_x_zero_based": [
            float(x[row, col]) - 1.0 for row, col in zip(rows, cols, strict=True)
        ],
        "display_y_zero_based": [
            float(y[row, col]) - 1.0 for row, col in zip(rows, cols, strict=True)
        ],
    }


def grid_comparison_payload(
    *,
    position: str,
    dic_x: np.ndarray,
    dic_y: np.ndarray,
    traction_x: np.ndarray,
    traction_y: np.ndarray,
    alignment: dict[str, Any],
    point_stride: int = 8,
) -> dict[str, Any]:
    dic_sample = (slice(None, None, point_stride), slice(None, None, point_stride))
    traction_sample = (slice(None, None, point_stride), slice(None, None, point_stride))

    def points(x: np.ndarray, y: np.ndarray, sample: tuple[slice, slice]) -> dict[str, Any]:
        source_x = x[sample].ravel().astype(np.float64)
        source_y = y[sample].ravel().astype(np.float64)
        return {
            "source_x": source_x.tolist(),
            "source_y": source_y.tolist(),
            "display_x_zero_based": (source_x - 1.0).tolist(),
            "display_y_zero_based": (source_y - 1.0).tolist(),
        }

    return {
        "position": position,
        "point_stride": point_stride,
        "coordinate_display": "zero_based_top_left_image_coordinates",
        "dic": {
            "shape": list(dic_x.shape),
            "boundary": _grid_boundary(dic_x, dic_y),
            "sample_points": points(dic_x, dic_y, dic_sample),
            "median_dx": alignment["dic"]["median_dx"],
            "median_dy": alignment["dic"]["median_dy"],
        },
        "traction": {
            "shape": list(traction_x.shape),
            "boundary": _grid_boundary(traction_x, traction_y),
            "sample_points": points(traction_x, traction_y, traction_sample),
            "median_dx": alignment["traction"]["median_dx"],
            "median_dy": alignment["traction"]["median_dy"],
        },
        "mapping": alignment["grid_mapping"],
    }
