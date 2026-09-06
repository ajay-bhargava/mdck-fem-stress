"""MATLAB v5 readers and lossless NPZ conversion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from scipy.io import loadmat


MATLAB_METADATA_KEYS = {"__header__", "__version__", "__globals__"}


def load_mat_variables(path: Path, required: Iterable[str]) -> dict[str, np.ndarray]:
    """Load selected MATLAB variables and fail with a useful missing-key error."""
    required = tuple(required)
    data = loadmat(path, variable_names=required, squeeze_me=False, struct_as_record=False)
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"{path}: missing MATLAB variables: {', '.join(missing)}")
    return {name: np.asarray(data[name]) for name in required}


def save_numeric_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    source_filename: str,
) -> None:
    """Store numeric MATLAB arrays without squeezing, casting, or reordering."""
    payload = {name: np.asarray(value) for name, value in arrays.items()}
    payload["format_version"] = np.asarray("1.0")
    payload["source_filename"] = np.asarray(source_filename)
    _savez(path, payload)


def pack_matlab_cell(name: str, cell: np.ndarray) -> dict[str, np.ndarray]:
    """Pack a MATLAB cell array as values, offsets, and per-cell shapes.

    No object arrays are emitted, so consumers can load the NPZ with
    ``allow_pickle=False``. Values are flattened in NumPy C order and can be
    reconstructed using the shape recorded for each cell.
    """
    cell_array = np.asarray(cell)
    if cell_array.dtype != object:
        raise TypeError(f"{name} is not a MATLAB cell array")

    items = [np.asarray(item) for item in cell_array.ravel(order="C")]
    if not items:
        value_dtype = np.dtype(np.float64)
    else:
        if any(item.dtype == object for item in items):
            raise TypeError(f"{name} contains nested or non-numeric cell values")
        value_dtype = np.result_type(*(item.dtype for item in items))

    offsets = np.zeros(len(items) + 1, dtype=np.int64)
    for index, item in enumerate(items):
        offsets[index + 1] = offsets[index] + item.size

    if offsets[-1]:
        values = np.concatenate(
            [item.astype(value_dtype, copy=False).ravel(order="C") for item in items]
        )
    else:
        values = np.empty(0, dtype=value_dtype)

    max_ndim = max((item.ndim for item in items), default=0)
    shapes = np.full((len(items), max_ndim), -1, dtype=np.int64)
    ndims = np.empty(len(items), dtype=np.int8)
    for index, item in enumerate(items):
        ndims[index] = item.ndim
        shapes[index, : item.ndim] = item.shape

    return {
        f"{name}_values": values,
        f"{name}_offsets": offsets,
        f"{name}_shapes": shapes,
        f"{name}_ndims": ndims,
        f"{name}_cell_shape": np.asarray(cell_array.shape, dtype=np.int64),
    }


def save_centroids_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    cell_fields: Iterable[str],
    source_filename: str,
) -> None:
    """Save centroid/Voronoi data, replacing MATLAB cells with ragged arrays."""
    cell_fields = set(cell_fields)
    payload: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if name in cell_fields:
            payload.update(pack_matlab_cell(name, value))
        else:
            value = np.asarray(value)
            if value.dtype == object:
                raise TypeError(f"unexpected object-valued MATLAB variable: {name}")
            payload[name] = value
    payload["format_version"] = np.asarray("1.0")
    payload["ragged_encoding"] = np.asarray("values_offsets_shapes")
    payload["source_filename"] = np.asarray(source_filename)
    _savez(path, payload)


def _savez(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    temporary.replace(path)
