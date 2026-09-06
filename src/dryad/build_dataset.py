#!/usr/bin/env python3
"""Build a standardized, FEM-ready (but unsolved) dataset from Dryad files."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from dryad.io_mat import load_mat_variables, save_centroids_npz, save_numeric_npz
from dryad.metadata import (
    make_position_metadata,
    parse_dataset_readme,
    parse_experimental_settings,
    write_yaml,
)
from dryad.validate import (
    ValidationIssue,
    validate_dic_arrays,
    validate_output_position,
    validate_required_sources,
    validate_tiff,
    validate_traction_arrays,
)


LOGGER = logging.getLogger("dryad.build")
DEFAULT_EXPECTED_FRAMES = 97

DIC_FILE = "DIC_c1_64w0_16d0.mat"
TRACTION_FILE = "tractions_DIC_c1_64w0.mat"
CENTROIDS_FILE = "Centroids_to_Voronoi.mat"
TRAJECTORIES_FILE = "KNN_results_JN_refine.mat"

DIC_FIELDS = ("x", "y", "u", "v", "c_peak", "w0", "d0", "inc")
TRACTION_FIELDS = ("x", "y", "u", "v", "tx", "ty", "w0", "d0", "inc")
CENTROID_FIELDS = (
    "centroids",
    "areas",
    "perimeters",
    "shape_index",
    "count_vertices",
    "Tsx",
    "Tsy",
    "Avg_shape_index",
    "NNZ_shape_index",
    "Avg_amp_T_in",
    "Avg_amp_T_in_by_area",
    "startFrame",
    "endFrame",
    "pix_size",
)
CENTROID_CELL_FIELDS = (
    "centroids",
    "areas",
    "perimeters",
    "shape_index",
    "count_vertices",
    "Tsx",
    "Tsy",
    "NNZ_shape_index",
)
TRAJECTORY_FIELDS = (
    "traj_x_retained",
    "traj_y_retained",
    "traj_x_retained_atleastT",
    "traj_y_retained_atleastT",
    "min_surviv_hr",
    "start_frame",
    "end_frame",
)


@dataclass
class PositionResult:
    position: str
    source_dir: str
    metadata: dict[str, Any]
    facts: dict[str, Any] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="immutable Dryad source root")
    parser.add_argument("--output", type=Path, required=True, help="standardized output root")
    parser.add_argument("--positions", nargs="+", help="position names to process")
    parser.add_argument(
        "--overwrite", action="store_true", help="atomically replace existing position outputs"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate existing outputs without writing or rebuilding",
    )
    return parser.parse_args()


def relative_to_working_directory(path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), Path.cwd().resolve())).as_posix()


def discover_positions(source_root: Path, requested: list[str] | None) -> list[str]:
    available = {
        path.name
        for path in source_root.iterdir()
        if path.is_dir() and path.name.startswith("pos") and path.name[3:].isdigit()
    }
    if requested:
        invalid = [name for name in requested if name not in available]
        if invalid:
            raise ValueError(f"positions not found under {source_root}: {', '.join(invalid)}")
        selected = set(requested)
    else:
        selected = available
    return sorted(selected, key=lambda name: int(name[3:]))


def make_source_index(source_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(source_dir.iterdir(), key=lambda item: item.name):
        if path.is_file():
            files.append(
                {
                    "filename": path.name,
                    "relative_path": relative_to_working_directory(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {
        "position": source_dir.name,
        "source_dir": relative_to_working_directory(source_dir),
        "files": files,
    }


def make_image_paths(source_dir: Path, documentation: dict[str, Any]) -> dict[str, Any]:
    mapping = documentation.get("channel_mapping", {})

    def mapped(semantic_name: str, raw_fallback: str) -> str | None:
        filename = mapping.get(semantic_name)
        if filename is None:
            return None
        return relative_to_working_directory(source_dir / filename)

    return {
        "beads_loaded": mapped("beads_loaded", "c1.tif"),
        "beads_reference": mapped("beads_reference", "c1_ref.tif"),
        "cell_body": mapped("cell_body", "c2.tif"),
        "nuclei": mapped("nuclei", "nuc.tif"),
        "domain": relative_to_working_directory(source_dir / "domain.tif"),
        "labels": relative_to_working_directory(source_dir / "Label_Image.tif"),
        "mapping_status": documentation.get("channel_mapping_status", "unverified"),
        "raw_names": {
            "c1": relative_to_working_directory(source_dir / "c1.tif"),
            "c1_ref": relative_to_working_directory(source_dir / "c1_ref.tif"),
            "c2": relative_to_working_directory(source_dir / "c2.tif"),
            "nuc": relative_to_working_directory(source_dir / "nuc.tif"),
        },
    }


def fallback_metadata(position: str) -> dict[str, Any]:
    return {
        "position": position,
        "imaging": {
            "n_frames": None,
            "pixel_size_m": None,
            "pixel_size_um": None,
            "frame_interval_min": None,
        },
        "substrate": {
            "young_modulus_pa": None,
            "poisson_ratio": None,
            "thickness_m": None,
            "thickness_um": None,
        },
        "monolayer": {"poisson_ratio": None, "thickness_m": None},
        "geometry": {"type": "unknown", "source_flag": None},
        "coordinates": {
            "source_coordinate_units": "unknown",
            "image_origin": "top_left",
            "fem_origin": "bottom_left",
            "y_axis_transform_required": True,
        },
        "units": {
            "length": "m",
            "displacement": "source_preserved",
            "traction": "source_preserved",
            "stress": "Pa",
            "time": "unknown",
        },
    }


def build_position(
    source_dir: Path,
    output_root: Path,
    documentation: dict[str, Any],
    *,
    overwrite: bool,
) -> PositionResult:
    position = source_dir.name
    final_dir = output_root / position
    expected_frames = documentation.get("n_frames") or DEFAULT_EXPECTED_FRAMES

    if final_dir.exists() and not overwrite:
        LOGGER.info(
            "%s: output exists; refreshing source references and validating without rebuilding",
            position,
        )
        # Source locations may change independently of the converted arrays.
        # Refresh only the lightweight path indexes; large outputs remain untouched.
        write_yaml(final_dir / "source_index.yaml", make_source_index(source_dir))
        write_yaml(
            final_dir / "images/source_paths.yaml",
            make_image_paths(source_dir, documentation),
        )
        metadata = load_existing_metadata(final_dir, position)
        issues = validate_required_sources(source_dir)
        output_issues, facts = validate_output_position(final_dir, expected_frames)
        issues.extend(output_issues)
        result = PositionResult(
            position,
            relative_to_working_directory(source_dir),
            metadata,
            facts,
            deduplicate_issues(issues),
        )
        log_issues(position, result.issues)
        return result

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{position}.", dir=output_root))
    issues = validate_required_sources(source_dir)
    metadata = fallback_metadata(position)
    try:
        try:
            settings = parse_experimental_settings(source_dir / "ExperimentalSettings.txt")
            metadata = make_position_metadata(position, settings, documentation)
        except Exception as exc:
            issues.append(ValidationIssue("error", "metadata", str(exc)))

        write_yaml(temp_dir / "metadata.yaml", metadata)
        write_yaml(temp_dir / "source_index.yaml", make_source_index(source_dir))
        write_yaml(temp_dir / "images/source_paths.yaml", make_image_paths(source_dir, documentation))

        convert_dic(source_dir, temp_dir, expected_frames, issues)
        convert_traction(source_dir, temp_dir, expected_frames, issues)
        convert_centroids(source_dir, temp_dir, issues)
        convert_trajectories(source_dir, temp_dir, issues)
        copy_geometry_tiffs(source_dir, temp_dir, issues)

        output_issues, facts = validate_output_position(temp_dir, expected_frames)
        issues.extend(output_issues)
        issues = deduplicate_issues(issues)

        if final_dir.exists():
            shutil.rmtree(final_dir)
        temp_dir.replace(final_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    result = PositionResult(
        position,
        relative_to_working_directory(source_dir),
        metadata,
        facts,
        issues,
    )
    log_issues(position, result.issues)
    return result


def convert_dic(
    source_dir: Path,
    output_dir: Path,
    expected_frames: int | None,
    issues: list[ValidationIssue],
) -> None:
    path = source_dir / DIC_FILE
    if not path.is_file():
        return
    try:
        arrays = load_mat_variables(path, DIC_FIELDS)
        issues.extend(validate_dic_arrays(arrays, expected_frames))
        save_numeric_npz(
            output_dir / "displacement/published_dic.npz",
            arrays,
            source_filename=DIC_FILE,
        )
    except Exception as exc:
        issues.append(ValidationIssue("error", "dic_conversion", str(exc)))


def convert_traction(
    source_dir: Path,
    output_dir: Path,
    expected_frames: int | None,
    issues: list[ValidationIssue],
) -> None:
    path = source_dir / TRACTION_FILE
    if not path.is_file():
        return
    try:
        arrays = load_mat_variables(path, TRACTION_FIELDS)
        issues.extend(validate_traction_arrays(arrays, expected_frames))
        save_numeric_npz(
            output_dir / "traction/published.npz",
            arrays,
            source_filename=TRACTION_FILE,
        )
    except Exception as exc:
        issues.append(ValidationIssue("error", "traction_conversion", str(exc)))


def convert_centroids(
    source_dir: Path, output_dir: Path, issues: list[ValidationIssue]
) -> None:
    path = source_dir / CENTROIDS_FILE
    if not path.is_file():
        return
    try:
        arrays = load_mat_variables(path, CENTROID_FIELDS)
        save_centroids_npz(
            output_dir / "geometry/centroids.npz",
            arrays,
            cell_fields=CENTROID_CELL_FIELDS,
            source_filename=CENTROIDS_FILE,
        )
    except Exception as exc:
        issues.append(ValidationIssue("error", "centroid_conversion", str(exc)))


def convert_trajectories(
    source_dir: Path, output_dir: Path, issues: list[ValidationIssue]
) -> None:
    path = source_dir / TRAJECTORIES_FILE
    if not path.is_file():
        return
    try:
        arrays = load_mat_variables(path, TRAJECTORY_FIELDS)
        x_shape = arrays["traj_x_retained"].shape
        y_shape = arrays["traj_y_retained"].shape
        atleast_x_shape = arrays["traj_x_retained_atleastT"].shape
        atleast_y_shape = arrays["traj_y_retained_atleastT"].shape
        if x_shape != y_shape or atleast_x_shape != atleast_y_shape:
            issues.append(
                ValidationIssue(
                    "error",
                    "trajectory_shape",
                    f"trajectory x/y shapes differ: {x_shape}/{y_shape}, "
                    f"atleastT {atleast_x_shape}/{atleast_y_shape}",
                )
            )
        save_numeric_npz(
            output_dir / "geometry/trajectories.npz",
            arrays,
            source_filename=TRAJECTORIES_FILE,
        )
    except Exception as exc:
        issues.append(ValidationIssue("error", "trajectory_conversion", str(exc)))


def copy_geometry_tiffs(
    source_dir: Path, output_dir: Path, issues: list[ValidationIssue]
) -> None:
    geometry_dir = output_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    for source_name, output_name in (("domain.tif", "domain.tif"), ("Label_Image.tif", "labels.tif")):
        source_path = source_dir / source_name
        if not source_path.is_file():
            continue
        tiff_issues, _ = validate_tiff(source_path, source_name)
        issues.extend(tiff_issues)
        if tiff_issues:
            continue
        try:
            shutil.copy2(source_path, geometry_dir / output_name)
        except OSError as exc:
            issues.append(
                ValidationIssue("error", "tiff_copy", f"failed to copy {source_name}: {exc}")
            )


def load_existing_metadata(output_dir: Path, position: str) -> dict[str, Any]:
    path = output_dir / "metadata.yaml"
    if not path.is_file():
        return fallback_metadata(position)
    try:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data if isinstance(data, dict) else fallback_metadata(position)
    except Exception:
        return fallback_metadata(position)


def validate_existing_position(
    source_dir: Path, output_root: Path, expected_frames: int | None
) -> PositionResult:
    position = source_dir.name
    output_dir = output_root / position
    metadata = load_existing_metadata(output_dir, position)
    issues = validate_required_sources(source_dir)
    output_issues, facts = validate_output_position(output_dir, expected_frames)
    issues.extend(output_issues)
    issues = deduplicate_issues(issues)
    log_issues(position, issues)
    return PositionResult(
        position,
        relative_to_working_directory(source_dir),
        metadata,
        facts,
        issues,
    )


def deduplicate_issues(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    return list(dict.fromkeys(issues))


def log_issues(position: str, issues: list[ValidationIssue]) -> None:
    recoverable_source_codes = {
        "missing_source",
        "metadata",
        "dic_conversion",
        "traction_conversion",
        "centroid_conversion",
        "trajectory_conversion",
        "tiff_open",
        "tiff_copy",
    }
    for issue in issues:
        message = f"{position}: [{issue.code}] {issue.message}"
        if issue.code in recoverable_source_codes:
            # The manifest can still mark required-data failures as errors while
            # the log makes clear that processing will continue to other positions.
            LOGGER.warning(message)
        elif issue.severity == "error":
            LOGGER.error(message)
        else:
            LOGGER.warning(message)


def manifest_row(result: PositionResult) -> dict[str, Any]:
    imaging = result.metadata.get("imaging", {})
    substrate = result.metadata.get("substrate", {})
    monolayer = result.metadata.get("monolayer", {})
    warnings = [issue for issue in result.issues if issue.severity == "warning"]
    errors = [issue for issue in result.issues if issue.severity == "error"]
    facts = result.facts
    return {
        "position": result.position,
        "source_dir": result.source_dir,
        "n_frames": imaging.get("n_frames"),
        "pixel_size_m": imaging.get("pixel_size_m"),
        "substrate_E_pa": substrate.get("young_modulus_pa"),
        "substrate_nu": substrate.get("poisson_ratio"),
        "substrate_thickness_m": substrate.get("thickness_m"),
        "monolayer_nu": monolayer.get("poisson_ratio"),
        "monolayer_thickness_m": monolayer.get("thickness_m"),
        "dic_shape_y": facts.get("dic_shape_y"),
        "dic_shape_x": facts.get("dic_shape_x"),
        "traction_shape_y": facts.get("traction_shape_y"),
        "traction_shape_x": facts.get("traction_shape_x"),
        "has_domain": facts.get("has_domain", False),
        "has_labels": facts.get("has_labels", False),
        "has_dic": facts.get("has_dic", False),
        "has_traction": facts.get("has_traction", False),
        "has_centroids": facts.get("has_centroids", False),
        "has_trajectories": facts.get("has_trajectories", False),
        "status": "failed" if errors else ("warning" if warnings else "success"),
        "warning_count": len(warnings),
        "error_count": len(errors),
        "status_messages": " | ".join(
            f"{issue.severity}:{issue.code}:{issue.message}" for issue in result.issues
        ),
    }


def configure_logging(output_root: Path, validate_only: bool) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(levelname)s %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    LOGGER.addHandler(console)
    if not validate_only:
        output_root.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_root / "build.log", mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)


def write_manifest(path: Path, results: list[PositionResult]) -> None:
    rows = [manifest_row(result) for result in results]
    temporary = path.with_name(f".{path.name}.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()

    if not source_root.is_dir():
        raise SystemExit(f"source directory does not exist: {args.source}")
    if source_root == output_root or source_root in output_root.parents:
        raise SystemExit("output must not be the source directory or a directory inside it")
    if args.validate_only and args.overwrite:
        raise SystemExit("--validate-only and --overwrite cannot be used together")

    configure_logging(output_root, args.validate_only)
    try:
        positions = discover_positions(source_root, args.positions)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    documentation = parse_dataset_readme(source_root / "README.txt")
    expected_frames = documentation.get("n_frames") or DEFAULT_EXPECTED_FRAMES
    results: list[PositionResult] = []

    for position in positions:
        LOGGER.info("%s: %s", position, "validating" if args.validate_only else "processing")
        source_dir = source_root / position
        try:
            if args.validate_only:
                result = validate_existing_position(source_dir, output_root, expected_frames)
            else:
                result = build_position(
                    source_dir,
                    output_root,
                    documentation,
                    overwrite=args.overwrite,
                )
        except Exception as exc:
            LOGGER.exception("%s: unexpected processing failure", position)
            result = PositionResult(
                position,
                relative_to_working_directory(source_dir),
                fallback_metadata(position),
                {},
                [ValidationIssue("error", "unexpected", str(exc))],
            )
        results.append(result)

    if not args.validate_only:
        write_manifest(output_root / "manifest.csv", results)

    warning_count = sum(
        issue.severity == "warning" for result in results for issue in result.issues
    )
    failed_count = sum(result.failed for result in results)
    print(f"Processed positions: {len(results)}")
    print(f"Successful: {len(results) - failed_count}")
    print(f"Warnings: {warning_count}")
    print(f"Failed: {failed_count}")
    print(f"Manifest: {relative_to_working_directory(output_root / 'manifest.csv')}")
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
