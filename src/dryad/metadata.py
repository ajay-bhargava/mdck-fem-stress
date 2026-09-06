"""Parse source metadata and write standardized YAML documents."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


_VALUE_LINE = re.compile(r"^\s*([^\s]+)\s*---\s*(.*?)\s*$")


def parse_experimental_settings(path: Path) -> dict[str, Any]:
    """Parse the documented seven-line ExperimentalSettings text format."""
    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        match = _VALUE_LINE.match(line)
        if not match:
            raise ValueError(f"{path}:{line_number}: malformed settings line: {line!r}")
        raw_value, description = match.groups()
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: non-numeric setting {raw_value!r}"
            ) from exc
        entries.append(
            {
                "raw_value": raw_value,
                "value": value,
                "description": description,
            }
        )

    found: dict[str, dict[str, Any]] = {}
    for entry in entries:
        description = entry["description"].lower()
        if "pixel size" in description:
            found["pixel_size"] = entry
        elif "substrate young" in description:
            found["substrate_young_modulus"] = entry
        elif "substrate poisson" in description:
            found["substrate_poisson_ratio"] = entry
        elif "substrate thickness" in description:
            found["substrate_thickness"] = entry
        elif "monolayer poisson" in description:
            found["monolayer_poisson_ratio"] = entry
        elif "monolayer thickness" in description:
            found["monolayer_thickness"] = entry
        elif "strip" in description and ("hole" in description or "island" in description):
            found["geometry_flag"] = entry

    required = {
        "pixel_size",
        "substrate_young_modulus",
        "substrate_poisson_ratio",
        "substrate_thickness",
        "monolayer_poisson_ratio",
        "monolayer_thickness",
        "geometry_flag",
    }
    missing = sorted(required - found.keys())
    if missing:
        raise ValueError(f"{path}: missing settings: {', '.join(missing)}")

    geometry_flag = int(found["geometry_flag"]["value"])
    if found["geometry_flag"]["value"] != geometry_flag:
        raise ValueError(f"{path}: geometry flag must be an integer")

    return {
        "pixel_size_m": found["pixel_size"]["value"],
        "substrate_young_modulus_pa": found["substrate_young_modulus"]["value"],
        "substrate_poisson_ratio": found["substrate_poisson_ratio"]["value"],
        "substrate_thickness_um": found["substrate_thickness"]["value"],
        "monolayer_poisson_ratio": found["monolayer_poisson_ratio"]["value"],
        "monolayer_thickness_m": found["monolayer_thickness"]["value"],
        "geometry_flag": geometry_flag,
        "source_entries": entries,
    }


def parse_dataset_readme(path: Path) -> dict[str, Any]:
    """Extract only explicitly documented acquisition facts from README.txt."""
    if not path.exists():
        return {
            "n_frames": None,
            "frame_interval_min": None,
            "channel_mapping": {},
            "channel_mapping_status": "unverified",
        }

    text = path.read_text(encoding="utf-8-sig")
    interval_match = re.search(r"every\s+(\d+(?:\.\d+)?)\s+minutes", text, re.I)
    frames_match = re.search(r"\(=\s*(\d+)\s+images\)", text, re.I)

    channel_patterns = {
        "c1.tif": r"^\s*c1\.tif\s*-\s*time stack of beads\s*$",
        "c1_ref.tif": r"^\s*c1_ref\.tif\s*-\s*good reference beads",
        "c2.tif": r"^\s*c2\.tif\s*-\s*time stack of cells\s*$",
        "nuc.tif": r"^\s*nuc\.tif\s*-\s*time stack of nucs",
    }
    documented = {
        filename: bool(re.search(pattern, text, re.I | re.M))
        for filename, pattern in channel_patterns.items()
    }
    mapping_verified = all(documented.values())
    mapping = (
        {
            "beads_loaded": "c1.tif",
            "beads_reference": "c1_ref.tif",
            "cell_body": "c2.tif",
            "nuclei": "nuc.tif",
        }
        if mapping_verified
        else {}
    )
    return {
        "n_frames": int(frames_match.group(1)) if frames_match else None,
        "frame_interval_min": float(interval_match.group(1)) if interval_match else None,
        "channel_mapping": mapping,
        "channel_mapping_status": (
            "verified_from_dataset_readme" if mapping_verified else "unverified"
        ),
    }


def make_position_metadata(
    position: str,
    settings: dict[str, Any],
    documentation: dict[str, Any],
) -> dict[str, Any]:
    geometry_types = {1: "strip", 2: "island"}
    geometry_type = geometry_types.get(settings["geometry_flag"], "unknown")
    thickness_um = settings["substrate_thickness_um"]
    pixel_size_m = settings["pixel_size_m"]

    return {
        "position": position,
        "imaging": {
            "n_frames": documentation.get("n_frames"),
            "pixel_size_m": pixel_size_m,
            "pixel_size_um": pixel_size_m * 1.0e6,
            "frame_interval_min": documentation.get("frame_interval_min"),
        },
        "substrate": {
            "young_modulus_pa": settings["substrate_young_modulus_pa"],
            "poisson_ratio": settings["substrate_poisson_ratio"],
            "thickness_m": thickness_um * 1.0e-6,
            "thickness_um": thickness_um,
        },
        "monolayer": {
            "poisson_ratio": settings["monolayer_poisson_ratio"],
            "thickness_m": settings["monolayer_thickness_m"],
        },
        "geometry": {
            "type": geometry_type,
            "source_flag": settings["geometry_flag"],
        },
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
        "source_metadata": {
            "experimental_settings_file": "ExperimentalSettings.txt",
            "dataset_documentation_file": "README.txt",
            "experimental_settings_entries": settings["source_entries"],
            "channel_mapping_status": documentation.get(
                "channel_mapping_status", "unverified"
            ),
        },
    }


def write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    temporary.replace(path)
