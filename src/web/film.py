"""Bake educational-film rasters for a static Cloudflare-compatible viewer.

Python runs only at bake time. The public page swaps pre-rendered images.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from skimage.segmentation import find_boundaries
from skimage.transform import resize

from src.web.data_service import DataService

FILM_VERSION = "v1"
DISPLAY_WIDTH = 1280
CHAPTERS = [
    "nuclei",
    "nuclear_segmentation",
    "fem_story",
    "fem_nodal_forces",
    "force_balance_story",
    "predicted_stress",
]


def film_root(service: DataService, position: str) -> Path:
    return service.position_dir(position) / "film" / FILM_VERSION


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _display_geometry(service: DataService, position: str) -> tuple[int, int, float]:
    info = service.source_image_info(position, "nuclei")
    scale = DISPLAY_WIDTH / info["width_px"]
    height = max(1, round(info["height_px"] * scale))
    return DISPLAY_WIDTH, height, scale


def _canvas(width: int, height: int) -> Image.Image:
    return Image.new("RGBA", (width, height), (0, 0, 0, 0))


def _png(image: Image.Image) -> bytes:
    from io import BytesIO

    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _draw_boundary(draw: ImageDraw.ImageDraw, boundary: dict[str, Any], scale: float, color: tuple[int, int, int, int], width: int) -> None:
    xs, ys = boundary["line_x"], boundary["line_y"]
    segment: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            if len(segment) >= 2:
                draw.line(segment, fill=color, width=width)
            segment = []
            continue
        segment.append((float(x) * scale, float(y) * scale))
    if len(segment) >= 2:
        draw.line(segment, fill=color, width=width)


def _arrow_scale(dx: np.ndarray, dy: np.ndarray, target_length: float) -> float:
    mags = np.hypot(dx, dy)
    finite = mags[np.isfinite(mags)]
    if not len(finite):
        return 0.0
    p95 = float(np.percentile(finite, 95))
    return target_length / p95 if p95 > 0 else 0.0


def _draw_arrows(
    draw: ImageDraw.ImageDraw,
    xy: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    scale: float,
    color: tuple[int, int, int, int],
    target_length: float,
    max_arrows: int = 400,
) -> None:
    stride = max(1, int(math.ceil(len(xy) / max_arrows)))
    ids = np.arange(0, len(xy), stride)
    factor = _arrow_scale(dx[ids], dy[ids], target_length)
    if factor == 0:
        return
    for i in ids:
        x0 = float(xy[i, 0]) * scale
        y0 = float(xy[i, 1]) * scale
        x1 = (float(xy[i, 0]) + float(dx[i]) * factor) * scale
        y1 = (float(xy[i, 1]) + float(dy[i]) * factor) * scale
        draw.line([(x0, y0), (x1, y1)], fill=color, width=2)
        angle = math.atan2(y1 - y0, x1 - x0)
        head = 7.0
        left = (x1 - head * math.cos(angle - 0.4), y1 - head * math.sin(angle - 0.4))
        right = (x1 - head * math.cos(angle + 0.4), y1 - head * math.sin(angle + 0.4))
        draw.polygon([(x1, y1), left, right], fill=color)


def _mix_color(fraction: float) -> tuple[int, int, int, int]:
    stops = [(0.0, (37, 99, 235)), (0.55, (250, 204, 21)), (1.0, (220, 38, 38))]
    t = min(1.0, max(0.0, float(fraction)))
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            rgb = tuple(int(a + (b - a) * u) for a, b in zip(c0, c1))
            return (*rgb, 230)
    return (*stops[-1][1], 230)


def render_nuclear_overlay(service: DataService, position: str, frame: int) -> bytes:
    import tifffile

    width, height, _scale = _display_geometry(service, position)
    labels = tifffile.imread(service.position_dir(position) / "geometry/labels.tif", key=frame)
    small = resize(
        labels,
        (height, width),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(labels.dtype, copy=False)
    edges = find_boundaries(small, mode="thick")
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[edges] = (97, 230, 206, 235)
    from io import BytesIO

    output = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(output, format="PNG", optimize=True)
    return output.getvalue()


def render_mesh_overlay(service: DataService, position: str) -> bytes:
    width, height, scale = _display_geometry(service, position)
    mesh = service.mesh(position, max_elements=14_000)
    boundary = service.mesh_boundary(position)
    image = _canvas(width, height)
    draw = ImageDraw.Draw(image)
    nodes = np.asarray(mesh["nodes_xy_image_px"], dtype=np.float64) * scale
    for triangle in mesh["triangles"]:
        points = [tuple(nodes[index]) for index in triangle]
        points.append(points[0])
        draw.line(points, fill=(255, 255, 255, 122), width=1)
    _draw_boundary(draw, boundary, scale, (245, 158, 11, 230), 3)
    return _png(image)


def render_nodal_overlay(service: DataService, position: str, frame: int) -> bytes:
    width, height, scale = _display_geometry(service, position)
    payload = service.load_nodes(position, frame, max_nodes=1_400)
    boundary = service.mesh_boundary(position)
    image = _canvas(width, height)
    draw = ImageDraw.Draw(image)
    _draw_boundary(draw, boundary, scale, (245, 158, 11, 230), 3)
    xy = np.asarray(payload["xy_image_px"], dtype=np.float64)
    dx = np.asarray(payload["fx_display_N"], dtype=np.float64)
    dy = np.asarray(payload["fy_display_N"], dtype=np.float64)
    for point in xy:
        x, y = float(point[0]) * scale, float(point[1]) * scale
        draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(239, 68, 68, 90))
    _draw_arrows(draw, xy, dx, dy, scale, (239, 68, 68, 230), target_length=60)
    return _png(image)


def render_balance_overlay(service: DataService, position: str, frame: int) -> bytes:
    width, height, scale = _display_geometry(service, position)
    payload = service.load_balance_map(position, frame, bins=10)
    boundary = service.mesh_boundary(position)
    image = _canvas(width, height)
    draw = ImageDraw.Draw(image)
    _draw_boundary(draw, boundary, scale, (255, 255, 255, 200), 3)
    centers = np.asarray(payload["center_xy_image_px"], dtype=np.float64)
    fractions = np.asarray(payload["local_residual_fraction"], dtype=np.float64)
    dx = np.asarray(payload["fx_display_N"], dtype=np.float64)
    dy = np.asarray(payload["fy_display_N"], dtype=np.float64)
    for center, fraction in zip(centers, fractions):
        radius = 4.0 + 10.0 * float(fraction)
        x, y = float(center[0]) * scale, float(center[1]) * scale
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=_mix_color(fraction),
            outline=(255, 255, 255, 220),
            width=1,
        )
    _draw_arrows(draw, centers, dx, dy, scale, (248, 250, 252, 230), target_length=90)
    return _png(image)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def build_manifest(service: DataService, position: str, frame_count: int) -> dict[str, Any]:
    summary = service.stress_series_summary(position)
    metadata = service.metadata(position)
    interval_s = float(metadata["imaging"].get("frame_interval_s") or summary.get("frame_interval_s") or 900)
    metrics = []
    table = service.load_metrics(position)["rows"]
    by_frame = {int(row["frame"]): row for row in table}
    for frame in range(frame_count):
        row = by_frame.get(frame, {})
        metrics.append(
            {
                "frame": frame,
                "source_net_force_fraction": row.get("source_net_force_fraction"),
            }
        )
    color_max = float(summary["thickness_comparison_color_max_pa"])
    return {
        "position": position,
        "film_version": FILM_VERSION,
        "frame_count": frame_count,
        "frame_interval_s": interval_s,
        "hours_per_frame": interval_s / 3600.0,
        "chapters": CHAPTERS,
        "thickness_options_um": summary["thickness_options_um"],
        "reference_thickness_um": float(summary["assumptions"]["thickness_m"]) * 1e6,
        "thickness_comparison_color_max_pa": color_max,
        "thickness_scaling": summary["thickness_scaling"],
        "stress_palette": "thickness-v1",
        "stress_color_note": (
            "0 saturates above the shared maximum absolute principal stress "
            f"({color_max:.6f} Pa) for every frame and thickness; triangles are not smoothed."
        ),
        "urls": {
            "nuclei": "nuclei/frame{frame:03d}.jpg",
            "nuclear_segmentation": "nuclear_segmentation/frame{frame:03d}.png",
            "fem_story": "mesh.png",
            "fem_nodal_forces": "nodal_forces/frame{frame:03d}.png",
            "force_balance_story": "force_balance/frame{frame:03d}.png",
            "predicted_stress": "stress/{thickness_um:g}um/frame{frame:03d}.png",
        },
        "caveats": [
            "Label_Image / labels.tif are StarDist nuclear masks, not cell boundaries.",
            "The FEM mesh is a fixed reference triangulation, not a deforming tissue.",
            "Mapped traction is an area-averaged load, not internal stress.",
            "Unit Pa is inferred; action/reaction sign is unresolved.",
            "The balance map bins nodal loads; it is not local mechanical equilibrium.",
            "Predicted stress is a conditional homogeneous plane-stress reconstruction (E=1000 Pa, nu=0.3), not validated tissue rheology.",
            "Thickness control is exact linear-model scaling sigma(h)=sigma(5 um)*5/h; membrane resultant unchanged. Not a biological response and not a gel/TFM reinversion.",
            "Auxiliary strains are not observed nuclear motion.",
        ],
        "frames": metrics,
    }


def build_position(service: DataService, position: str) -> Path:
    root = film_root(service, position)
    root.mkdir(parents=True, exist_ok=True)
    info = service.source_image_info(position, "nuclei")
    frame_count = int(info["frame_count"])
    nuclei_dir = root / "nuclei"
    overlay_dir = root / "nuclear_segmentation"
    nodal_dir = root / "nodal_forces"
    balance_dir = root / "force_balance"
    for folder in (nuclei_dir, overlay_dir, nodal_dir, balance_dir):
        folder.mkdir(exist_ok=True)

    mesh_path = root / "mesh.png"
    if not mesh_path.is_file():
        _atomic_write(mesh_path, render_mesh_overlay(service, position))
        print(f"{position}: mesh overlay ready", flush=True)

    for frame in range(frame_count):
        jpeg_path = nuclei_dir / f"frame{frame:03d}.jpg"
        if not jpeg_path.is_file():
            _atomic_write(jpeg_path, service.source_image_jpeg(position, frame, "nuclei"))
        overlay_path = overlay_dir / f"frame{frame:03d}.png"
        if not overlay_path.is_file():
            _atomic_write(overlay_path, render_nuclear_overlay(service, position, frame))
        nodal_path = nodal_dir / f"frame{frame:03d}.png"
        if not nodal_path.is_file():
            _atomic_write(nodal_path, render_nodal_overlay(service, position, frame))
        balance_path = balance_dir / f"frame{frame:03d}.png"
        if not balance_path.is_file():
            _atomic_write(balance_path, render_balance_overlay(service, position, frame))
        if frame % 10 == 0 or frame == frame_count - 1:
            print(f"{position}: frames 0-{frame} / {frame_count - 1}", flush=True)

    summary = service.stress_series_summary(position)
    stress_src = service.position_dir(position) / "fem/stress/time_series/display_thickness_v1"
    for thickness in summary["thickness_options_um"]:
        source_dir = stress_src / f"{thickness:g}um"
        dest_dir = root / "stress" / f"{thickness:g}um"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for frame in range(frame_count):
            source = source_dir / f"frame{frame:03d}.png"
            if not source.is_file():
                raise FileNotFoundError(f"missing precomputed stress PNG: {source}")
            _link_or_copy(source, dest_dir / f"frame{frame:03d}.png")
    shutil.copy2(stress_src / "provenance.json", root / "stress" / "provenance.json")

    manifest = build_manifest(service, position, frame_count)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{position}: manifest written ({frame_count} frames)", flush=True)
    return root


def export_site(service: DataService, position: str, site_root: Path, index_html: Path) -> Path:
    source = film_root(service, position)
    if not (source / "manifest.json").is_file():
        raise FileNotFoundError("film bake is missing; run build_film_display first")
    site_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(index_html, site_root / "index.html")
    for item in source.iterdir():
        destination = site_root / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)
    print(f"static site exported to {site_root}", flush=True)
    return site_root
