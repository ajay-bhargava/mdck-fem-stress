"""FastAPI application for the Dryad educational viewer and diagnostics."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import plotly  # noqa: E402
import pyvista  # noqa: E402
from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, Response  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402
from plotly.offline import get_plotlyjs  # noqa: E402

from src.web.data_service import DataService, FrameOutOfRange, PositionNotFound  # noqa: E402
from src.web.film import film_root  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DRYAD_DATA_DIR", PROJECT_ROOT / "data"))
TEMPLATE_ROOT = Path(__file__).resolve().parent / "templates"

app = FastAPI(
    title="Dryad epithelial monolayer explorer",
    description="Educational time-lapse viewer with read-only scientific diagnostics",
    version="0.5.0",
)
app.add_middleware(GZipMiddleware, minimum_size=1_000)
templates = Jinja2Templates(directory=TEMPLATE_ROOT)
service = DataService(DATA_ROOT)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, PositionNotFound):
        return HTTPException(status_code=404, detail=f"unknown position: {exc.args[0]}")
    if isinstance(exc, FrameOutOfRange):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail="data could not be read")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"positions": service.positions()},
    )


@app.get("/diagnostics/{position}", response_class=HTMLResponse)
def diagnostics_page(request: Request, position: str) -> HTMLResponse:
    try:
        report = service.diagnostics_report(position)
    except Exception as exc:
        raise _http_error(exc) from exc
    return templates.TemplateResponse(
        request=request,
        name="diagnostics.html",
        context={"report": report},
    )


@app.get("/film/{position}/{asset_path:path}")
def film_asset(position: str, asset_path: str) -> FileResponse:
    try:
        root = film_root(service, position).resolve()
    except Exception as exc:
        raise _http_error(exc) from exc
    path = (root / asset_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(status_code=404, detail="film asset not found")
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".json": "application/json",
    }
    return FileResponse(
        path,
        media_type=media_types.get(path.suffix.lower(), "application/octet-stream"),
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "data_root_available": DATA_ROOT.is_dir(),
        "position_count": len(service.positions()),
        "pyvista_available": True,
        "pyvista_version": pyvista.__version__,
        "vtk_version": list(pyvista.vtk_version_info),
        "pyvista_off_screen": os.environ.get("PYVISTA_OFF_SCREEN", "").lower()
        in {"1", "true", "yes"},
        "plotly_version": plotly.__version__,
    }


@app.get("/assets/plotly.min.js", include_in_schema=False)
def plotly_javascript() -> Response:
    return Response(content=get_plotlyjs(), media_type="application/javascript")


@app.get("/api/positions")
def positions() -> dict[str, object]:
    return {"positions": service.positions()}


@app.get("/api/positions/{position}/metadata")
def position_metadata(position: str) -> dict[str, object]:
    try:
        return service.metadata(position)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/alignment")
def position_alignment(position: str) -> dict[str, object]:
    try:
        return service.alignment(position)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/coordinate-transform")
def position_coordinate_transform(position: str) -> dict[str, object]:
    try:
        return service.coordinate_transform(position)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/grid-comparison")
def position_grid_comparison(position: str) -> dict[str, object]:
    try:
        return service.grid_comparison(position)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/geometry/summary")
def geometry_summary(position: str) -> dict[str, object]:
    try:
        return service.geometry_summary(position)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/geometry/domain")
def geometry_domain(
    position: str,
    simplify_px: float = Query(default=0.75, ge=0.0, le=5.0),
) -> dict[str, object]:
    try:
        return service.geometry_domain(position, simplify_px)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/geometry/cells")
def geometry_cells(
    position: str,
    simplify_px: float = Query(default=0.75, ge=0.0, le=5.0),
) -> dict[str, object]:
    try:
        return service.geometry_cells(position, simplify_px)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/geometry/cell/{cell_id}")
def geometry_cell(position: str, cell_id: int) -> dict[str, object]:
    try:
        return service.geometry_cell(position, cell_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/geometry/adjacency")
def geometry_adjacency(position: str) -> dict[str, object]:
    try:
        return service.geometry_adjacency(position)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/mesh/summary")
def mesh_summary(position: str) -> dict[str, object]:
    try:
        return service.mesh_summary(position)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/mesh")
def position_mesh(
    position: str,
    max_elements: int = Query(default=30_000, ge=1_000, le=100_000),
) -> dict[str, object]:
    try:
        return service.mesh(position, max_elements)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/mesh/quality")
def mesh_quality(
    position: str,
    metric: str = Query(default="quality_metric"),
    max_elements: int = Query(default=30_000, ge=1_000, le=100_000),
) -> dict[str, object]:
    try:
        return service.mesh_quality(position, metric, max_elements)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/mesh/boundary")
def mesh_boundary(position: str) -> dict[str, object]:
    try:
        return service.mesh_boundary(position)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/stress/series")
def stress_series(position: str) -> dict[str, object]:
    try:
        return service.stress_series_summary(position)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/stress/frame/{frame}.png")
def stress_series_image(position: str, frame: int, thickness_um: float = Query(default=5.0)) -> Response:
    try:
        return Response(service.stress_series_png(position, frame, thickness_um), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/stress/pilot")
def stress_pilot(position: str) -> dict[str, object]:
    try:
        return service.stress_pilot_summary(position)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/stress/pilot.png")
def stress_pilot_image(position: str) -> Response:
    try:
        return Response(service.stress_pilot_png(position), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/loads/summary")
def load_summary(position: str) -> dict[str, object]:
    try:
        return service.load_summary(position)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/loads/frame/{frame}")
def load_frame(position: str, frame: int) -> dict[str, object]:
    try:
        return service.load_frame(position, frame)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/loads/frame/{frame}/elements")
def load_elements(
    position: str,
    frame: int,
    max_elements: int = Query(default=20_000, ge=1_000, le=100_000),
) -> dict[str, object]:
    try:
        return service.load_elements(position, frame, max_elements)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/loads/frame/{frame}/nodes")
def load_nodes(
    position: str,
    frame: int,
    max_nodes: int = Query(default=2_000, ge=100, le=20_000),
) -> dict[str, object]:
    try:
        return service.load_nodes(position, frame, max_nodes)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/loads/metrics")
def load_metrics(position: str) -> dict[str, object]:
    try:
        return service.load_metrics(position)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/loads/frame/{frame}/balance-map")
def load_balance_map(
    position: str,
    frame: int,
    bins: int = Query(default=10, ge=4, le=20),
) -> dict[str, object]:
    try:
        return service.load_balance_map(position, frame, bins)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/frame/{frame}/dic")
def dic_frame(position: str, frame: int) -> dict[str, object]:
    try:
        return service.dic_frame(position, frame)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/frame/{frame}/traction")
def traction_frame(position: str, frame: int) -> dict[str, object]:
    try:
        return service.traction_frame(position, frame)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/frame/{frame}/nuclear-overlay.png")
def nuclear_overlay(position: str, frame: int) -> Response:
    try:
        return Response(service.nuclear_overlay_png(position, frame), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/source-image/{layer}/info")
def source_image_info(position: str, layer: str) -> dict[str, object]:
    try:
        return service.source_image_info(position, layer)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/frame/{frame}/source-image/{layer}.jpg")
def source_image_jpeg(position: str, frame: int, layer: str) -> Response:
    try:
        content = service.source_image_jpeg(position, frame, layer)
        return Response(
            content=content,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/frame/{frame}/image/{layer}")
def image_frame(position: str, frame: int, layer: str) -> dict[str, object]:
    try:
        return service.image_frame(position, frame, layer)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/positions/{position}/frame/{frame}/domain-outline")
def domain_outline(position: str, frame: int) -> dict[str, object]:
    try:
        return service.domain_outline(position, frame)
    except Exception as exc:
        raise _http_error(exc) from exc
