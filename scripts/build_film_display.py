"""Bake pos01 educational-film rasters and a static site folder.

Run: uv run python -m scripts.build_film_display
Does not deploy or upload.
"""

from pathlib import Path

from src.web.data_service import DataService
from src.web.film import build_position, export_site


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    service = DataService(project / "data")
    positions = service.positions()
    if positions != ["pos01"]:
        raise SystemExit(f"expected only pos01, found {positions}")
    build_position(service, "pos01")
    export_site(
        service,
        "pos01",
        project / "site",
        project / "src/web/templates/index.html",
    )


if __name__ == "__main__":
    main()
