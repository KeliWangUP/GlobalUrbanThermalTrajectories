import json
from pathlib import Path
from typing import Any

import ee


LST_BANDS = [
    "terra_day",
    "terra_night",
    "aqua_day",
    "aqua_night"
]

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the JSON configuration used by the GEE scripts."""
    path = Path(config_path)
    if not path.is_absolute():
        path = CONFIG_PATH.parent / path
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    print(f"[INFO] Config loaded from: {path}")
    print(f"[INFO] Config version: {config['project']['version']}")
    return config


def initialize_gee(config: dict[str, Any]) -> None:
    """Initialize Earth Engine using the project configured in JSON."""
    project_id = config["project"]["gee_project"]
    try:
        ee.Initialize(project=project_id)
    except Exception as exc:
        print(f"[ERROR] Earth Engine initialization failed: {exc}")
        raise

    print("[INFO] Earth Engine initialized.")
    print(f"[INFO] GEE project: {project_id}")


def get_modis_reference_grid(
    config: dict[str, Any]
) -> tuple[ee.Projection, float]:
    """Get the reference MODIS projection and native pixel scale."""
    grid_config = config["reference_grid"]
    reference = (
        ee.ImageCollection(grid_config["collection"])
        .filterDate(
            grid_config["reference_date"],
            ee.Date(grid_config["reference_date"]).advance(1, "day")
        )
        .first()
    )
    projection = ee.Image(reference).select(
        grid_config["band"]
    ).projection()
    projection_info = projection.getInfo()
    native_scale = projection.nominalScale().getInfo()

    print("[INFO] MODIS reference grid loaded.")
    print(f"[INFO] Collection: {grid_config['collection']}")
    print(f"[INFO] Band: {grid_config['band']}")
    print(f"[INFO] CRS: {projection_info['crs']}")
    print(f"[INFO] Native scale: {native_scale:.3f} m")
    return projection, native_scale


def build_spatial_reducer() -> ee.Reducer:
    """Build the shared spatial summary reducer."""
    return (
        ee.Reducer.mean()
        .combine(
            reducer2=ee.Reducer.median(),
            sharedInputs=True
        )
        .combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True
        )
        .combine(
            reducer2=ee.Reducer.count(),
            sharedInputs=True
        )
    )


def reduce_monthly_lst(
    image: ee.Image,
    regions: ee.FeatureCollection,
    year: int,
    month: int,
    projection: ee.Projection,
    scale: float,
    tile_scale: int
) -> ee.FeatureCollection:
    """Calculate monthly regional LST statistics and attach period fields."""
    result = image.select(LST_BANDS).reduceRegions(
        collection=regions,
        reducer=build_spatial_reducer(),
        crs=projection,
        scale=scale,
        tileScale=tile_scale
    )
    return result.map(
        lambda feature: feature.set({
            "year": year,
            "month": month
        }).setGeometry(None)
    )


def export_table_to_drive(
    collection: ee.FeatureCollection,
    description: str,
    folder: str,
    file_name: str
) -> ee.batch.Task:
    """Submit an Earth Engine CSV export task to Google Drive."""
    task = ee.batch.Export.table.toDrive(
        collection=collection,
        description=description,
        folder=folder,
        fileNamePrefix=file_name,
        fileFormat="CSV"
    )
    task.start()
    print(f"[EXPORT] {description}")
    print(f"[TASK] {task.id}")
    return task


def build_yearly_lst_statistics(
    year: int,
    regions: ee.FeatureCollection,
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float,
    tile_scale: int,
    spatial_mask: ee.Image | None = None
) -> ee.FeatureCollection:
    """Build and combine the twelve monthly regional LST summaries."""
    from src.modis_lst_utils import build_monthly_four_slot_image

    monthly_results = []
    for month in range(1, 13):
        monthly_image = build_monthly_four_slot_image(
            year=year,
            month=month,
            config=config,
            reference_projection=projection
        )
        if spatial_mask is not None:
            monthly_image = monthly_image.updateMask(spatial_mask)

        monthly_results.append(
            reduce_monthly_lst(
                image=monthly_image,
                regions=regions,
                year=year,
                month=month,
                projection=projection,
                scale=scale,
                tile_scale=tile_scale
            )
        )

    return ee.FeatureCollection(monthly_results).flatten()


def run_annual_exports(
    dataset_name: str,
    regions: ee.FeatureCollection,
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float,
    spatial_mask: ee.Image | None = None
) -> None:
    """Submit one annual CSV export for each configured study year."""
    study_config = config["study"]
    tile_scale = config["processing"]["tile_scale"]
    export_config = config["export"]

    for year in range(
        study_config["start_year"],
        study_config["end_year"] + 1
    ):
        print(f"[INFO] Building {dataset_name}: {year}")
        annual_collection = build_yearly_lst_statistics(
            year=year,
            regions=regions,
            config=config,
            projection=projection,
            scale=scale,
            tile_scale=tile_scale,
            spatial_mask=spatial_mask
        )
        export_table_to_drive(
            collection=annual_collection,
            description=f"{dataset_name}_{year}",
            folder=export_config["folder"],
            file_name=f"{dataset_name}_{year}"
        )
