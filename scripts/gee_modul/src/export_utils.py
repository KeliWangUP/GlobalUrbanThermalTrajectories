import json
from pathlib import Path
from typing import Any

import ee
from src.modis_lst_utils import build_monthly_four_slot_image

LST_BANDS = [
    "terra_day",
    "terra_night",
    "aqua_day",
    "aqua_night"
]

QC_BANDS = [
    "terra_day_clear_days",
    "terra_day_n8day",
    "terra_night_clear_days",
    "terra_night_n8day",
    "aqua_day_clear_days",
    "aqua_day_n8day",
    "aqua_night_clear_days",
    "aqua_night_n8day"
]


LST_STATISTICS = [
    "mean",
    "median",
    "stdDev",
    "count"
]


LST_OUTPUT_FIELDS = [
    f"{band}_{stat}"
    for band in LST_BANDS
    for stat in LST_STATISTICS
]


QC_OUTPUT_FIELDS = [
    f"{band}_mean"
    for band in QC_BANDS
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


def asset_exists(asset_id: str) -> bool:
    """Return True when an Earth Engine asset exists and is readable."""
    try:
        ee.data.getAsset(asset_id)
    except ee.EEException:
        return False
    return True


def get_modis_reference_grid(
    config: dict[str, Any]
) -> tuple[ee.Projection, float]:
    """Get the native MODIS projection and nominal pixel scale."""

    grid_config = config["reference_grid"]

    reference = (
        ee.ImageCollection(grid_config["collection"])
        .select(grid_config["band"])
        .first()
    )

    projection = ee.Image(reference).projection()

    projection_info = projection.getInfo()
    native_scale = projection.nominalScale().getInfo()

    print("[INFO] MODIS reference grid loaded.")
    print(f"[INFO] Collection: {grid_config['collection']}")
    print(f"[INFO] Band: {grid_config['band']}")
    print(f"[INFO] CRS: {projection_info['crs']}")
    print(f"[INFO] Native scale: {native_scale:.3f} m")

    return projection, native_scale

def build_export_selectors(
    id_field: str
) -> list[str]:
    """
    Build a fixed output schema for annual CSV exports.
    """

    return [
        id_field,
        "year",
        "month",
        *LST_OUTPUT_FIELDS,
        *QC_OUTPUT_FIELDS
    ]


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


def build_qc_reducer() -> ee.Reducer:
    """
    Build the spatial reducer for monthly QC variables.

    QC variables are summarized using the spatial mean only.
    """

    return ee.Reducer.mean()


def reduce_monthly_lst(
    image: ee.Image,
    regions: ee.FeatureCollection,
    year: int,
    month: int,
    projection: ee.Projection,
    scale: float,
    tile_scale: int,
    id_field: str
) -> ee.FeatureCollection:
    """
    Calculate monthly regional LST statistics and QC summaries.

    LST statistics
    --------------
    mean
    median
    stdDev
    count

    QC statistics
    -------------
    spatial mean of:
    - clear-sky observation count
    - number of valid A2 composites
    """

    # Keep only the region identifier.
    regions_clean = regions.select(
        [id_field]
    )

    # ==================================================
    # 1. LST statistics
    # ==================================================

    lst_stats = (
        image
        .select(LST_BANDS)
        .reduceRegions(
            collection=regions_clean,
            reducer=build_spatial_reducer(),
            crs=projection,
            scale=scale,
            tileScale=tile_scale
        )
    )

    # ==================================================
    # 2. QC statistics
    # ==================================================

    qc_stats = (
        image
        .select(QC_BANDS)
        .reduceRegions(
            collection=regions_clean,
            reducer=build_qc_reducer(),
            crs=projection,
            scale=scale,
            tileScale=tile_scale
        )
    )

    # With ee.Reducer.mean() alone, Earth Engine generally
    # preserves the input band name as the output property.
    #
    # Therefore:
    # terra_day_clear_days
    # becomes:
    # terra_day_clear_days
    #
    # We explicitly rename the properties later to *_mean.

    # ==================================================
    # 3. Join by city ID
    # ==================================================

    joined = ee.Join.saveFirst(
        matchKey="qc"
    ).apply(
        primary=lst_stats,
        secondary=qc_stats,
        condition=ee.Filter.equals(
            leftField=id_field,
            rightField=id_field
        )
    )

    joined = ee.FeatureCollection(joined)

    # ==================================================
    # 4. Construct clean output
    # ==================================================

    def _format_feature(feature):

        feature = ee.Feature(feature)

        qc = ee.Feature(
            feature.get("qc")
        )

        # Output names must match LST_OUTPUT_FIELDS and QC_OUTPUT_FIELDS.
        output = {
            id_field: feature.get(id_field),
            "year": year,
            "month": month
        }
        for field in LST_OUTPUT_FIELDS:
            output[field] = feature.get(field)
        for qc_band in QC_BANDS:
            output[f"{qc_band}_mean"] = qc.get(qc_band)

        return ee.Feature(
            None,
            ee.Dictionary(output)
        )

    return joined.map(
        _format_feature
    )


def export_table_to_drive(
    collection: ee.FeatureCollection,
    description: str,
    folder: str,
    file_name: str,
    selectors: list[str]
) -> ee.batch.Task:
    """Submit a schema-explicit Earth Engine CSV export task to Google Drive."""
    task = ee.batch.Export.table.toDrive(
        collection=collection,
        description=description,
        folder=folder,
        fileNamePrefix=file_name,
        fileFormat="CSV",
        selectors=selectors
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
    id_field: str,
    spatial_mask: ee.Image | None = None
) -> ee.FeatureCollection:
    """Build and combine the twelve monthly regional LST summaries."""

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
                tile_scale=tile_scale,
                id_field=id_field
            )
        )
        print(f"[BUILD] {year}-{month:02d}")

    return ee.FeatureCollection(monthly_results).flatten()


def run_annual_exports(
    dataset_name: str,
    regions: ee.FeatureCollection,
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float,
    spatial_mask: ee.Image | None = None
) -> list[ee.batch.Task]:
    """Submit one annual CSV export for each configured study year."""

    study_config = config["study"]
    tile_scale = config["processing"]["tile_scale"]
    export_config = config["export"]
    id_field = config["cities"]["id_field"]
    version = config["project"]["version"]

    tasks = []

    for year in range(
        study_config["start_year"],
        study_config["end_year"] + 1
    ):
        print(f"\n[INFO] Building {dataset_name}: {year}")

        annual_collection = build_yearly_lst_statistics(
            year=year,
            regions=regions,
            config=config,
            projection=projection,
            scale=scale,
            tile_scale=tile_scale,
            id_field=id_field,
            spatial_mask=spatial_mask
        )

        task = export_table_to_drive(
            collection=annual_collection,
            description=f"{dataset_name}_{version}_{year}",
            folder=export_config["folder"],
            file_name=f"{dataset_name}_{version}_{year}",
            selectors=build_export_selectors(id_field)
        )

        tasks.append(task)

    print(f"[INFO] Submitted {len(tasks)} export tasks.")

    return tasks