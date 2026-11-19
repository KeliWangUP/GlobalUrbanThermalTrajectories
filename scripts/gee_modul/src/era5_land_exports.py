from collections import Counter
from datetime import datetime, timezone
from typing import Any

import ee

from src.batch_utils import filter_nonempty_geometries
from src.export_utils import export_table_to_drive

OUTPUT_FIELDS = ["city_id", "month", "temperature_2m_c"]


def study_date_range(config: dict[str, Any]) -> tuple[str, str]:
    """Return an inclusive-start, exclusive-end date range for whole years."""
    study = config["study"]
    start_year, end_year = study["start_year"], study["end_year"]
    if (
        type(start_year) is not int
        or type(end_year) is not int
        or start_year > end_year
    ):
        raise ValueError(
            "study.start_year and study.end_year must be ordered integers."
        )
    return f"{start_year}-01-01", f"{end_year + 1}-01-01"


def expected_month_labels(start_year: int, end_year: int) -> list[str]:
    """Build calendar-month labels for an inclusive range of years."""
    return [
        f"{year:04d}-{month:02d}"
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    ]


def validate_monthly_source_coverage(
    timestamps: list[int | float],
    expected_months: list[str]
) -> None:
    """Require one ERA5 monthly image for every expected calendar month."""
    observed = [
        datetime.fromtimestamp(
            timestamp / 1000,
            tz=timezone.utc
        ).strftime("%Y-%m")
        for timestamp in timestamps
    ]
    counts = Counter(observed)
    duplicates = sorted(month for month, count in counts.items() if count > 1)
    missing = sorted(set(expected_months) - set(observed))
    unexpected = sorted(set(observed) - set(expected_months))
    if duplicates or missing or unexpected:
        raise ValueError(
            "ERA5-Land monthly coverage does not match the configured "
            f"calendar months; missing={missing[:5]}, "
            f"duplicate={duplicates[:5]}, unexpected={unexpected[:5]}."
        )


def build_city_bounding_boxes(
    halos: ee.FeatureCollection,
    city_ids: ee.List,
    id_field: str,
    max_error_m: int
) -> ee.FeatureCollection:
    """Union each city's halo geometries and return its enclosing rectangle."""

    def _city_bbox(city_id: ee.ComputedObject) -> ee.Feature:
        city_halos = halos.filter(ee.Filter.eq(id_field, city_id))
        bounds = city_halos.geometry(max_error_m).bounds(max_error_m)
        return ee.Feature(bounds, {"city_id": city_id})

    return ee.FeatureCollection(city_ids.map(_city_bbox))


def build_yearly_era5_rows(
    collection: ee.ImageCollection,
    city_bboxes: ee.FeatureCollection,
    year: int,
    temperature_band: str,
    kelvin_offset: float,
    projection: ee.Projection,
    tile_scale: int
) -> ee.FeatureCollection:
    """Reduce each calendar month's Celsius image over every city bbox."""
    monthly_results = []

    for month in range(1, 13):
        month_start = f"{year:04d}-{month:02d}-01"
        month_end = (
            ee.Date(month_start)
            .advance(1, "month")
        )
        image = (
            ee.Image(
                collection.filterDate(month_start, month_end).first()
            )
            .select(temperature_band)
            .add(kelvin_offset)
            .rename("temperature_2m_c")
        )
        month_label = f"{year:04d}-{month:02d}"
        reduced = image.reduceRegions(
            collection=city_bboxes,
            reducer=ee.Reducer.mean(),
            crs=projection,
            tileScale=tile_scale
        )

        def _format_row(feature: ee.Feature) -> ee.Feature:
            feature = ee.Feature(feature)
            properties = ee.Dictionary({
                "city_id": feature.get("city_id"),
                "month": month_label,
                "temperature_2m_c": feature.get("mean")
            })
            return ee.Feature(None, properties)

        monthly_results.append(reduced.map(_format_row))

    return ee.FeatureCollection(monthly_results).flatten()


def summarize_sample_rows(
    features: list[dict[str, Any]],
    city_ids: list[int | str],
    months: list[str]
) -> dict[str, Any]:
    """Summarize sample output for row, null-value, and key checks."""
    properties = [feature.get("properties", {}) for feature in features]
    key_counts = Counter(
        (str(row.get("city_id")), str(row.get("month")))
        for row in properties
        if row.get("city_id") is not None and row.get("month") is not None
    )
    expected_keys = {
        (str(city_id), month)
        for city_id in city_ids
        for month in months
    }
    found_keys = set(key_counts)
    return {
        "city_count": len({str(row.get("city_id")) for row in properties
                           if row.get("city_id") is not None}),
        "row_count": len(properties),
        "missing_values": sum(
            row.get("temperature_2m_c") is None for row in properties
        ),
        "missing_keys": len(expected_keys - found_keys),
        "duplicate_keys": sum(count > 1 for count in key_counts.values()),
    }


def print_sample_diagnostics(
    collection: ee.FeatureCollection,
    city_ids: list[int | str],
    months: list[str]
) -> None:
    """Evaluate the small sample and print the requested diagnostics."""
    response = collection.getInfo()
    features = response.get("features", [])
    summary = summarize_sample_rows(features, city_ids, months)
    print(f"[SAMPLE] City count: {summary['city_count']}")
    print(
        f"[SAMPLE] Month range: {months[0]} to {months[-1]} "
        f"({len(months)} months)"
    )
    print(f"[SAMPLE] Row count: {summary['row_count']}")
    print(f"[SAMPLE] Missing temperature values: {summary['missing_values']}")
    print(f"[SAMPLE] Missing city-month keys: {summary['missing_keys']}")
    print(f"[SAMPLE] Duplicate city-month keys: {summary['duplicate_keys']}")
    print("[SAMPLE] Example rows:")
    for feature in features[:10]:
        print(feature.get("properties", {}))


def run_era5_land_export(config: dict[str, Any]) -> list[ee.batch.Task]:
    """Run sample diagnostics or submit one full-study CSV export per year."""
    study = config["study"]
    era5_config = config["era5_land"]
    export_config = config["era5_land_export"]
    cities_config = config["cities"]
    start_date, end_date = study_date_range(config)
    mode = export_config["mode"]
    if mode not in ("sample", "full"):
        raise ValueError("era5_land_export.mode must be 'sample' or 'full'.")

    max_error_m = export_config["max_error_m"]
    tile_scale = export_config["tile_scale"]
    if type(max_error_m) is not int or max_error_m < 1:
        raise ValueError("era5_land_export.max_error_m must be a positive integer.")
    if type(tile_scale) is not int or tile_scale < 1:
        raise ValueError("era5_land_export.tile_scale must be a positive integer.")

    source = ee.ImageCollection(era5_config["collection"]).filterDate(
        start_date,
        end_date
    )
    source = source.select(era5_config["temperature_band"])
    expected_months = expected_month_labels(
        study["start_year"],
        study["end_year"]
    )
    timestamps = source.aggregate_array("system:time_start").getInfo()
    validate_monthly_source_coverage(timestamps, expected_months)
    print(
        f"[INFO] ERA5-Land source: {era5_config['collection']}; "
        f"band: {era5_config['temperature_band']} (K)."
    )
    print(
        f"[INFO] Calendar coverage: {start_date} inclusive to "
        f"{end_date} exclusive; months {expected_months[0]}-"
        f"{expected_months[-1]} ({len(expected_months)} months)."
    )

    projection = ee.Image(source.first()).projection()
    projection_info = projection.getInfo()
    native_scale = projection.nominalScale().getInfo()
    print(
        f"[INFO] ERA5 native grid: {projection_info['crs']}; "
        f"nominal scale {native_scale:.2f} m."
    )

    halos = ee.FeatureCollection(
        config["rural"]["static_assets"]["halo_asset"]
    ).filter(ee.Filter.notNull([cities_config["id_field"]]))
    halos = filter_nonempty_geometries(halos, max_error_m)
    city_ids = (
        halos.aggregate_array(cities_config["id_field"])
        .distinct()
        .sort()
    )
    city_count = city_ids.size().getInfo()
    if city_count == 0:
        raise ValueError("No non-empty rural halo city IDs were found.")

    if mode == "sample":
        sample_city_count = export_config["sample_city_count"]
        sample_year = export_config["sample_year"]
        if type(sample_city_count) is not int or sample_city_count < 1:
            raise ValueError(
                "era5_land_export.sample_city_count must be a positive integer."
            )
        if (
            type(sample_year) is not int
            or not study["start_year"] <= sample_year <= study["end_year"]
        ):
            raise ValueError(
                "era5_land_export.sample_year must be within study years."
            )
        if city_count < sample_city_count:
            raise ValueError(
                f"Sample requests {sample_city_count} cities, but only "
                f"{city_count} non-empty halos are available."
            )
        selected_ids = city_ids.slice(0, sample_city_count).getInfo()
        selected_city_ids = ee.List(selected_ids)
        selected_months = expected_month_labels(sample_year, sample_year)
        print(
            f"[INFO] Sample mode: {sample_city_count} cities; "
            f"{sample_year}-01 through {sample_year}-12."
        )
        bboxes = build_city_bounding_boxes(
            halos, selected_city_ids, cities_config["id_field"], max_error_m
        )
        sample_rows = build_yearly_era5_rows(
            source,
            bboxes,
            sample_year,
            era5_config["temperature_band"],
            era5_config["kelvin_to_celsius_offset"],
            projection,
            tile_scale
        )
        print_sample_diagnostics(sample_rows, selected_ids, selected_months)
        return []

    bboxes = build_city_bounding_boxes(
        halos, city_ids, cities_config["id_field"], max_error_m
    )
    print(
        f"[INFO] Full mode: {city_count} cities; "
        f"{len(expected_months)} city-months per city over the study period."
    )
    tasks = []
    dataset_name = export_config["dataset_name"]
    folder = export_config["folder"]
    version = config["project"]["version"]
    for year in range(study["start_year"], study["end_year"] + 1):
        rows = build_yearly_era5_rows(
            source,
            bboxes,
            year,
            era5_config["temperature_band"],
            era5_config["kelvin_to_celsius_offset"],
            projection,
            tile_scale
        )
        export_name = f"{dataset_name}_{version}_{year}"
        task = export_table_to_drive(
            collection=rows,
            description=export_name,
            folder=folder,
            file_name=export_name,
            selectors=OUTPUT_FIELDS
        )
        tasks.append(task)
    print(
        f"[INFO] Submitted {len(tasks)} annual CSV tasks for "
        f"{city_count} cities."
    )
    return tasks
