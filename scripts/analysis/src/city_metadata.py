from collections.abc import Mapping
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


CITY_ID_COLUMN = "ID_UC_G0"
CENTROID_LON_COLUMN = "centroid_lon"
CENTROID_LAT_COLUMN = "centroid_lat"
CITY_AREA_KM2_COLUMN = "city_area_km2"
COUNTRY_COLUMN = "country"
REGION_COLUMN = "GC_DEV_USR"
GHSL_AREA_COLUMN = "GC_UCA_KM2"
GHSL_COUNTRY_COLUMN = "GC_CNT_GAD"
KOPPEN_CODE_COLUMN = "koppen_geiger_code"
KOPPEN_CLASS_COLUMN = "koppen_geiger_class"
KOPPEN_MACRO_GROUP_COLUMN = "koppen_geiger_macro_group"
BUILT_DEVELOPMENT_YEARS = (2000, 2005, 2010, 2015, 2020)
BUILT_SURFACE_2000_COLUMN = "built_surface_2000_m2"
BUILT_SURFACE_SLOPE_COLUMN = "built_surface_slope_m2_per_year"
BUILT_VOLUME_2000_COLUMN = "built_volume_total_2000_m3"
BUILT_VOLUME_SLOPE_COLUMN = "built_volume_total_slope_m3_per_year"
CLIMATE_CLASS_BY_CODE = {
    1: "Af",
    2: "Am",
    3: "Aw",
    4: "BWh",
    5: "BWk",
    6: "BSh",
    7: "BSk",
    8: "Csa",
    9: "Csb",
    10: "Csc",
    11: "Cwa",
    12: "Cwb",
    13: "Cwc",
    14: "Cfa",
    15: "Cfb",
    16: "Cfc",
    17: "Dsa",
    18: "Dsb",
    19: "Dsc",
    20: "Dsd",
    21: "Dwa",
    22: "Dwb",
    23: "Dwc",
    24: "Dwd",
    25: "Dfa",
    26: "Dfb",
    27: "Dfc",
    28: "Dfd",
    29: "ET",
    30: "EF"
}


def _sample_koppen_geiger(
    raster_path: Path,
    longitudes: np.ndarray,
    latitudes: np.ndarray
) -> tuple[pd.arrays.IntegerArray, list[str | None]]:
    if not raster_path.is_file():
        raise FileNotFoundError(
            f"Köppen-Geiger raster not found: {raster_path}"
        )
    if len(longitudes) != len(latitudes):
        raise ValueError("Centroid longitude and latitude lengths differ.")

    with rasterio.open(raster_path) as raster:
        if raster.count != 1:
            raise ValueError(
                "Köppen-Geiger raster must contain exactly one band: "
                f"{raster_path}"
            )
        if raster.crs is None or raster.crs.to_epsg() != 4326:
            raise ValueError(
                "Köppen-Geiger raster must use EPSG:4326: "
                f"{raster_path} (found {raster.crs})."
            )

        coordinates = zip(longitudes.tolist(), latitudes.tolist())
        climate_codes: list[int | None] = []
        for sample in raster.sample(coordinates, indexes=1, masked=True):
            value = sample[0]
            if np.ma.is_masked(value):
                climate_codes.append(None)
                continue

            code = int(value)
            if code == 0:
                climate_codes.append(None)
            elif code not in CLIMATE_CLASS_BY_CODE:
                raise ValueError(
                    "Köppen-Geiger raster contains an unsupported class "
                    f"code {code}: {raster_path}"
                )
            else:
                climate_codes.append(code)

    classes = [
        CLIMATE_CLASS_BY_CODE.get(code) if code is not None else None
        for code in climate_codes
    ]
    return pd.array(climate_codes, dtype="Int64"), classes


def _macro_group_by_code(
    cz_macro_groups: Mapping[str, tuple[int, ...]]
) -> dict[int, str]:
    code_to_group = {}
    for group, codes in cz_macro_groups.items():
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Climate macro-group names must be non-empty.")
        for code in codes:
            if (
                isinstance(code, bool)
                or not isinstance(code, int)
                or code not in CLIMATE_CLASS_BY_CODE
            ):
                raise ValueError(
                    f"Invalid class code in climate macro-group '{group}': "
                    f"{code!r}"
                )
            if code in code_to_group:
                raise ValueError(
                    f"Köppen-Geiger code {code} appears in multiple groups."
                )
            code_to_group[code] = group
    if set(code_to_group) != set(CLIMATE_CLASS_BY_CODE):
        missing = sorted(set(CLIMATE_CLASS_BY_CODE) - set(code_to_group))
        raise ValueError(
            "Climate macro-groups must cover codes 1..30 exactly once; "
            f"missing codes: {missing}."
        )
    return code_to_group


def _read_built_development_metrics(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"GHSL built-development CSV not found: {path}")

    input_columns = [
        CITY_ID_COLUMN,
        *(
            f"built_surface_sum_{year}_m2"
            for year in BUILT_DEVELOPMENT_YEARS
        ),
        *(
            f"built_volume_total_sum_{year}_m3"
            for year in BUILT_DEVELOPMENT_YEARS
        )
    ]
    source_columns = pd.read_csv(path, nrows=0).columns
    missing_columns = set(input_columns) - set(source_columns)
    if missing_columns:
        raise ValueError(
            "GHSL built-development CSV is missing required columns: "
            f"{sorted(missing_columns)}: {path}"
        )

    source = pd.read_csv(path, usecols=input_columns)
    if source[CITY_ID_COLUMN].isna().any():
        raise ValueError(f"GHSL built-development CSV has null city IDs: {path}")
    if source[CITY_ID_COLUMN].duplicated().any():
        raise ValueError(
            f"GHSL built-development CSV has duplicate city IDs: {path}"
        )

    metric_values: dict[str, np.ndarray] = {}
    for metric, unit in (
        ("built_surface_sum", "m2"),
        ("built_volume_total_sum", "m3")
    ):
        columns = [
            f"{metric}_{year}_{unit}"
            for year in BUILT_DEVELOPMENT_YEARS
        ]
        values = source[columns].apply(pd.to_numeric, errors="raise")
        matrix = values.to_numpy(dtype="float64")
        if (
            not np.isfinite(matrix).all()
            or (matrix < 0).any()
        ):
            raise ValueError(
                f"GHSL built-development values must be finite and "
                f"non-negative in {columns}: {path}"
            )
        metric_values[metric] = matrix

    centered_years = np.asarray(
        BUILT_DEVELOPMENT_YEARS, dtype="float64"
    )
    centered_years -= centered_years.mean()
    denominator = float(centered_years @ centered_years)
    surface_values = metric_values["built_surface_sum"]
    volume_values = metric_values["built_volume_total_sum"]
    return pd.DataFrame({
        CITY_ID_COLUMN: source[CITY_ID_COLUMN].to_numpy(),
        BUILT_SURFACE_2000_COLUMN: surface_values[:, 0],
        BUILT_SURFACE_SLOPE_COLUMN: (
            surface_values @ centered_years / denominator
        ),
        BUILT_VOLUME_2000_COLUMN: volume_values[:, 0],
        BUILT_VOLUME_SLOPE_COLUMN: (
            volume_values @ centered_years / denominator
        )
    })


def _merge_built_development_metrics(
    metadata: pd.DataFrame,
    built_development_csv_path: Path
) -> pd.DataFrame:
    metrics = _read_built_development_metrics(built_development_csv_path)
    metadata_ids = set(metadata[CITY_ID_COLUMN])
    metrics_ids = set(metrics[CITY_ID_COLUMN])
    missing = metadata_ids - metrics_ids
    extra = metrics_ids - metadata_ids
    if missing or extra:
        raise ValueError(
            "GHSL built-development city IDs do not match city-boundary "
            f"IDs; missing metrics for {len(missing)} cities "
            f"(examples: {sorted(missing)[:10]}), and "
            f"{len(extra)} extra metric IDs (examples: "
            f"{sorted(extra)[:10]})."
        )
    return metadata.merge(
        metrics,
        on=CITY_ID_COLUMN,
        how="left",
        validate="one_to_one",
        sort=False
    )


def build_city_metadata(
    shapefile_path: Path,
    climate_raster_path: Path,
    cz_macro_groups: Mapping[str, tuple[int, ...]],
    id_column: str = CITY_ID_COLUMN,
    built_development_csv_path: Path | None = None
) -> tuple[pd.DataFrame, int]:
    """Calculate projected city centroids and sample their climate classes."""
    if not shapefile_path.is_file():
        raise FileNotFoundError(
            f"City boundary shapefile not found: {shapefile_path}"
        )

    cities = gpd.read_file(shapefile_path)
    if id_column not in cities.columns:
        raise ValueError(
            f"City boundary file is missing ID column '{id_column}': "
            f"{shapefile_path}"
        )
    required_source_columns = {
        GHSL_AREA_COLUMN,
        GHSL_COUNTRY_COLUMN,
        REGION_COLUMN
    }
    missing_source_columns = required_source_columns - set(cities.columns)
    if missing_source_columns:
        raise ValueError(
            "City boundary file is missing required GHSL attributes: "
            f"{sorted(missing_source_columns)}: {shapefile_path}"
        )
    if cities.crs is None:
        raise ValueError(
            f"City boundary file has no declared CRS: {shapefile_path}"
        )
    if cities[id_column].isna().any():
        raise ValueError(f"City IDs in {shapefile_path} must not be null.")
    if cities[id_column].duplicated().any():
        examples = cities.loc[
            cities[id_column].duplicated(keep=False),
            id_column
        ].head(5).tolist()
        raise ValueError(
            f"City IDs in {shapefile_path} must be unique; "
            f"examples: {examples}"
        )

    area_km2 = pd.to_numeric(cities[GHSL_AREA_COLUMN], errors="raise")
    if (
        area_km2.isna().any()
        or not np.isfinite(area_km2.to_numpy(dtype="float64")).all()
        or (area_km2 <= 0).any()
    ):
        raise ValueError(
            f"GHSL {GHSL_AREA_COLUMN} must contain finite positive km² "
            f"values: {shapefile_path}"
        )
    country = cities[GHSL_COUNTRY_COLUMN].astype("string").str.strip()
    region = cities[REGION_COLUMN].astype("string").str.strip()
    if country.isna().any() or country.eq("").any():
        raise ValueError(
            f"GHSL {GHSL_COUNTRY_COLUMN} contains missing country names: "
            f"{shapefile_path}"
        )
    if region.isna().any() or region.eq("").any():
        raise ValueError(
            f"GHSL {REGION_COLUMN} contains missing region names: "
            f"{shapefile_path}"
        )

    geometries = cities.geometry.copy()
    if geometries.isna().any() or geometries.is_empty.any():
        raise ValueError(
            f"City boundary file contains null or empty geometries: "
            f"{shapefile_path}"
        )
    invalid_geometry_count = int((~geometries.is_valid).sum())
    if invalid_geometry_count:
        invalid = ~geometries.is_valid
        geometries.loc[invalid] = geometries.loc[invalid].make_valid()
        if (~geometries.is_valid).any():
            raise ValueError(
                f"Could not repair all invalid geometries in "
                f"{shapefile_path}."
            )

    projected_centroids = geometries.centroid
    geographic_centroids = gpd.GeoSeries(
        projected_centroids,
        index=cities.index,
        crs=cities.crs
    ).to_crs("EPSG:4326")
    climate_codes, climate_classes = _sample_koppen_geiger(
        climate_raster_path,
        geographic_centroids.x.to_numpy(),
        geographic_centroids.y.to_numpy()
    )
    climate_code_to_group = _macro_group_by_code(cz_macro_groups)
    climate_macro_groups = [
        climate_code_to_group.get(int(code)) if not pd.isna(code) else None
        for code in climate_codes
    ]
    metadata = pd.DataFrame({
        CITY_ID_COLUMN: cities[id_column].to_numpy(),
        CENTROID_LON_COLUMN: geographic_centroids.x.to_numpy(),
        CENTROID_LAT_COLUMN: geographic_centroids.y.to_numpy(),
        CITY_AREA_KM2_COLUMN: area_km2.to_numpy(dtype="float64"),
        COUNTRY_COLUMN: country.to_numpy(),
        REGION_COLUMN: region.to_numpy(),
        KOPPEN_CODE_COLUMN: climate_codes,
        KOPPEN_CLASS_COLUMN: climate_classes,
        KOPPEN_MACRO_GROUP_COLUMN: climate_macro_groups
    })

    if (
        metadata[CENTROID_LON_COLUMN].isna().any()
        or not np.isfinite(
            metadata[CENTROID_LON_COLUMN].to_numpy(dtype="float64")
        ).all()
        or not metadata[CENTROID_LON_COLUMN].between(-180, 180).all()
    ):
        raise ValueError(
            f"Centroid longitude is invalid for {shapefile_path}."
        )
    if (
        metadata[CENTROID_LAT_COLUMN].isna().any()
        or not np.isfinite(
            metadata[CENTROID_LAT_COLUMN].to_numpy(dtype="float64")
        ).all()
        or not metadata[CENTROID_LAT_COLUMN].between(-90, 90).all()
    ):
        raise ValueError(
            f"Centroid latitude is invalid for {shapefile_path}."
        )
    if built_development_csv_path is not None:
        metadata = _merge_built_development_metrics(
            metadata, built_development_csv_path
        )
    return metadata, invalid_geometry_count


def merge_city_metadata(
    panel: pd.DataFrame,
    metadata: pd.DataFrame,
    cz_macro_groups: Mapping[str, tuple[int, ...]]
) -> pd.DataFrame:
    """Attach authoritative centroid and climate attributes to panel rows."""
    required_metadata = {
        CITY_ID_COLUMN,
        CENTROID_LON_COLUMN,
        CENTROID_LAT_COLUMN,
        CITY_AREA_KM2_COLUMN,
        COUNTRY_COLUMN,
        REGION_COLUMN,
        KOPPEN_CODE_COLUMN,
        KOPPEN_CLASS_COLUMN,
        KOPPEN_MACRO_GROUP_COLUMN
    }
    missing = required_metadata - set(metadata.columns)
    if missing:
        raise ValueError(
            f"City metadata is missing required columns: {sorted(missing)}"
        )
    if CITY_ID_COLUMN not in panel.columns:
        raise ValueError(f"Analysis panel is missing {CITY_ID_COLUMN}.")
    if metadata[CITY_ID_COLUMN].isna().any():
        raise ValueError("City metadata contains null city IDs.")
    if metadata[CITY_ID_COLUMN].duplicated().any():
        raise ValueError("City metadata contains duplicate city IDs.")
    for column, lower, upper in (
        (CENTROID_LON_COLUMN, -180, 180),
        (CENTROID_LAT_COLUMN, -90, 90)
    ):
        values = pd.to_numeric(metadata[column], errors="raise")
        if (
            values.isna().any()
            or not np.isfinite(values.to_numpy(dtype="float64")).all()
            or not values.between(lower, upper).all()
        ):
            raise ValueError(
                f"City metadata {column} must be finite and in "
                f"[{lower}, {upper}]."
            )
    area_km2 = pd.to_numeric(metadata[CITY_AREA_KM2_COLUMN], errors="raise")
    if (
        area_km2.isna().any()
        or not np.isfinite(area_km2.to_numpy(dtype="float64")).all()
        or (area_km2 <= 0).any()
    ):
        raise ValueError(
            f"City metadata {CITY_AREA_KM2_COLUMN} must contain finite "
            "positive values."
        )
    for column in (COUNTRY_COLUMN, REGION_COLUMN):
        labels = metadata[column].astype("string").str.strip()
        if labels.isna().any() or labels.eq("").any():
            raise ValueError(
                f"City metadata {column} contains missing values."
            )

    codes = pd.to_numeric(metadata[KOPPEN_CODE_COLUMN], errors="raise")
    invalid_codes = codes.notna() & ~codes.isin(CLIMATE_CLASS_BY_CODE)
    if invalid_codes.any():
        raise ValueError(
            "City metadata contains unsupported Köppen-Geiger class codes: "
            f"{codes.loc[invalid_codes].unique().tolist()}"
        )
    missing_code = codes.isna()
    missing_class = metadata[KOPPEN_CLASS_COLUMN].isna()
    missing_macro_group = metadata[KOPPEN_MACRO_GROUP_COLUMN].isna()
    if (
        not missing_code.equals(missing_class)
        or not missing_code.equals(missing_macro_group)
    ):
        raise ValueError(
            "Köppen-Geiger code, class label, and macro group must be null "
            "together."
        )
    expected_classes = codes.map(CLIMATE_CLASS_BY_CODE)
    mismatched_classes = (
        ~missing_code
        & metadata[KOPPEN_CLASS_COLUMN].ne(expected_classes)
    )
    if mismatched_classes.any():
        raise ValueError(
            "Köppen-Geiger class labels do not match their class codes."
        )
    climate_code_to_group = _macro_group_by_code(cz_macro_groups)
    expected_macro_groups = codes.map(climate_code_to_group)
    mismatched_groups = (
        ~missing_code
        & metadata[KOPPEN_MACRO_GROUP_COLUMN].ne(expected_macro_groups)
    )
    if mismatched_groups.any():
        raise ValueError(
            "Köppen-Geiger macro groups do not match their class codes."
        )

    metadata_columns = [
        column for column in metadata.columns if column != CITY_ID_COLUMN
    ]
    panel_for_merge = panel.drop(
        columns=metadata_columns,
        errors="ignore"
    )
    merged = panel_for_merge.merge(
        metadata[[CITY_ID_COLUMN, *metadata_columns]],
        on=CITY_ID_COLUMN,
        how="left",
        validate="many_to_one",
        indicator=True
    )
    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        examples = merged.loc[unmatched, CITY_ID_COLUMN].drop_duplicates()
        raise ValueError(
            "Analysis rows have city IDs missing from the boundary metadata: "
            f"{examples.head(10).tolist()}"
        )
    return merged.drop(columns="_merge")
