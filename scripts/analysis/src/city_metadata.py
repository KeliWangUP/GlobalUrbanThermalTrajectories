from collections.abc import Mapping
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


CITY_ID_COLUMN = "ID_UC_G0"
CENTROID_LAT_COLUMN = "centroid_lat"
KOPPEN_CODE_COLUMN = "koppen_geiger_code"
KOPPEN_CLASS_COLUMN = "koppen_geiger_class"
KOPPEN_MACRO_GROUP_COLUMN = "koppen_geiger_macro_group"
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


def build_city_metadata(
    shapefile_path: Path,
    climate_raster_path: Path,
    cz_macro_groups: Mapping[str, tuple[int, ...]],
    id_column: str = CITY_ID_COLUMN
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
        CENTROID_LAT_COLUMN: geographic_centroids.y.to_numpy(),
        KOPPEN_CODE_COLUMN: climate_codes,
        KOPPEN_CLASS_COLUMN: climate_classes,
        KOPPEN_MACRO_GROUP_COLUMN: climate_macro_groups
    })

    if metadata[CENTROID_LAT_COLUMN].isna().any():
        raise ValueError(
            f"Centroid latitude calculation produced null values for "
            f"{shapefile_path}."
        )
    if not metadata[CENTROID_LAT_COLUMN].between(-90, 90).all():
        raise ValueError(
            f"Centroid latitude is outside [-90, 90] for {shapefile_path}."
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
        CENTROID_LAT_COLUMN,
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
