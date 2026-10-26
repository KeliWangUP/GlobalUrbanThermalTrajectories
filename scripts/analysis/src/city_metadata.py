from pathlib import Path

import geopandas as gpd
import pandas as pd


CITY_ID_COLUMN = "ID_UC_G0"
CENTROID_LAT_COLUMN = "centroid_lat"


def build_city_metadata(
    shapefile_path: Path,
    id_column: str = CITY_ID_COLUMN
) -> tuple[pd.DataFrame, int]:
    """Read city polygons and calculate WGS84 latitude of each projected centroid."""
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
    metadata = pd.DataFrame({
        CITY_ID_COLUMN: cities[id_column].to_numpy(),
        CENTROID_LAT_COLUMN: geographic_centroids.y.to_numpy()
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
    metadata: pd.DataFrame
) -> pd.DataFrame:
    """Attach one city centroid latitude to each monthly or aggregate row."""
    required_metadata = {CITY_ID_COLUMN, CENTROID_LAT_COLUMN}
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

    panel_for_merge = panel.drop(
        columns=[CENTROID_LAT_COLUMN],
        errors="ignore"
    )
    merged = panel_for_merge.merge(
        metadata[[CITY_ID_COLUMN, CENTROID_LAT_COLUMN]],
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
