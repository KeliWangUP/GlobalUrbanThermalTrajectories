from typing import Any

import ee

from src.export_utils import run_annual_exports


def load_rural_inputs(
    config: dict[str, Any],
    city_id: int | None = None
) -> tuple[ee.FeatureCollection, ee.Image]:
    """Load the persistent rural assets used by temporal MODIS processing."""
    rural_config = config["rural"]
    static_assets = rural_config["static_assets"]
    id_field = config["cities"]["id_field"]

    fraction = ee.Image(
        static_assets["fraction_asset"]
    )
    band_names = fraction.bandNames().getInfo()
    if "rural_fraction" not in band_names:
        raise ValueError(
            "Rural fraction asset is missing the 'rural_fraction' band: "
            f"{static_assets['fraction_asset']}"
        )

    rural_halos = ee.FeatureCollection(
        static_assets["halos_asset"]
    )
    if city_id is not None:
        rural_halos = rural_halos.filter(
            ee.Filter.eq(id_field, city_id)
        )
    if rural_halos.size().getInfo() == 0:
        if city_id is None:
            message = (
                "Rural halo asset contains no features: "
                f"{static_assets['halos_asset']}"
            )
        else:
            message = (
                "Rural halo asset contains no feature for the requested "
                f"city ID {city_id}."
            )
        raise ValueError(message)

    rural_modis_mask = (
        fraction
        .select("rural_fraction")
        .gte(
            rural_config["minimum_rural_fraction_per_modis_pixel"]
        )
        .rename("rural_modis_mask")
    )
    return rural_halos, rural_modis_mask


def export_rural(
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float
) -> None:
    """Export annual LST statistics for rural halo regions."""
    rural_halos, rural_modis_mask = load_rural_inputs(config)
    run_annual_exports(
        dataset_name=config["export"]["rural_dataset_name"],
        regions=rural_halos,
        config=config,
        projection=projection,
        scale=scale,
        spatial_mask=rural_modis_mask
    )
