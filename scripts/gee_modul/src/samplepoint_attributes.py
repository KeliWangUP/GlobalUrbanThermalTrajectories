from dataclasses import dataclass
from typing import Any

import ee


@dataclass(frozen=True)
class SamplePointSources:
    isa_image: ee.Image
    isa_band: str
    isa_years: tuple[int, ...]
    ghsl_image: ee.Image
    water_image: ee.Image


def build_samplepoint_selectors(
    config: dict[str, Any]
) -> list[str]:
    id_field = config["cities"]["id_field"]
    selectors = [
        "pixel_id",
        id_field,
        "modis_x",
        "modis_y"
    ]
    if config["sample_points"]["include_lonlat"]:
        selectors.extend(["lon", "lat"])
    selectors.extend(
        f"isa_urban_fraction_{year}"
        for year in config["samplepoint_attributes"]["isa_years"]
    )
    selectors.extend([
        "water_area_m2",
        "water_valid_area_m2",
        "water_fraction"
    ])
    for year in config["ghsl_built"]["years"]:
        selectors.extend([
            f"built_surface_{year}_m2_mean",
            f"built_surface_{year}_m2_count",
            f"built_volume_{year}_m3_mean",
            f"built_volume_{year}_m3_count"
        ])
    if len(selectors) != len(set(selectors)):
        raise ValueError("Sample-point export selectors contain duplicates.")
    return selectors


def _resolve_isa_band(
    isa_image: ee.Image,
    configured_band: str | None
) -> str:
    band_names = isa_image.bandNames().getInfo()
    if configured_band is not None:
        if configured_band not in band_names:
            raise ValueError(
                f"Configured ISA band '{configured_band}' was not found. "
                f"Available bands: {band_names}"
            )
        return configured_band
    if len(band_names) != 1:
        raise ValueError(
            "samplepoint_attributes.isa_band must be configured when the "
            f"ISA asset has multiple bands: {band_names}"
        )
    return band_names[0]


def prepare_samplepoint_sources(
    config: dict[str, Any]
) -> SamplePointSources:
    """Load the selected ISA, GHSL and water raster sources."""
    attribute_config = config["samplepoint_attributes"]
    isa_asset = attribute_config["isa_asset"]
    if not isinstance(isa_asset, str) or not isa_asset.strip():
        raise ValueError(
            "samplepoint_attributes.isa_asset must be a non-empty asset ID."
        )

    isa_image = ee.Image(isa_asset)
    isa_years = attribute_config["isa_years"]
    if (
        not isinstance(isa_years, list)
        or any(type(year) is not int for year in isa_years)
        or isa_years != list(range(1985, 2022))
    ):
        raise ValueError(
            "samplepoint_attributes.isa_years must list every year from "
            "1985 through 2021 in order."
        )
    isa_band = _resolve_isa_band(
        isa_image,
        attribute_config.get("isa_band")
    )
    isa_image = isa_image.select(isa_band)

    ghsl_config = config["ghsl_built"]
    years = ghsl_config["years"]
    if (
        not years
        or any(type(year) is not int for year in years)
        or years != sorted(set(years))
    ):
        raise ValueError(
            "ghsl_built.years must be a non-empty, increasing list of "
            "unique integer years."
        )

    ghsl_bands = []
    for year in years:
        surface_field = f"built_surface_{year}_m2"
        volume_field = f"built_volume_{year}_m3"
        ghsl_bands.extend([
            ee.Image(
                f"{ghsl_config['built_surface_collection']}/{year}"
            ).select(
                ghsl_config["built_surface_band"]
            ).rename(surface_field),
            ee.Image(
                f"{ghsl_config['built_volume_collection']}/{year}"
            ).select(
                ghsl_config["built_volume_band"]
            ).rename(volume_field)
        ])
    ghsl_image = ee.Image.cat(ghsl_bands)

    rural_config = config["rural"]
    water_band = attribute_config["water_band"]
    dem_collection = ee.ImageCollection(rural_config["dem_collection"])
    dem_projection = (
        ee.Image(dem_collection.first())
        .select("DEM")
        .projection()
    )
    water_image = (
        dem_collection
        .select(water_band)
        .mosaic()
        .setDefaultProjection(dem_projection)
        .neq(0)
        .rename("water")
    )
    return SamplePointSources(
        isa_image=isa_image,
        isa_band=isa_band,
        isa_years=tuple(isa_years),
        ghsl_image=ghsl_image,
        water_image=water_image
    )


def _sample_modis_grid_image(
    image: ee.Image,
    reducer: ee.Reducer,
    points: ee.FeatureCollection,
    properties: list[str],
    projection: ee.Projection,
    scale: float,
    max_pixels: int,
    tile_scale: int,
    fill_value: int | float
) -> ee.FeatureCollection:
    """Aggregate a raster to MODIS pixels, then sample at their centers."""
    modis_image = (
        image
        .reduceResolution(
            reducer=reducer,
            maxPixels=max_pixels
        )
        .unmask(fill_value)
    )
    return modis_image.sampleRegions(
        collection=points,
        properties=properties,
        projection=projection,
        scale=scale,
        tileScale=tile_scale,
        geometries=False
    )


def _join_by_point_and_city(
    primary: ee.FeatureCollection,
    secondary: ee.FeatureCollection,
    id_field: str,
    match_property: str
) -> ee.FeatureCollection:
    return ee.FeatureCollection(
        ee.Join.saveFirst(match_property).apply(
            primary=primary,
            secondary=secondary,
            condition=ee.Filter.And(
                ee.Filter.equals(
                    leftField="pixel_id",
                    rightField="pixel_id"
                ),
                ee.Filter.equals(
                    leftField=id_field,
                    rightField=id_field
                )
            )
        )
    )


def build_samplepoint_attribute_collection(
    points: ee.FeatureCollection,
    config: dict[str, Any],
    projection: ee.Projection,
    modis_scale: float,
    sources: SamplePointSources
) -> ee.FeatureCollection:
    """Summarize ISA, GHSL and water within each MODIS point cell."""
    id_field = config["cities"]["id_field"]
    tile_scale = config["processing"]["tile_scale"]
    max_pixels = config["samplepoint_attributes"]["max_pixels_per_cell"]
    if (
        isinstance(max_pixels, bool)
        or not isinstance(max_pixels, int)
        or max_pixels < 1
    ):
        raise ValueError(
            "samplepoint_attributes.max_pixels_per_cell must be a "
            "positive integer."
        )

    base_field_count = 6 if config["sample_points"]["include_lonlat"] else 4
    base_fields = build_samplepoint_selectors(config)[:base_field_count]
    isa_fields = [
        f"isa_urban_fraction_{year}"
        for year in sources.isa_years
    ]
    isa_valid = (
        sources.isa_image.eq(0)
        .Or(
            sources.isa_image.gte(2)
            .And(sources.isa_image.lte(38))
        )
    )
    isa_fraction_bands = [
        sources.isa_image.gte(2023 - year)
        .updateMask(isa_valid)
        .rename(f"isa_urban_fraction_{year}")
        for year in sources.isa_years
    ]
    isa_fraction_image = ee.Image.cat(isa_fraction_bands)
    isa_stats = _sample_modis_grid_image(
        image=isa_fraction_image,
        reducer=ee.Reducer.mean().unweighted(),
        points=points,
        properties=base_fields,
        projection=projection,
        scale=modis_scale,
        max_pixels=max_pixels,
        tile_scale=tile_scale,
        fill_value=-1
    )

    pixel_area = ee.Image.pixelArea().setDefaultProjection(
        sources.water_image.projection()
    )
    water_area_image = ee.Image.cat([
        pixel_area.updateMask(sources.water_image)
        .unmask(0)
        .rename("water_area_m2"),
        pixel_area.updateMask(
            sources.water_image.mask()
        ).unmask(0)
        .rename("water_valid_area_m2")
    ])
    water_stats = _sample_modis_grid_image(
        image=water_area_image,
        reducer=ee.Reducer.sum(),
        points=points,
        properties=base_fields,
        projection=projection,
        scale=modis_scale,
        max_pixels=max_pixels,
        tile_scale=tile_scale,
        fill_value=0
    )
    ghsl_reducer = (
        ee.Reducer.mean()
        .combine(
            reducer2=ee.Reducer.count(),
            sharedInputs=True
        )
    )
    ghsl_stats = _sample_modis_grid_image(
        image=sources.ghsl_image,
        reducer=ghsl_reducer,
        points=points,
        properties=base_fields,
        projection=projection,
        scale=modis_scale,
        max_pixels=max_pixels,
        tile_scale=tile_scale,
        fill_value=-1
    )

    isa_water = _join_by_point_and_city(
        primary=isa_stats,
        secondary=water_stats,
        id_field=id_field,
        match_property="_water_stats"
    )
    joined = _join_by_point_and_city(
        primary=isa_water,
        secondary=ghsl_stats,
        id_field=id_field,
        match_property="_ghsl_stats"
    )

    def _format_feature(feature: ee.Feature) -> ee.Feature:
        isa_properties = ee.Dictionary({
            field: ee.Algorithms.If(
                ee.Number(feature.get(field)).gte(0),
                feature.get(field),
                None
            )
            for field in isa_fields
        })

        water_feature = ee.Feature(feature.get("_water_stats"))
        water_stats_dictionary = ee.Dictionary(
            water_feature.toDictionary()
        )
        water_area = ee.Number(
            water_stats_dictionary.get("water_area_m2", 0)
        )
        water_valid_area = ee.Number(
            water_stats_dictionary.get("water_valid_area_m2", 0)
        )
        water_fraction = ee.Algorithms.If(
            water_valid_area.gt(0),
            water_area.divide(water_valid_area),
            None
        )
        water_properties = ee.Dictionary({
            "water_area_m2": water_area,
            "water_valid_area_m2": water_valid_area,
            "water_fraction": water_fraction
        })
        ghsl_feature = ee.Feature(feature.get("_ghsl_stats"))
        ghsl_properties: dict[str, Any] = {}
        for year in config["ghsl_built"]["years"]:
            for measure in ("surface", "volume"):
                name = (
                    f"built_{measure}_{year}_m2"
                    if measure == "surface"
                    else f"built_{measure}_{year}_m3"
                )
                mean_field = f"{name}_mean"
                count_field = f"{name}_count"
                count = ee.Number(ghsl_feature.get(count_field))
                has_pixels = count.gt(0)
                ghsl_properties[mean_field] = ee.Algorithms.If(
                    has_pixels,
                    ghsl_feature.get(mean_field),
                    None
                )
                ghsl_properties[count_field] = ee.Algorithms.If(
                    has_pixels,
                    count,
                    0
                )
        properties = (
            ee.Dictionary(feature.toDictionary(base_fields))
            .combine(isa_properties, overwrite=True)
            .combine(water_properties, overwrite=True)
            .combine(ee.Dictionary(ghsl_properties), overwrite=True)
        )
        return ee.Feature(None, properties)

    return joined.map(_format_feature)
