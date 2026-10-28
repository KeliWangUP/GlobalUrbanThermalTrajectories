from dataclasses import dataclass
from typing import Any

import ee


@dataclass(frozen=True)
class SamplePointSources:
    isa_image: ee.Image
    isa_projection: ee.Projection
    isa_scale: float
    isa_band: str
    isa_years: tuple[int, ...]
    ghsl_image: ee.Image
    ghsl_projection: ee.Projection
    ghsl_fields: list[str]
    ghsl_scale: float
    water_image: ee.Image
    water_projection: ee.Projection
    water_scale: float


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
    """Load raster sources and resolve their native projections once."""
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
    isa_projection = isa_image.projection()
    isa_scale = isa_projection.nominalScale().getInfo()

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
    ghsl_fields = []
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
        ghsl_fields.extend([
            f"{surface_field}_mean",
            f"{surface_field}_count",
            f"{volume_field}_mean",
            f"{volume_field}_count"
        ])
    ghsl_image = ee.Image.cat(ghsl_bands)
    ghsl_projection = ghsl_image.select(0).projection()
    ghsl_scale = ghsl_config["scale_m"]
    if (
        isinstance(ghsl_scale, bool)
        or not isinstance(ghsl_scale, (int, float))
        or ghsl_scale <= 0
    ):
        raise ValueError("ghsl_built.scale_m must be a positive number.")

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
    water_projection = water_image.projection()
    water_scale = water_projection.nominalScale().getInfo()

    return SamplePointSources(
        isa_image=isa_image,
        isa_projection=isa_projection,
        isa_scale=isa_scale,
        isa_band=isa_band,
        isa_years=tuple(isa_years),
        ghsl_image=ghsl_image,
        ghsl_projection=ghsl_projection,
        ghsl_fields=ghsl_fields,
        ghsl_scale=ghsl_scale,
        water_image=water_image,
        water_projection=water_projection,
        water_scale=water_scale
    )


def _build_cell_regions(
    points: ee.FeatureCollection,
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float
) -> ee.FeatureCollection:
    base_field_count = (
        6 if config["sample_points"]["include_lonlat"] else 4
    )
    base_fields = build_samplepoint_selectors(config)[:base_field_count]
    max_error = config["samplepoint_attributes"]["geometry_error_m"]
    if (
        isinstance(max_error, bool)
        or not isinstance(max_error, (int, float))
        or max_error <= 0
    ):
        raise ValueError(
            "samplepoint_attributes.geometry_error_m must be positive."
        )
    max_error_margin = ee.ErrorMargin(max_error, "meters")

    def _to_cell(feature: ee.Feature) -> ee.Feature:
        coordinates = ee.List(
            feature.geometry()
            .transform(projection, maxError=max_error_margin)
            .coordinates()
        )
        center_x = ee.Number(coordinates.get(0))
        center_y = ee.Number(coordinates.get(1))
        half_scale = ee.Number(scale).divide(2)
        geometry = ee.Geometry.Rectangle(
            [
                center_x.subtract(half_scale),
                center_y.subtract(half_scale),
                center_x.add(half_scale),
                center_y.add(half_scale)
            ],
            proj=projection,
            geodesic=False
        )
        return ee.Feature(
            geometry,
            feature.toDictionary(base_fields)
        )

    return points.map(_to_cell)


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

    cells = _build_cell_regions(
        points=points,
        config=config,
        projection=projection,
        scale=modis_scale
    )
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
    isa_fields = [
        f"isa_urban_fraction_{year}"
        for year in sources.isa_years
    ]
    isa_stats = isa_fraction_image.reduceRegions(
        collection=cells,
        reducer=ee.Reducer.mean().unweighted(),
        crs=sources.isa_projection,
        scale=sources.isa_scale,
        tileScale=tile_scale,
        maxPixelsPerRegion=max_pixels
    )

    pixel_area = ee.Image.pixelArea()
    water_area_image = ee.Image.cat([
        pixel_area.updateMask(sources.water_image).rename("water_area_m2"),
        pixel_area.updateMask(
            sources.water_image.mask()
        ).rename("water_valid_area_m2")
    ])
    water_stats = water_area_image.reduceRegions(
        collection=cells,
        reducer=ee.Reducer.sum(),
        crs=sources.water_projection,
        scale=sources.water_scale,
        tileScale=tile_scale,
        maxPixelsPerRegion=max_pixels
    )
    ghsl_reducer = (
        ee.Reducer.mean()
        .combine(
            reducer2=ee.Reducer.count(),
            sharedInputs=True
        )
    )
    ghsl_stats = sources.ghsl_image.reduceRegions(
        collection=cells,
        reducer=ghsl_reducer,
        crs=sources.ghsl_projection,
        scale=sources.ghsl_scale,
        tileScale=tile_scale,
        maxPixelsPerRegion=max_pixels
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
    selectors = build_samplepoint_selectors(config)
    base_field_count = 6 if config["sample_points"]["include_lonlat"] else 4
    base_fields = selectors[:base_field_count]

    def _format_feature(feature: ee.Feature) -> ee.Feature:
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
        properties = (
            ee.Dictionary(feature.toDictionary(base_fields))
            .combine(
                ee.Dictionary(feature.toDictionary(isa_fields)),
                overwrite=True
            )
            .combine(water_properties, overwrite=True)
            .combine(
                ee.Dictionary(ghsl_feature.toDictionary(sources.ghsl_fields)),
                overwrite=True
            )
        )
        return ee.Feature(None, properties)

    return joined.map(_format_feature)
