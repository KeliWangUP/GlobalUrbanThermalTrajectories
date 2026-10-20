import ee

def build_rural_halo(
    feature: ee.Feature,
    inner_buffer_m: int,
    outer_buffer_m: int,
    city_id_field: str,
    max_error_m: int
) -> ee.Feature:
    """
    Build a ring-shaped rural search area around one city.
    """

    geometry = feature.geometry()

    inner = geometry.buffer(
        inner_buffer_m,
        max_error_m
    )

    outer = geometry.buffer(
        outer_buffer_m,
        max_error_m
    )

    halo = outer.difference(
        inner,
        max_error_m
    )

    return (
        ee.Feature(halo)
        .copyProperties(feature, [city_id_field])
    )

def build_all_rural_halos(
    cities: ee.FeatureCollection,
    inner_buffer_m: int,
    outer_buffer_m: int,
    city_id_field: str,
    max_error_m: int
) -> ee.FeatureCollection:

    return cities.map(
        lambda f: build_rural_halo(
            f,
            inner_buffer_m=inner_buffer_m,
            outer_buffer_m=outer_buffer_m,
            city_id_field=city_id_field,
            max_error_m=max_error_m
        )
    )

def build_global_urban_exclusion_mask(
    cities: ee.FeatureCollection,
    buffer_m: int,
    max_error_m: int
) -> ee.Image:
    """
    Create a global urban exclusion zone.

    Pixels inside any city or within buffer_m of any city
    are marked as excluded.
    """

    buffered = cities.map(
        lambda f: f.buffer(
            buffer_m,
            max_error_m
        )
    )

    exclusion = (
        ee.Image.constant(0)
        .byte()
        .paint(buffered, 1)
        .rename("urban_exclusion")
    )

    return exclusion

def build_global_stable_rural_binary(
    urban_mask_source: ee.Image,
    dem_mask_source: ee.ImageCollection,
    global_exclusion_mask: ee.Image,
    max_slope_deg: float,
    source_scale_m: int
) -> ee.Image:
    """
    Create an unmasked binary rural eligibility image.

    1 = valid stable rural land
    0 = unsuitable land

    Important:
    The zero values must be retained before aggregation to MODIS scale.
    """

    # DEM
    dem = (
        dem_mask_source
        .select("DEM")
        .mosaic()
    )

    dem = dem.setDefaultProjection(
        crs="EPSG:4326",
        scale=source_scale_m
    )

    slope = ee.Terrain.slope(dem)

    slope_ok = slope.lt(max_slope_deg)

    # Water mask
    water = (
        dem_mask_source
        .select("WBM")
        .mosaic()
        .setDefaultProjection(
            crs="EPSG:4326",
            scale=source_scale_m
        )
    )

    non_water = water.eq(0)

    # GAIA stable non-urban
    gaia = urban_mask_source.select(
        "change_year_index"
    )

    gaia_filled = (
        gaia
        .updateMask(gaia)
        .unmask(0)
    )

    stable_nonurban = gaia_filled.eq(0)

    # Outside all city + 2 km exclusion zones
    outside_city_influence = (
        global_exclusion_mask.eq(0)
    )

    rural_binary = (
        stable_nonurban
        .And(slope_ok)
        .And(non_water)
        .And(outside_city_influence)
        .unmask(0)
        .rename("rural_binary")
        .byte()
    )

    return rural_binary

def aggregate_rural_mask_to_modis(
    rural_binary: ee.Image,
    modis_projection: ee.Projection,
    minimum_rural_fraction: float
) -> tuple[ee.Image, ee.Image]:
    """
    Convert fine-resolution rural suitability to MODIS-grid purity.
    """

    rural_fraction = (
        rural_binary
        .toFloat()
        .reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=4096
        )
        .reproject(
            crs=modis_projection
        )
        .rename("rural_fraction")
    )

    rural_mask = (
        rural_fraction
        .gte(minimum_rural_fraction)
        .rename("rural_modis_mask")
    )

    return rural_fraction, rural_mask
