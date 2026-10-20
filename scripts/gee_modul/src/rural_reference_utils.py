import ee

LST_BANDS = [
    "terra_day",
    "terra_night",
    "aqua_day",
    "aqua_night"
]

def build_rural_halo(
    feature: ee.Feature,
    inner_buffer_m: int,
    outer_buffer_m: int,
    max_error_m: int = 100
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
        .copyProperties(feature, ["ID_UC_G0"])
    )

def build_all_rural_halos(
    cities: ee.FeatureCollection,
    inner_buffer_m: int,
    outer_buffer_m: int
) -> ee.FeatureCollection:

    return cities.map(
        lambda f: build_rural_halo(
            f,
            inner_buffer_m=inner_buffer_m,
            outer_buffer_m=outer_buffer_m
        )
    )

def build_global_urban_exclusion_mask(
    cities: ee.FeatureCollection,
    buffer_m: int = 2000,
    max_error_m: int = 100
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
    max_slope_deg: float = 5
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
        scale=30
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
            scale=30
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
    minimum_rural_fraction: float = 0.8
):
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

def build_spatial_reducer() -> ee.Reducer:
    """
    Build spatial statistics reducer.
    """

    return (
        ee.Reducer.mean()
        .combine(
            ee.Reducer.median(),
            sharedInputs=True
        )
        .combine(
            ee.Reducer.stdDev(),
            sharedInputs=True
        )
        .combine(
            ee.Reducer.count(),
            sharedInputs=True
        )
    )

def reduce_monthly_lst(
    monthly_image: ee.Image,
    regions: ee.FeatureCollection,
    year: int,
    month: int,
    projection: ee.Projection,
    native_scale: float
) -> ee.FeatureCollection:
    """
    Reduce monthly four-slot LST to regions.
    """

    lst_image = monthly_image.select(
        LST_BANDS
    )

    stats = lst_image.reduceRegions(
        collection=regions,
        reducer=build_spatial_reducer(),
        crs=projection,
        scale=native_scale,
        tileScale=4
    )

    def _add_metadata(feature):

        return (
            feature
            .set({
                "year": year,
                "month": month
            })
            .setGeometry(None)
        )

    return stats.map(_add_metadata)

