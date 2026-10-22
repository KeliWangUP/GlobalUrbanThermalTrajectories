import ee


def build_rural_halo(
    feature: ee.Feature,
    inner_buffer_m: int,
    outer_buffer_m: int,
    city_id_field: str,
    max_error_m: int
) -> ee.Feature:
    """Build a ring-shaped rural search area around one city."""

    geometry = feature.geometry()
    inner = geometry.buffer(inner_buffer_m, max_error_m)
    outer = geometry.buffer(outer_buffer_m, max_error_m)
    halo = outer.difference(inner, max_error_m)

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
    """Build one rural halo per city."""

    return cities.map(
        lambda feature: build_rural_halo(
            feature,
            inner_buffer_m=inner_buffer_m,
            outer_buffer_m=outer_buffer_m,
            city_id_field=city_id_field,
            max_error_m=max_error_m
        )
    )


def build_global_urban_exclusion_mask(
    cities: ee.FeatureCollection,
    buffer_m: int,
    max_error_m: int,
    reference_projection: ee.Projection
) -> ee.Image:
    """
    Create an exclusion mask around all study cities.

    1 = excluded
    0 = potentially available

    Pass every city, not only the cities of one export batch, so rural
    pixels near other cities stay excluded.
    """

    buffered = cities.map(
        lambda feature: feature.buffer(buffer_m, max_error_m)
    )

    return (
        ee.Image.constant(0)
        .byte()
        .paint(buffered, 1)
        .rename("urban_exclusion")
        .setDefaultProjection(reference_projection)
    )


def build_global_stable_rural_binary(
    urban_mask_source: ee.Image,
    dem_mask_source: ee.ImageCollection,
    global_exclusion_mask: ee.Image,
    max_slope_deg: float
) -> ee.Image:
    """
    Create a fine-resolution binary rural eligibility image.

    1 = valid rural land
    0 = unsuitable land

    Criteria: not impervious in GAIA, slope below the threshold, not
    water, and outside all city exclusion buffers. The image is unmasked
    so that unsuitable pixels contribute zeros to the MODIS-pixel fraction.
    """

    # GAIA transition pixels have values 1-34; masked pixels never changed.
    stable_nonurban = (
        urban_mask_source
        .select("change_year_index")
        .unmask(0)
        .eq(0)
    )

    dem_projection = (
        ee.Image(dem_mask_source.first())
        .select("DEM")
        .projection()
    )

    dem = (
        dem_mask_source
        .select("DEM")
        .mosaic()
        .setDefaultProjection(dem_projection)
    )
    slope_ok = ee.Terrain.slope(dem).lt(max_slope_deg)

    non_water = (
        dem_mask_source
        .select("WBM")
        .mosaic()
        .setDefaultProjection(dem_projection)
        .eq(0)
    )

    outside_city_influence = global_exclusion_mask.eq(0)

    return (
        stable_nonurban
        .And(slope_ok)
        .And(non_water)
        .And(outside_city_influence)
        .unmask(0)
        .rename("rural_binary")
        .byte()
    )


def aggregate_rural_mask_to_modis(
    rural_binary: ee.Image,
    modis_projection: ee.Projection,
    minimum_rural_fraction: float,
    max_input_pixels: int
) -> ee.Image:
    """
    Convert fine-resolution rural suitability to a MODIS-grid mask.

    A MODIS pixel is kept when its rural fraction reaches the threshold.
    max_input_pixels must cover the largest source-pixel count that any
    MODIS pixel needs under the source and MODIS projections.
    """

    rural_fraction = (
        rural_binary
        .toFloat()
        .reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=max_input_pixels
        )
        .reproject(crs=modis_projection)
    )

    return (
        rural_fraction
        .gte(minimum_rural_fraction)
        .rename("rural_modis_mask")
    )


def build_rural_modis_mask(
    cities: ee.FeatureCollection,
    urban_mask_source: ee.Image,
    dem_mask_source: ee.ImageCollection,
    modis_projection: ee.Projection,
    inner_buffer_m: int,
    max_error_m: int,
    max_slope_deg: float,
    minimum_rural_fraction: float,
    max_input_pixels: int
) -> ee.Image:
    """Build the MODIS-grid rural mask used to mask LST before reduction."""

    urban_exclusion = build_global_urban_exclusion_mask(
        cities=cities,
        buffer_m=inner_buffer_m,
        max_error_m=max_error_m,
        reference_projection=(
            urban_mask_source
            .select("change_year_index")
            .projection()
        )
    )
    rural_binary = build_global_stable_rural_binary(
        urban_mask_source=urban_mask_source,
        dem_mask_source=dem_mask_source,
        global_exclusion_mask=urban_exclusion,
        max_slope_deg=max_slope_deg
    )
    return aggregate_rural_mask_to_modis(
        rural_binary=rural_binary,
        modis_projection=modis_projection,
        minimum_rural_fraction=minimum_rural_fraction,
        max_input_pixels=max_input_pixels
    )
