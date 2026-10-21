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
    max_error_m: int,
    reference_projection: ee.Projection
) -> ee.Image:
    """
    Create a global exclusion mask around all study cities.

    1 = excluded
    0 = potentially available
    """

    buffered = cities.map(
        lambda feature: feature.buffer(
            buffer_m,
            max_error_m
        )
    )

    exclusion = (
        ee.Image.constant(0)
        .byte()
        .paint(buffered, 1)
        .rename("urban_exclusion")
        .setDefaultProjection(
            reference_projection
        )
    )

    return exclusion

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

    Criteria
    --------
    - Not impervious according to GAIA
    - Slope below threshold
    - Not water
    - Outside all city exclusion buffers

    Notes
    -----
    The binary image remains unmasked before MODIS aggregation
    so that unsuitable pixels contribute zeros to the rural fraction.
    """

    # --------------------------------------------------
    # GAIA
    # --------------------------------------------------

    gaia = urban_mask_source.select(
        "change_year_index"
    )

    # GAIA transition pixels have values 1-34.
    # Masked/non-transition pixels are converted to zero.
    gaia_filled = gaia.unmask(0)

    stable_nonurban = (
        gaia_filled
        .eq(0)
        .rename("stable_nonurban")
    )

    # --------------------------------------------------
    # DEM
    # --------------------------------------------------

    first_dem = ee.Image(
        dem_mask_source.first()
    )

    dem_projection = (
        first_dem
        .select("DEM")
        .projection()
    )

    dem = (
        dem_mask_source
        .select("DEM")
        .mosaic()
        .setDefaultProjection(
            dem_projection
        )
    )

    slope = ee.Terrain.slope(dem)

    slope_ok = (
        slope
        .lt(max_slope_deg)
        .rename("slope_ok")
    )

    # --------------------------------------------------
    # Water
    # --------------------------------------------------

    water = (
        dem_mask_source
        .select("WBM")
        .mosaic()
        .setDefaultProjection(
            dem_projection
        )
    )

    non_water = (
        water
        .eq(0)
        .rename("non_water")
    )

    # --------------------------------------------------
    # City exclusion
    # --------------------------------------------------

    outside_city_influence = (
        global_exclusion_mask
        .eq(0)
        .rename("outside_city")
    )

    # --------------------------------------------------
    # Final binary mask
    # --------------------------------------------------

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
    minimum_rural_fraction: float,
    max_input_pixels: int
) -> tuple[ee.Image, ee.Image]:
    """
    Convert fine-resolution rural suitability to MODIS-grid purity.

    max_input_pixels must cover the greatest source-pixel count needed
    for any output pixel under the source and MODIS projections.
    """

    rural_fraction = (
        rural_binary
        .toFloat()
        .reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=max_input_pixels
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


def build_rural_fraction_image(
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
    """Build the MODIS-aligned rural fraction image for static export."""

    gaia_projection = (
        urban_mask_source
        .select("change_year_index")
        .projection()
    )
    urban_exclusion = build_global_urban_exclusion_mask(
        cities=cities,
        buffer_m=inner_buffer_m,
        max_error_m=max_error_m,
        reference_projection=gaia_projection
    )
    rural_binary = build_global_stable_rural_binary(
        urban_mask_source=urban_mask_source,
        dem_mask_source=dem_mask_source,
        global_exclusion_mask=urban_exclusion,
        max_slope_deg=max_slope_deg
    )
    rural_fraction, _ = aggregate_rural_mask_to_modis(
        rural_binary=rural_binary,
        modis_projection=modis_projection,
        minimum_rural_fraction=minimum_rural_fraction,
        max_input_pixels=max_input_pixels
    )
    return rural_fraction


def build_rural_halos_with_metrics(
    cities: ee.FeatureCollection,
    rural_fraction: ee.Image,
    city_id_field: str,
    inner_buffer_m: int,
    outer_buffer_m: int,
    max_error_m: int,
    minimum_rural_fraction: float,
    projection: ee.Projection,
    scale: float,
    tile_scale: int
) -> ee.FeatureCollection:
    """Build halo polygons with area and eligible MODIS pixel counts."""

    halos = build_all_rural_halos(
        cities=cities,
        inner_buffer_m=inner_buffer_m,
        outer_buffer_m=outer_buffer_m,
        city_id_field=city_id_field,
        max_error_m=max_error_m
    )

    def _add_area(feature):
        feature = ee.Feature(feature)
        area_km2 = (
            feature.geometry()
            .area(maxError=max_error_m)
            .divide(1_000_000)
        )
        return feature.set("halo_area_km2", area_km2)

    halos = halos.map(_add_area)
    eligible_pixels = (
        rural_fraction
        .gte(minimum_rural_fraction)
        .selfMask()
        .rename("rural_pixel")
    )
    counted_halos = eligible_pixels.reduceRegions(
        collection=halos,
        reducer=ee.Reducer.count(),
        crs=projection,
        scale=scale,
        tileScale=tile_scale
    )

    def _add_pixel_count(feature):
        feature = ee.Feature(feature)
        has_count = feature.propertyNames().contains("rural_pixel")
        pixel_count = ee.Algorithms.If(
            has_count,
            feature.get("rural_pixel"),
            0
        )
        return feature.set(
            "rural_n_modis_pixels",
            pixel_count
        ).select([
            city_id_field,
            "halo_area_km2",
            "rural_n_modis_pixels"
        ])

    return counted_halos.map(_add_pixel_count)