import ee

LST_BANDS = [
    "terra_day",
    "terra_night",
    "aqua_day",
    "aqua_night"
]

CLEAR_BANDS = [
    "terra_day_clear_days",
    "terra_night_clear_days",
    "aqua_day_clear_days",
    "aqua_night_clear_days"
]

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

# def reduce_monthly_coverage(
#     monthly_image: ee.Image,
#     regions: ee.FeatureCollection,
#     projection: ee.Projection,
#     native_scale: float
# ) -> ee.FeatureCollection:

#     return (
#         monthly_image
#         .select(CLEAR_BANDS)
#         .reduceRegions(
#             collection=regions,
#             reducer=ee.Reducer.mean(),
#             crs=projection,
#             scale=native_scale,
#             tileScale=4
#         )
#     )