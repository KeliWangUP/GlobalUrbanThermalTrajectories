# src/export_utils.py

import ee


LST_BANDS = [
    "terra_day",
    "terra_night",
    "aqua_day",
    "aqua_night"
]


def build_spatial_reducer() -> ee.Reducer:
    """
    Build standard spatial LST reducer.
    """

    return (
        ee.Reducer.mean()
        .combine(
            reducer2=ee.Reducer.median(),
            sharedInputs=True
        )
        .combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True
        )
        .combine(
            reducer2=ee.Reducer.count(),
            sharedInputs=True
        )
    )


def reduce_monthly_lst(
    image: ee.Image,
    regions: ee.FeatureCollection,
    year: int,
    month: int,
    projection: ee.Projection,
    scale: ee.Number
) -> ee.FeatureCollection:
    """
    Reduce monthly MODIS LST to city/rural regions.
    """

    result = (
        image
        .select(LST_BANDS)
        .reduceRegions(
            collection=regions,
            reducer=build_spatial_reducer(),
            crs=projection,
            scale=scale,
            tileScale=4
        )
    )

    return result.map(
        lambda f: (
            f.set({
                "year": year,
                "month": month
            })
            .setGeometry(None)
        )
    )


def export_table_to_drive(
    collection: ee.FeatureCollection,
    description: str,
    folder: str,
    file_name: str
):
    """
    Submit an Earth Engine CSV export task.
    """

    task = ee.batch.Export.table.toDrive(
        collection=collection,
        description=description,
        folder=folder,
        fileNamePrefix=file_name,
        fileFormat="CSV"
    )

    task.start()

    print(f"[EXPORT] {description}")
    print(f"[TASK] {task.id}")

    return task

def build_yearly_lst_statistics(
    year: int,
    regions: ee.FeatureCollection,
    config: dict,
    projection: ee.Projection,
    scale: ee.Number,
    spatial_mask: ee.Image | None = None
) -> ee.FeatureCollection:
    """
    Build 12 months of regional MODIS LST statistics.
    """

    from src.modis_lst_utils import (
        build_monthly_four_slot_image
    )

    monthly_results = []

    for month in range(1, 13):

        monthly_image = (
            build_monthly_four_slot_image(
                year=year,
                month=month,
                config=config,
                reference_projection=projection
            )
        )

        if spatial_mask is not None:
            monthly_image = (
                monthly_image
                .updateMask(spatial_mask)
            )

        monthly_stats = reduce_monthly_lst(
            image=monthly_image,
            regions=regions,
            year=year,
            month=month,
            projection=projection,
            scale=scale
        )

        monthly_results.append(
            monthly_stats
        )

    return ee.FeatureCollection(
        monthly_results
    ).flatten()