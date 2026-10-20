def run_annual_exports(
    dataset_name: str,
    regions: ee.FeatureCollection,
    config: dict,
    projection: ee.Projection,
    scale: ee.Number,
    spatial_mask: ee.Image | None = None
):
    """
    Submit annual GEE export tasks.
    """

    for year in range(
        config["study"]["start_year"],
        config["study"]["end_year"] + 1
    ):

        print(
            f"[INFO] Building "
            f"{dataset_name}: {year}"
        )

        annual_fc = (
            build_yearly_lst_statistics(
                year=year,
                regions=regions,
                config=config,
                projection=projection,
                scale=scale,
                spatial_mask=spatial_mask
            )
        )

        export_table_to_drive(
            collection=annual_fc,
            description=f"{dataset_name}_{year}",
            folder=config["export"]["folder"],
            file_name=f"{dataset_name}_{year}"
        )