def main():
    run_annual_exports(
    dataset_name="rural_lst",
    regions=rural_halos,
    config=config,
    projection=projection,
    scale=scale,
    spatial_mask=rural_modis_mask
    )

if __name__ == "__main__":
    main()