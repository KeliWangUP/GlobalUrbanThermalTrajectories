def main():
    run_annual_exports(
    dataset_name="city_lst_fixed2024",
    regions=cities,
    config=config,
    projection=projection,
    scale=scale
    )


if __name__ == "__main__":
    main()