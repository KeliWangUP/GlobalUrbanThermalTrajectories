from src.export_utils import initialize_gee, load_config
from src.era5_land_exports import run_era5_land_export


def main() -> None:
    config = load_config()
    initialize_gee(config)
    run_era5_land_export(config)


if __name__ == "__main__":
    main()
