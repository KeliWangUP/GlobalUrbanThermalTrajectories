from src.export_utils import get_modis_reference_grid, initialize_gee, load_config
from src.samplepoint_development_checks import check_samplepoint_development_city


def main() -> None:
    config = load_config()
    initialize_gee(config)
    projection, scale = get_modis_reference_grid(config)
    check_samplepoint_development_city(config, projection, scale)


if __name__ == "__main__":
    main()
