from src.export_utils import get_modis_reference_grid, initialize_gee, load_config
from src.samplepoint_development_exports import export_samplepoint_development


def main() -> None:
    config = load_config()
    initialize_gee(config)
    projection, scale = get_modis_reference_grid(config)
    export_samplepoint_development(config, projection, scale)


if __name__ == "__main__":
    main()
