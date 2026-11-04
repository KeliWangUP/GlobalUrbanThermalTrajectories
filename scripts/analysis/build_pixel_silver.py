from src.pixel_silver import build_pixel_silver, load_pixel_silver_config


def main() -> None:
    config = load_pixel_silver_config()
    build_pixel_silver(config)


if __name__ == "__main__":
    main()
