import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class QualityControlConfig:
    min_valid_pixels_per_side_slot: int
    min_clear_days_mean_per_side_slot: float
    min_n8day_mean_per_side_slot: float
    min_valid_months_per_season: int
    min_valid_months_per_year: int


@dataclass(frozen=True)
class EdaConfig:
    start_year: int
    end_year: int
    city_month_aggregation: str
    equatorial_latitude_band_degrees: float
    figure_dpi: int


@dataclass(frozen=True)
class AnalysisConfig:
    urban_raw_dir: Path
    rural_raw_dir: Path
    urban_parquet_dir: Path
    rural_parquet_dir: Path
    monthly_suhi_file: Path
    seasonal_suhi_file: Path
    annual_suhi_file: Path
    city_bounds_shapefile: Path
    city_metadata_file: Path
    eda_figure_dir: Path
    eda_log_dir: Path
    parquet_compression: str
    eda: EdaConfig
    quality_control: QualityControlConfig


def _require_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Analysis config field '{name}' must be a JSON object.")
    return value


def _require_string(mapping: dict[str, object], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Analysis config field '{name}' must be a non-empty string."
        )
    return value


def _require_integer(
    mapping: dict[str, object],
    name: str,
    minimum: int,
    maximum: int
) -> int:
    value = mapping.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(
            f"Analysis config field '{name}' must be an integer "
            f"between {minimum} and {maximum}."
        )
    return value


def _require_nonnegative_number(
    mapping: dict[str, object],
    name: str
) -> float:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Analysis config field '{name}' must be a non-negative number."
        )
    number = float(value)
    if not isfinite(number) or number < 0:
        raise ValueError(
            f"Analysis config field '{name}' must be a finite, "
            "non-negative number."
        )
    return number


def _resolve_path(config_file: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_file.parent / path
    return path.resolve()


def load_analysis_config(config_path: Path | None = None) -> AnalysisConfig:
    config_file = (
        config_path or Path(__file__).resolve().parents[1] / "config.json"
    ).resolve()
    config_data = json.loads(config_file.read_text(encoding="utf-8"))
    root = _require_object(config_data, "root")
    paths = _require_object(root.get("paths"), "paths")
    parquet = _require_object(root.get("parquet"), "parquet")
    eda_data = _require_object(root.get("eda"), "eda")
    quality_control_data = _require_object(
        root.get("quality_control"),
        "quality_control"
    )
    eda_start_year = _require_integer(eda_data, "start_year", 1, 9999)
    eda_end_year = _require_integer(eda_data, "end_year", 1, 9999)
    if eda_start_year > eda_end_year:
        raise ValueError("EDA start_year must not exceed end_year.")
    city_month_aggregation = _require_string(
        eda_data,
        "city_month_aggregation"
    )
    if city_month_aggregation not in {"mean", "median"}:
        raise ValueError(
            "EDA city_month_aggregation must be 'mean' or 'median'."
        )
    equatorial_latitude_band_degrees = _require_nonnegative_number(
        eda_data,
        "equatorial_latitude_band_degrees"
    )
    if equatorial_latitude_band_degrees > 90:
        raise ValueError(
            "EDA equatorial_latitude_band_degrees must not exceed 90."
        )

    return AnalysisConfig(
        urban_raw_dir=_resolve_path(
            config_file, _require_string(paths, "urban_raw_dir")
        ),
        rural_raw_dir=_resolve_path(
            config_file, _require_string(paths, "rural_raw_dir")
        ),
        urban_parquet_dir=_resolve_path(
            config_file, _require_string(paths, "urban_parquet_dir")
        ),
        rural_parquet_dir=_resolve_path(
            config_file, _require_string(paths, "rural_parquet_dir")
        ),
        monthly_suhi_file=_resolve_path(
            config_file, _require_string(paths, "monthly_suhi_file")
        ),
        seasonal_suhi_file=_resolve_path(
            config_file, _require_string(paths, "seasonal_suhi_file")
        ),
        annual_suhi_file=_resolve_path(
            config_file, _require_string(paths, "annual_suhi_file")
        ),
        city_bounds_shapefile=_resolve_path(
            config_file, _require_string(paths, "city_bounds_shapefile")
        ),
        city_metadata_file=_resolve_path(
            config_file, _require_string(paths, "city_metadata_file")
        ),
        eda_figure_dir=_resolve_path(
            config_file, _require_string(paths, "eda_figure_dir")
        ),
        eda_log_dir=_resolve_path(
            config_file, _require_string(paths, "eda_log_dir")
        ),
        parquet_compression=_require_string(parquet, "compression"),
        eda=EdaConfig(
            start_year=eda_start_year,
            end_year=eda_end_year,
            city_month_aggregation=city_month_aggregation,
            equatorial_latitude_band_degrees=(
                equatorial_latitude_band_degrees
            ),
            figure_dpi=_require_integer(eda_data, "figure_dpi", 72, 1200)
        ),
        quality_control=QualityControlConfig(
            min_valid_pixels_per_side_slot=_require_integer(
                quality_control_data,
                "min_valid_pixels_per_side_slot",
                1,
                2147483647
            ),
            min_clear_days_mean_per_side_slot=_require_nonnegative_number(
                quality_control_data,
                "min_clear_days_mean_per_side_slot"
            ),
            min_n8day_mean_per_side_slot=_require_nonnegative_number(
                quality_control_data,
                "min_n8day_mean_per_side_slot"
            ),
            min_valid_months_per_season=_require_integer(
                quality_control_data,
                "min_valid_months_per_season",
                1,
                3
            ),
            min_valid_months_per_year=_require_integer(
                quality_control_data,
                "min_valid_months_per_year",
                1,
                12
            )
        )
    )
