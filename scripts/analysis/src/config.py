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
class AnalysisConfig:
    urban_raw_dir: Path
    rural_raw_dir: Path
    urban_parquet_dir: Path
    rural_parquet_dir: Path
    monthly_suhi_file: Path
    seasonal_suhi_file: Path
    annual_suhi_file: Path
    parquet_compression: str
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
    quality_control_data = _require_object(
        root.get("quality_control"),
        "quality_control"
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
        parquet_compression=_require_string(parquet, "compression"),
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
