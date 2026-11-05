import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from diagnose_city_month_gold import (
    SLOTS,
    run_diagnostics,
    summarize_qc_completeness,
)


def _wide_frame():
    cities = [
        (1, "Af", "A"),
        (2, "Am", "A"),
        (3, "Aw", "A"),
        (4, "Cfa", "C"),
    ]
    frame = pd.DataFrame(
        cities,
        columns=("ID_UC_G0", "koppen_geiger_class", "climate_major"),
    )
    for code, _, _, _ in SLOTS:
        for month in range(1, 13):
            suffix = f"{code}_m{month:02d}"
            frame[f"qc_{suffix}"] = False
            frame[f"n_years_{suffix}"] = 0
    frame.loc[0, "qc_td_m01"] = True
    frame.loc[0, "n_years_td_m01"] = 17
    frame.loc[1, "n_years_td_m01"] = 17
    frame.loc[3, "qc_td_m01"] = True
    frame.loc[3, "n_years_td_m01"] = 17
    return frame


def _long_frame():
    records = []
    for city_id, climate_class, climate_major in (
        (1, "Af", "A"),
        (2, "Am", "A"),
        (3, "Aw", "A"),
        (4, "Cfa", "C"),
    ):
        for month in range(1, 13):
            for _, slot, _, _ in SLOTS:
                records.append({
                    "ID_UC_G0": city_id,
                    "koppen_geiger_class": climate_class,
                    "climate_major": climate_major,
                    "month": month,
                    "slot": slot,
                    "qc_pass": True,
                    "urban_trend_decade": 0.2,
                    "rural_trend_decade": 0.1,
                    "suhi_trend_decade": 0.1,
                })
    return pd.DataFrame(records)


class CityMonthGoldDiagnosticsTests(unittest.TestCase):
    def test_completeness_counts_include_zero_year_and_failed_city_fits(self):
        summary = summarize_qc_completeness(_wide_frame())
        af = summary.loc[
            (summary["climate_level"] == "class")
            & (summary["climate_zone"] == "Af")
            & (summary["month"] == 1)
            & (summary["slot"] == "terra_day")
        ].iloc[0]
        self.assertEqual(af["n_cities"], 1)
        self.assertEqual(af["n_cities_pass"], 1)
        self.assertEqual(af["n_cities_with_fit"], 1)

        tropical = summary.loc[
            (summary["climate_level"] == "major")
            & (summary["climate_zone"] == "A")
            & (summary["month"] == 1)
            & (summary["slot"] == "terra_day")
        ].iloc[0]
        self.assertEqual(tropical["n_cities"], 3)
        self.assertEqual(tropical["n_cities_pass"], 1)
        self.assertEqual(tropical["n_cities_with_fit"], 2)
        self.assertEqual(tropical["n_cities_zero_valid_years"], 1)

    def test_writes_completeness_tables_and_trend_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_dir = root / "gold"
            gold_dir.mkdir()
            connection = duckdb.connect(":memory:")
            connection.register("wide_frame", _wide_frame())
            connection.register("long_frame", _long_frame())
            connection.execute(
                f"COPY wide_frame TO '{(gold_dir / 'city_month_trend_wide_2003_2019.parquet').as_posix()}' "
                "(FORMAT PARQUET)"
            )
            connection.execute(
                f"COPY long_frame TO '{(gold_dir / 'city_month_slot_trends_2003_2019.parquet').as_posix()}' "
                "(FORMAT PARQUET)"
            )
            connection.close()

            gold_manifest = gold_dir / "manifest.json"
            gold_manifest.write_text(
                json.dumps({"trend_windows": [[2003, 2019]]}),
                encoding="utf-8",
            )
            output_dir = root / "diagnostics"
            config_path = root / "diagnostics_config.json"
            config_path.write_text(
                json.dumps({
                    "paths": {
                        "gold_dir": str(gold_dir),
                        "output_dir": str(output_dir),
                    },
                    "expected_city_count": 4,
                    "plot": {"dpi": 40},
                }),
                encoding="utf-8",
            )

            self.assertEqual(run_diagnostics(config_path), output_dir)
            completeness = pd.read_csv(
                output_dir / "qc_completeness_2003_2019.csv"
            )
            focus = pd.read_csv(output_dir / "qc_Af_Am_Aw_2003_2019.csv")
            af_row = focus.loc[
                (focus["climate_zone"] == "Af")
                & (focus["month"] == 1)
                & (focus["slot"] == "terra_day")
            ].iloc[0]
            self.assertEqual(af_row["n_cities_pass"], 1)
            self.assertEqual(len(completeness), 528)
            self.assertEqual(
                len(list(output_dir.glob("beta_*_climate_*_2003_2019.png"))),
                12,
            )
            self.assertTrue(
                (output_dir / "qc_pass_rate_Af_Am_Aw_2003_2019.png").is_file()
            )
            self.assertTrue((output_dir / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
