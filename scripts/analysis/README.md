# Analysis data preparation

The preprocessing pipeline keeps the downloaded CSVs unchanged and creates:

```text
data/
├── raw/
│   ├── urban/
│   ├── rural/
│   └── rural_all_cities/
├── parquet/
│   ├── urban/
│   └── rural/
└── processed/
    ├── suhi_monthly.parquet
    ├── suhi_seasonal.parquet
    └── suhi_annual.parquet
```

The existing `data/city` and `data/rural_1700` CSVs were copied byte-for-byte
into the canonical raw folders; the original directories and ZIP archives are
left untouched.

The old priority-only rural CSVs remain in `data/raw/rural`. The analysis
config now reads the replacement all-city exports from
`data/raw/rural_all_cities`, so the two datasets are not mixed or mistaken for
duplicate years.

Input directories, output file paths, and Parquet compression are set in
[`config.json`](./config.json). Relative paths in that file are resolved from
the config file's directory, so they can be changed without editing the
Python scripts. It also contains the quality thresholds used by the analysis:
each urban and rural slot must have at least 5 valid pixels, a spatial-mean
clear-day count of 4, and a spatial-mean valid 8-day composite count of 1.
Both urban and rural values must meet all three thresholds for that slot.
Failed monthly values are set to null in the analysis panel, while the original
QC fields and per-slot pass flags are retained. These are provisional starting
thresholds and are applied together.

Run the steps from the repository root:

```bash
python scripts/analysis/01_convert_exports.py
python scripts/analysis/02_build_monthly_suhi.py
```

The first step validates one row per city/year/month, converts numeric columns,
preserves empty measurements as nulls, and writes one compressed Parquet file
per source year. The second step joins rural rows to urban rows on
`ID_UC_G0`, `year`, and `month`, then calculates each monthly SUHI as urban
mean LST minus rural mean LST for Terra/Aqua day/night.

`suhi_monthly.parquet` is the master analysis table. In addition to the four
slot-level SUHI columns (`suhi_terra_day_mean`, `suhi_terra_night_mean`,
`suhi_aqua_day_mean`, and `suhi_aqua_night_mean`), it includes:

```text
A_T = suhi_terra_day_mean - suhi_terra_night_mean
A_A = suhi_aqua_day_mean - suhi_aqua_night_mean
M_T = (suhi_terra_day_mean + suhi_terra_night_mean) / 2
M_A = (suhi_aqua_day_mean + suhi_aqua_night_mean) / 2
```

Each derived value requires both corresponding day and night SUHI values to
pass QC. Seasonal and annual versions are equal-month averages of the derived
monthly values, so each uses months with valid paired day/night observations.

Seasonal groups use fixed meteorological months: DJF, MAM, JJA, and SON.
December belongs to the following season year. Seasonal and annual means give
each available monthly value equal weight. Outputs include valid-SUHI-month
counts for each slot and derived metric, along with completeness flags. A
season is marked QC-pass per metric with at least 2 valid months; a year
requires at least 9. Boundary DJF groups remain present and are separately
marked incomplete when fewer than three calendar months are available.
Seasonal and annual values are null when their per-metric minimum number of
QC-passing months is not reached.
LST mean columns are already in degrees Celsius in the GEE exports, so no
additional unit conversion is applied here.
