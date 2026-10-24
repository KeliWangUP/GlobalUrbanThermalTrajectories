# Analysis data preparation

The preprocessing pipeline keeps the downloaded CSVs unchanged and creates:

```text
data/
├── raw/
│   ├── urban/
│   └── rural/
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

Input directories, output file paths, and Parquet compression are set in
[`config.json`](./config.json). Relative paths in that file are resolved from
the config file's directory, so they can be changed without editing the
Python scripts.

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

Seasonal groups use fixed meteorological months: DJF, MAM, JJA, and SON.
December belongs to the following season year. Seasonal and annual means give
each available monthly value equal weight. Outputs include valid-SUHI-month
counts and completeness flags; boundary DJF groups remain present but are
marked incomplete when fewer than three calendar months are available.
LST mean columns are already in degrees Celsius in the GEE exports, so no
additional unit conversion is applied here.
