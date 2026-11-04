#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Build the analysis frames from the raw Stata file data-main.dta
(downloaded + unzipped from Harvard Dataverse doi:10.7910/DVN/TESJMM).
Produces rd_unique.parquet (the unique-threshold sharp-RD sample) and
analysis_full.parquet."""

import pyreadstat

cols = (
    [
        "year",
        "country",
        "citycode",
        "city_pop_new",
        "salary_ppp",
        "salary_group",
        "margin",
        "uniques",
        "non_uniques",
        "cri",
        "cri2",
        "cri3",
        "corr_proc",
        "corr_singleb",
    ]
    + [f"ccode{i}" for i in range(1, 12)]
    + [f"year_{i}" for i in range(1, 16)]
    + [f"threshold_{i}" for i in range(1, 34)]
)

df, meta = pyreadstat.read_dta("data-main.dta", usecols=cols)
print("loaded", df.shape)
print("memory MB:", round(df.memory_usage(deep=True).sum() / 1e6, 1))

print("\n--- margin describe (all) ---")
print(df["margin"].describe())
print("\n--- cri2 describe ---")
print(df["cri2"].describe())
print("\ncountries:", df["country"].value_counts().to_dict())

# unique-threshold sharp RD sample (Table 2 col 1): uniques==1 & |margin|<1
u = df[(df["uniques"] == 1) & (df["margin"].abs() < 1)].copy()
print("\nunique sample rows:", len(u), "cities:", u["citycode"].nunique())
print("margin<0:", (u["margin"] < 0).sum(), "margin>=0:", (u["margin"] >= 0).sum())
print("cri2 non-null in unique sample:", u["cri2"].notna().sum())

# first-stage check: does salary jump at cutoff? (collapse to city-year-salary_group like Fig1a)
fs = df[df["margin"].abs() < 0.1].dropna(subset=["salary_ppp", "margin"])
print("\nfirst-stage rows |margin|<0.1:", len(fs))
print(
    "mean salary_ppp just below (margin in [-0.05,0)):",
    fs.loc[(fs.margin >= -0.05) & (fs.margin < 0), "salary_ppp"].mean(),
)
print(
    "mean salary_ppp just above (margin in [0,0.05)):",
    fs.loc[(fs.margin >= 0) & (fs.margin < 0.05), "salary_ppp"].mean(),
)

# save compact analysis frames
keep = (
    [
        "year",
        "country",
        "citycode",
        "city_pop_new",
        "salary_ppp",
        "salary_group",
        "margin",
        "uniques",
        "non_uniques",
        "cri",
        "cri2",
        "cri3",
        "corr_proc",
        "corr_singleb",
    ]
    + [f"ccode{i}" for i in range(1, 12)]
    + [f"year_{i}" for i in range(1, 16)]
)
df[keep].to_parquet("analysis_full.parquet")
u.to_parquet("rd_unique.parquet")
print("\nsaved analysis_full.parquet + rd_unique.parquet")
