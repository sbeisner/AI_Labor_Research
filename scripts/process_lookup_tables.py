"""Build lookup tables for the AI Labor Market ABM from raw data sources.

Produces four parquet files in data/processed/:
  - job_zone_lookup.parquet       OCC2010 -> job_zone (1-5)
  - occ_wage_lookup.parquet       OCC2010 -> median_wage ($K annual)
  - btos_sector_signals.parquet   naics_sector -> a_init, g_init
  - bds_sector_dynamics.parquet   sector -> entry_rate, exit_rate

Run directly (no arguments).  Set FORCE_RERUN = True to rebuild everything.
"""

import sys
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.resolve()
PROCESSED = ROOT / "data" / "processed"
FORCE_RERUN = True


# ---------------------------------------------------------------------------
# 1. Job-zone lookup
# ---------------------------------------------------------------------------

def build_job_zone_lookup():
    out = PROCESSED / "job_zone_lookup.parquet"
    if out.exists() and not FORCE_RERUN:
        print(f"[job_zone_lookup] already exists – skipping (FORCE_RERUN=False)")
        return

    print("[job_zone_lookup] Building …")

    # --- O*NET Job Zones ---
    jz_path = ROOT / "data" / "onet_db_files" / "Job Zones.txt"
    jz = pd.read_csv(jz_path, sep="\t")
    # Strip .xx minor suffix: "11-1011.00" -> "11-1011"
    jz["major_soc"] = jz["O*NET-SOC Code"].str.replace(r"\.\d+$", "", regex=True)
    # One row per major SOC – take the first Job Zone value
    jz_major = (
        jz.sort_values("O*NET-SOC Code")
        .groupby("major_soc", as_index=False)["Job Zone"]
        .first()
    )
    print(f"  O*NET job zones: {len(jz_major)} major SOC codes")

    # --- Crosswalk: OCC2010 <-> Census_SOC ---
    cw_path = ROOT / "data" / "crosswalks" / "occ2010_soc_aioe_crosswalk.parquet"
    cw = pd.read_parquet(cw_path)[["OCC2010", "Census_SOC"]]

    # Join crosswalk -> job zones on Census_SOC == major_soc
    merged = cw.merge(jz_major, left_on="Census_SOC", right_on="major_soc", how="left")
    merged = merged.rename(columns={"Job Zone": "job_zone"})

    # Group by OCC2010 – take modal job_zone; fill missing with 3
    def _mode_first(s):
        m = s.dropna()
        if m.empty:
            return np.nan
        return m.mode().iloc[0]

    result = (
        merged.groupby("OCC2010")["job_zone"]
        .agg(_mode_first)
        .reset_index()
    )
    result["job_zone"] = result["job_zone"].fillna(3).astype(int)

    print(f"  job_zone_lookup: {len(result)} rows")
    result.to_parquet(out, index=False)
    print(f"  Written -> {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 2. Occupation wage lookup
# ---------------------------------------------------------------------------

def build_occ_wage_lookup():
    out = PROCESSED / "occ_wage_lookup.parquet"
    if out.exists() and not FORCE_RERUN:
        print(f"[occ_wage_lookup] already exists – skipping (FORCE_RERUN=False)")
        return

    print("[occ_wage_lookup] Building …")

    # --- OEWS salary data ---
    sal_path = ROOT / "data" / "external" / "felten_aioe" / "Generative AI" / "occ_salary_data_2021.dta"
    sal = pd.read_stata(sal_path)[["occ_code", "median_salary_2021"]]
    print(f"  OEWS salary records: {len(sal)}")

    # --- Crosswalk ---
    cw_path = ROOT / "data" / "crosswalks" / "occ2010_soc_aioe_crosswalk.parquet"
    cw = pd.read_parquet(cw_path)[["OCC2010", "Census_SOC"]]

    # Try exact join first (both already use 7-char major codes like "11-1011")
    merged = sal.merge(cw, left_on="occ_code", right_on="Census_SOC", how="left")

    # For rows that didn't match, strip any trailing ".xx" from occ_code and retry
    unmatched_mask = merged["OCC2010"].isna()
    n_unmatched = unmatched_mask.sum()
    if n_unmatched > 0:
        print(f"  {n_unmatched} OEWS codes unmatched on exact join – retrying with stripped suffix …")
        sal_stripped = sal.copy()
        sal_stripped["occ_code_strip"] = sal_stripped["occ_code"].str.replace(r"\.\d+$", "", regex=True)
        retry = sal_stripped[unmatched_mask.values].merge(
            cw, left_on="occ_code_strip", right_on="Census_SOC", how="left"
        )[["occ_code", "median_salary_2021", "OCC2010", "Census_SOC"]]
        merged = pd.concat(
            [merged[~unmatched_mask], retry],
            ignore_index=True,
        )

    # Drop rows that still have no OCC2010 match
    matched = merged.dropna(subset=["OCC2010"]).copy()
    matched["OCC2010"] = matched["OCC2010"].astype(int)
    print(f"  Matched {len(matched)} OEWS -> OCC2010 pairs")

    # Group by OCC2010, take median wage; fill missing with 45.0
    result = (
        matched.groupby("OCC2010")["median_salary_2021"]
        .median()
        .reset_index()
        .rename(columns={"median_salary_2021": "median_wage"})
    )

    # Ensure every OCC2010 in crosswalk has an entry
    all_occ = cw[["OCC2010"]].drop_duplicates()
    result = all_occ.merge(result, on="OCC2010", how="left")
    result["median_wage"] = result["median_wage"].fillna(45.0)

    print(f"  occ_wage_lookup: {len(result)} rows")
    result.to_parquet(out, index=False)
    print(f"  Written -> {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 3. BTOS sector signals
# ---------------------------------------------------------------------------

def build_btos_sector_signals():
    out = PROCESSED / "btos_sector_signals.parquet"
    if out.exists() and not FORCE_RERUN:
        print(f"[btos_sector_signals] already exists – skipping (FORCE_RERUN=False)")
        return

    print("[btos_sector_signals] Building …")

    btos_path = ROOT / "data" / "external" / "BTOS - Sector.csv"
    # Read with Sector as string; skip footer rows that aren't real sectors
    df = pd.read_csv(btos_path, dtype={"Sector": str})

    # Drop rows where Sector is NaN or clearly not a NAICS/XX code
    # Valid sectors are 2-digit numeric strings or "XX"
    valid_sector_mask = df["Sector"].str.match(r"^\d{2}$|^XX$", na=False)
    df = df[valid_sector_mask].copy()
    df["Question ID"] = pd.to_numeric(df["Question ID"], errors="coerce")
    df["Answer ID"] = pd.to_numeric(df["Answer ID"], errors="coerce")

    # Identify time-period columns (format YYYYPP)
    meta_cols = {"Sector", "Question ID", "Question", "Answer ID", "Answer"}
    time_cols = [c for c in df.columns if c not in meta_cols]

    def parse_pct(v):
        if isinstance(v, str) and v.strip().endswith("%"):
            return float(v.strip().rstrip("%"))
        return float("nan")

    # ---- a_init: Q7 "Yes" (Answer ID == 1) ----
    q7_yes = df[(df["Question ID"] == 7.0) & (df["Answer ID"] == 1.0)].copy()
    q7_yes[time_cols] = q7_yes[time_cols].map(parse_pct)
    # Average most-recent 12 available (non-NaN) time periods per row, divide by 100
    q7_yes["a_init"] = q7_yes[time_cols].apply(
        lambda r: r.dropna().iloc[:12].mean() / 100.0, axis=1
    )
    a_init_df = q7_yes[["Sector", "a_init"]].copy()

    # ---- g_init: Q3 performance score ----
    perf_scores = {1.0: 2, 2.0: 1, 3.0: 0, 4.0: -1, 5.0: -2}
    q3 = df[df["Question ID"] == 3.0].copy()
    q3[time_cols] = q3[time_cols].map(parse_pct)
    q3["score"] = q3["Answer ID"].map(perf_scores)

    def sector_g_init(sub):
        """For one sector's Q3 rows, compute time-weighted performance score."""
        sub = sub.copy()
        # For each time column, compute weighted score if any data present
        period_scores = []
        for tc in time_cols:
            col_data = sub[[tc, "score"]].dropna(subset=[tc])
            if col_data.empty:
                continue
            total_pct = col_data[tc].sum()
            if total_pct == 0:
                continue
            w_score = (col_data[tc] * col_data["score"]).sum() / total_pct
            period_scores.append(w_score)
            if len(period_scores) == 12:
                break
        if not period_scores:
            return np.nan
        # The raw BTOS sentiment score sits in [-2, 2].  Scaling by 0.075 maps
        # it to an annual-equivalent fraction in [-0.15, 0.15].  Because g_jt
        # is applied as a *monthly* capacity modifier in the C* formula, we
        # convert to a true monthly rate using the standard compound formula
        # so that a +6.65 % annual signal becomes ~+0.54 % per month rather
        # than 12× over-amplified.
        annual_g  = float(np.mean(period_scores)) * 0.075
        monthly_g = (1.0 + annual_g) ** (1.0 / 12.0) - 1.0
        return monthly_g

    g_rows = []
    for sector, grp in q3.groupby("Sector"):
        g_rows.append({"Sector": sector, "g_init": sector_g_init(grp)})
    g_init_df = pd.DataFrame(g_rows)

    # ---- Merge ----
    result = a_init_df.merge(g_init_df, on="Sector", how="outer")

    # Fill NaN a_init with sector mean, then with 0.1
    sector_mean_a = result["a_init"].mean()
    result["a_init"] = result["a_init"].fillna(sector_mean_a if not np.isnan(sector_mean_a) else 0.1)
    result["a_init"] = result["a_init"].fillna(0.1)

    # Fill NaN g_init with 0.0
    result["g_init"] = result["g_init"].fillna(0.0)

    # Rename Sector -> naics_sector
    result = result.rename(columns={"Sector": "naics_sector"})

    # Note XX (cross-sector aggregate)
    if "XX" in result["naics_sector"].values:
        xx_row = result[result["naics_sector"] == "XX"].iloc[0]
        print(f"  NOTE: cross-sector aggregate (XX): a_init={xx_row['a_init']:.4f}, g_init={xx_row['g_init']:.4f}")

    print(f"  btos_sector_signals: {len(result)} rows")
    result.to_parquet(out, index=False)
    print(f"  Written -> {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 4. BDS sector dynamics
# ---------------------------------------------------------------------------

def build_bds_sector_dynamics():
    out = PROCESSED / "bds_sector_dynamics.parquet"
    if out.exists() and not FORCE_RERUN:
        print(f"[bds_sector_dynamics] already exists – skipping (FORCE_RERUN=False)")
        return

    print("[bds_sector_dynamics] Building …")

    # --- Sector-level data ---
    sec_path = ROOT / "data" / "raw" / "bds" / "sector.csv"
    sec = pd.read_csv(sec_path)

    # Coerce any "X" suppressed values to NaN
    for col in ["estabs_entry_rate", "firmdeath_firms", "firms"]:
        sec[col] = pd.to_numeric(sec[col], errors="coerce")

    # Filter 2015–2019
    sec15 = sec[sec["year"].between(2015, 2019)].copy()
    print(f"  BDS sector rows 2015-2019: {len(sec15)}")

    # entry_rate = average of estabs_entry_rate / 1000 (rate is per-1000 estabs)
    entry = (
        sec15.groupby("sector")["estabs_entry_rate"]
        .mean()
        .reset_index()
        .rename(columns={"estabs_entry_rate": "entry_rate"})
    )
    entry["entry_rate"] = entry["entry_rate"] / 1000.0

    # exit_rate = average of (firmdeath_firms / firms)
    sec15 = sec15.copy()
    sec15["firm_exit_frac"] = sec15["firmdeath_firms"] / sec15["firms"]
    exit_ = (
        sec15.groupby("sector")["firm_exit_frac"]
        .mean()
        .reset_index()
        .rename(columns={"firm_exit_frac": "exit_rate"})
    )

    result = entry.merge(exit_, on="sector", how="outer")
    result["entry_rate"] = result["entry_rate"].fillna(result["entry_rate"].mean())
    result["exit_rate"] = result["exit_rate"].fillna(result["exit_rate"].mean())

    # --- National commentary from firm-age data ---
    fa_path = ROOT / "data" / "raw" / "bds" / "firm-age.csv"
    fa = pd.read_csv(fa_path)
    for col in ["firms"]:
        fa[col] = pd.to_numeric(fa[col], errors="coerce")
    fa15 = fa[fa["year"].between(2015, 2019)].copy()

    nat_entry_rates = []
    for yr, grp in fa15.groupby("year"):
        total_firms = grp["firms"].sum()
        new_firms = grp.loc[grp["fage"] == "a) 0", "firms"].sum()
        if total_firms > 0:
            nat_entry_rates.append(new_firms / total_firms)

    nat_entry_rate = float(np.mean(nat_entry_rates)) if nat_entry_rates else np.nan
    print(f"  [commentary] National annual firm entry rate 2015-2019 (firm-age): {nat_entry_rate:.4f}")

    print(f"  bds_sector_dynamics: {len(result)} rows")
    result.to_parquet(out, index=False)
    print(f"  Written -> {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)

    build_job_zone_lookup()
    build_occ_wage_lookup()
    build_btos_sector_signals()
    build_bds_sector_dynamics()

    print()
    print("[process_lookup_tables] Written:")
    for name in [
        "job_zone_lookup.parquet",
        "occ_wage_lookup.parquet",
        "btos_sector_signals.parquet",
        "bds_sector_dynamics.parquet",
    ]:
        p = PROCESSED / name
        if p.exists():
            df = pd.read_parquet(p)
            print(f"  data/processed/{name:<42} {len(df)} rows")
        else:
            print(f"  data/processed/{name:<42} NOT FOUND")
