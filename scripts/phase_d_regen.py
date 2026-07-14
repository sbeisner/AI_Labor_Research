"""Phase D regeneration batch — one frozen-engine pass over every dataset the
paper quotes.

Runs, IN ORDER, under the engine frozen at git tag `engine-freeze`:

  1. Headline paired runs   -> output/paired_runs.parquet     (100 paired seeds)
                               (old file archived .pre_phase_d.bak first)
  2. Downstream analyses     -> output/industry_analysis.parquet,
                               job_zone_analysis.parquet, wage_heterogeneity.parquet,
                               hard_skill_quintile_analysis.parquet  (50 seeds)
                            -> output/q1_decomposition.parquet   (30 seeds)
  3. eta = 0.05 arm          -> merged into output/eta_sensitivity_v2.parquet
                               (other arms already frozen-engine: file dated
                               2026-07-11, engine source dated 2026-07-10)
  4. info-noise v2           -> output/info_noise_sensitivity_v2.parquet (50 seeds)
  5. adoption-ceiling sweep  -> output/ceiling_sweep.parquet          (100 seeds)

Modes
-----
  --smoke : run every stage at 2 seeds to *_smoke.parquet, validate each schema,
            NEVER touch a real output file. Then print full-run commands with
            wall-time estimates and stop. (This is the pre-handoff check.)
  --full  : the real regeneration. Archives, runs at full seed counts, merges the
            eta arm. Launch this yourself (long compute).

Only stages named on the command line run; with no stage flags, all run.
Stage flags: --paired --downstream --eta --noise --ceiling
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).parent.parent.resolve()
PY = sys.executable
OUT = ROOT / "output"
SCRIPTS = ROOT / "scripts"

# Full-run seed counts / grids (established per-script design; preserved).
PAIRED_SEEDS   = 100
INDUSTRY_SEEDS = 50
Q1_SEEDS       = 30
ETA_NEW_ARM    = "0.05"
ETA_ARM_SEEDS  = 100
NOISE_GRID     = "0,0.1,0.2"
NOISE_SEEDS    = 50
CEILING_GRID   = "0.5,0.75,1.0"
CEILING_SEEDS  = 100

N_TICKS = 180  # analysis window; per-run cost basis for wall estimates

# Canonical paired_runs schema (superset check target for every sweep parquet).
CORE_COLS = {"tick", "seed", "scenario", "Employment_Rate", "unemployment_rate",
             "Emp_Rate_HSQ1_Low", "Emp_Rate_HSQ5_High", "Avg_A_jt"}


def _run(cmd: list[str]) -> float:
    """Run a subprocess, streaming output; return wall seconds. Raise on failure."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.monotonic()
    r = subprocess.run(cmd, cwd=str(ROOT))
    dt = time.monotonic() - t0
    if r.returncode != 0:
        raise SystemExit(f"[phase_d_regen] STAGE FAILED (exit {r.returncode}): {' '.join(cmd)}")
    return dt


def _check_schema(path: pathlib.Path, sweep_col: str | None, expect_seeds: int) -> None:
    import pandas as pd
    if not path.exists():
        raise SystemExit(f"[phase_d_regen] schema check: {path} not written")
    df = pd.read_parquet(path)
    cols = set(df.columns)
    missing = CORE_COLS - cols
    if missing:
        raise SystemExit(f"[phase_d_regen] {path.name}: missing core columns {missing}")
    if sweep_col and sweep_col not in cols:
        raise SystemExit(f"[phase_d_regen] {path.name}: missing sweep column '{sweep_col}'")
    scen = set(df["scenario"].unique())
    if scen != {"AI", "Control"}:
        raise SystemExit(f"[phase_d_regen] {path.name}: scenarios {scen} != AI/Control")
    n_seeds = df["seed"].nunique()
    if n_seeds != expect_seeds:
        raise SystemExit(f"[phase_d_regen] {path.name}: {n_seeds} seeds != expected {expect_seeds}")
    print(f"      schema OK  {path.name}  "
          f"(cols={len(cols)}, seeds={n_seeds}, sweep={sweep_col or '-'}, "
          f"ticks {df['tick'].min()}–{df['tick'].max()})", flush=True)


def _check_analysis_schema(paths: dict[str, set[str]], smoke: bool) -> None:
    """Downstream analyses have bespoke (non-paired) schemas; check key columns."""
    import pandas as pd
    for name, need in paths.items():
        p = OUT / (name if not smoke else name.replace(".parquet", "_smoke.parquet"))
        if not p.exists():
            raise SystemExit(f"[phase_d_regen] schema check: {p} not written")
        cols = set(pd.read_parquet(p).columns)
        miss = need - cols
        if miss:
            raise SystemExit(f"[phase_d_regen] {p.name}: missing columns {miss}")
        print(f"      schema OK  {p.name}  (cols={sorted(cols)})", flush=True)


def merge_eta_arm(arm_path: pathlib.Path, target: pathlib.Path) -> None:
    """Concatenate a freshly-computed eta arm into the frozen-engine eta parquet."""
    import pandas as pd
    base = pd.read_parquet(target)
    arm = pd.read_parquet(arm_path)
    new_eta = float(arm["eta"].iloc[0])
    if new_eta in set(base["eta"].unique()):
        print(f"      eta={new_eta} already present in {target.name}; replacing it.", flush=True)
        base = base[base["eta"] != new_eta]
    merged = (pd.concat([base, arm], ignore_index=True)
              .sort_values(["eta", "seed", "tick"])
              .reset_index(drop=True))
    merged.to_parquet(target, index=False)
    print(f"      merged eta={new_eta} → {target.name}: "
          f"eta arms now {sorted(merged['eta'].unique())}, "
          f"{merged['seed'].nunique()} seeds/arm", flush=True)


# ── Stages ────────────────────────────────────────────────────────────────────
def stage_paired(smoke: bool, times: dict) -> None:
    print("\n[1/5] Headline paired runs …", flush=True)
    if smoke:
        times["paired"] = _run([PY, str(SCRIPTS / "paired_bootstrap.py"), "--smoke"])
        _check_schema(OUT / "paired_runs_smoke.parquet", None, 2)
    else:
        old = OUT / "paired_runs.parquet"
        if old.exists():
            bak = OUT / "paired_runs.parquet.pre_phase_d.bak"
            shutil.copy2(old, bak)
            print(f"      archived {old.name} → {bak.name}", flush=True)
        times["paired"] = _run([PY, str(SCRIPTS / "paired_bootstrap.py"),
                                "--n-runs", str(PAIRED_SEEDS)])
        _check_schema(OUT / "paired_runs.parquet", None, PAIRED_SEEDS)


def stage_downstream(smoke: bool, times: dict) -> None:
    print("\n[2/5] Downstream analyses (industry / job-zone / wage / HSQ, q1) …", flush=True)
    import pandas as pd
    ind_cmd = [PY, str(SCRIPTS / "industry_analysis.py")]
    q1_cmd = [PY, str(SCRIPTS / "q1_decomposition.py")]
    if smoke:
        ind_cmd.append("--smoke")
        q1p = OUT / "q1_decomposition_smoke.parquet"
        q1_cmd += ["--n-runs", "2", "--out", str(q1p)]
    else:
        ind_cmd += ["--n-seeds", str(INDUSTRY_SEEDS)]
        q1p = OUT / "q1_decomposition.parquet"
        q1_cmd += ["--n-runs", str(Q1_SEEDS)]
    times["industry"] = _run(ind_cmd)
    _check_analysis_schema({
        "industry_analysis.parquet": {"scenario", "seed", "industry", "ur", "mean_wage"},
        "job_zone_analysis.parquet": {"scenario", "seed", "job_zone", "ur"},
        "wage_heterogeneity.parquet": {"scenario", "seed", "group_type", "group", "mean_wage"},
        "hard_skill_quintile_analysis.parquet": {"scenario", "seed", "hard_skill_quintile", "ur"},
    }, smoke)
    times["q1"] = _run(q1_cmd)
    df = pd.read_parquet(q1p)
    need = {"scenario", "seed", "tick", "Q1_Displaced",
            "Q1_Credential_Blocked", "Q1_Cascade_Bumped"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"[phase_d_regen] {q1p.name} missing {miss}")
    print(f"      schema OK  {q1p.name} (seeds={df['seed'].nunique()})", flush=True)


def stage_eta(smoke: bool, times: dict) -> None:
    print("\n[3/5] eta = 0.05 arm (merge into eta_sensitivity_v2.parquet) …", flush=True)
    if smoke:
        times["eta"] = _run([PY, str(SCRIPTS / "eta_sensitivity.py"), "--smoke"])
        _check_schema(OUT / "eta_sensitivity_v2_smoke.parquet", "eta", 2)
        print("      (smoke does not merge; --full merges the real 0.05 arm.)", flush=True)
    else:
        arm_path = OUT / "_eta_005_arm.parquet"
        times["eta"] = _run([PY, str(SCRIPTS / "eta_sensitivity.py"),
                             "--grid", ETA_NEW_ARM, "--n-runs", str(ETA_ARM_SEEDS),
                             "--out", str(arm_path)])
        merge_eta_arm(arm_path, OUT / "eta_sensitivity_v2.parquet")
        arm_path.unlink(missing_ok=True)


def stage_noise(smoke: bool, times: dict) -> None:
    print("\n[4/5] Info-noise sweep v2 …", flush=True)
    if smoke:
        out = OUT / "info_noise_sensitivity_v2_smoke.parquet"
        times["noise"] = _run([PY, str(SCRIPTS / "info_noise_sensitivity.py"),
                               "--grid", NOISE_GRID, "--n-runs", "2", "--out", str(out)])
        _check_schema(out, "obs_noise", 2)
    else:
        out = OUT / "info_noise_sensitivity_v2.parquet"
        times["noise"] = _run([PY, str(SCRIPTS / "info_noise_sensitivity.py"),
                               "--grid", NOISE_GRID, "--n-runs", str(NOISE_SEEDS),
                               "--out", str(out)])
        _check_schema(out, "obs_noise", NOISE_SEEDS)


def stage_ceiling(smoke: bool, times: dict) -> None:
    print("\n[5/5] Adoption-ceiling sweep …", flush=True)
    if smoke:
        times["ceiling"] = _run([PY, str(SCRIPTS / "ceiling_sweep.py"), "--smoke"])
        _check_schema(OUT / "ceiling_sweep_smoke.parquet", "ceiling", 2)
    else:
        times["ceiling"] = _run([PY, str(SCRIPTS / "ceiling_sweep.py"),
                                 "--grid", CEILING_GRID, "--n-runs", str(CEILING_SEEDS)])
        _check_schema(OUT / "ceiling_sweep.parquet", "ceiling", CEILING_SEEDS)


# ── Wall-time estimate for the full run ─────────────────────────────────────────
def _print_full_commands(times: dict) -> None:
    """Estimate full-run wall time with a wave model (accurate for these pools).

    Each paired task runs 2 sequential 180-tick model runs (~T_RUN s each). With
    W workers, a stage of `tasks` tasks takes ceil(tasks/W) waves × 2 × T_RUN.
    This is far more accurate than scaling the 2-seed smoke wall, where one-time
    pool startup (loading the 10k-worker frame per worker) dominates.
    """
    import math
    T_RUN = 84.0   # measured: one 180-tick model run ≈ 84s (paired smoke: 2 seeds ≈ 2.8 min)
    W = N_WORKERS
    # stage -> (paired_seeds, arms)
    plan = {
        "paired":   (PAIRED_SEEDS, 1),
        "industry": (INDUSTRY_SEEDS, 1),
        "q1":       (Q1_SEEDS, 1),
        "eta":      (ETA_ARM_SEEDS, 1),
        "noise":    (NOISE_SEEDS, 3),
        "ceiling":  (CEILING_SEEDS, 3),
    }
    print("\n" + "=" * 68)
    print(f"  FULL-RUN WALL-TIME ESTIMATES (wave model, {W} workers, ~{T_RUN:.0f}s/run)")
    print("=" * 68)
    total = 0.0
    for stage, (seeds, arms) in plan.items():
        tasks = seeds * arms
        waves = math.ceil(tasks / W)
        est = waves * 2 * T_RUN
        total += est
        smk = f"smoke {times[stage]:5.0f}s → " if stage in times else ""
        print(f"  {stage:9s}: {smk}full ≈ {est/60:6.1f} min "
              f"({seeds} seeds × {arms} arm = {tasks} tasks, {waves} waves)")
    print("-" * 68)
    print(f"  TOTAL full regeneration ≈ {total/60:.0f} min ({total/3600:.1f} h) on {W} workers")
    print("=" * 68)
    print("\n  To launch the full regeneration yourself:\n")
    print("      python scripts/phase_d_regen.py --full\n")
    print("  Or stage-by-stage (each writes its real output):")
    print(f"      python scripts/paired_bootstrap.py --n-runs {PAIRED_SEEDS}")
    print(f"      python scripts/industry_analysis.py --n-seeds {INDUSTRY_SEEDS}")
    print(f"      python scripts/q1_decomposition.py --n-runs {Q1_SEEDS}")
    print(f"      python scripts/eta_sensitivity.py --grid {ETA_NEW_ARM} "
          f"--n-runs {ETA_ARM_SEEDS} --out output/_eta_005_arm.parquet   "
          f"# then merge (phase_d_regen --full does this)")
    print(f"      python scripts/info_noise_sensitivity.py --grid {NOISE_GRID} "
          f"--n-runs {NOISE_SEEDS} --out output/info_noise_sensitivity_v2.parquet")
    print(f"      python scripts/ceiling_sweep.py --grid {CEILING_GRID} --n-runs {CEILING_SEEDS}")
    print("\n  After the full run completes:")
    print("      python scripts/extract_stats.py")
    print("      python scripts/check_manuscript.py\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true",
                      help="2-seed schema validation of every stage; no real files touched")
    mode.add_argument("--full", action="store_true",
                      help="real regeneration at full seed counts (long compute)")
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--downstream", action="store_true")
    ap.add_argument("--eta", action="store_true")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--ceiling", action="store_true")
    args = ap.parse_args()

    any_stage = args.paired or args.downstream or args.eta or args.noise or args.ceiling
    run_all = not any_stage
    times: dict = {}

    print(f"[phase_d_regen] mode = {'SMOKE' if args.smoke else 'FULL'}", flush=True)
    if run_all or args.paired:     stage_paired(args.smoke, times)
    if run_all or args.downstream: stage_downstream(args.smoke, times)
    if run_all or args.eta:        stage_eta(args.smoke, times)
    if run_all or args.noise:      stage_noise(args.smoke, times)
    if run_all or args.ceiling:    stage_ceiling(args.smoke, times)

    if args.smoke:
        print("\n[phase_d_regen] ALL SMOKE STAGES PASSED — schemas valid.", flush=True)
        _print_full_commands(times)
    else:
        print("\n[phase_d_regen] FULL REGENERATION COMPLETE.", flush=True)
        print("  Next: python scripts/extract_stats.py && python scripts/check_manuscript.py",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
