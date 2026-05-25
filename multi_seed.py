import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple seeded experiments and aggregate publication-ready summaries."
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated list of integer seeds (default: 42,43,44,45,46)",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use for running main.py",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Bootstrap iterations for multi-seed confidence intervals",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for CIs (default 0.05 => 95%% CI)",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip seeds that already have saved results",
    )
    return parser.parse_args()


def _bootstrap_ci(values: np.ndarray,
                  n_boot: int,
                  alpha: float,
                  seed: int = 12345) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(sample.mean())

    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return lo, hi


def _prepare_results_root(project_root: Path,
                          out_root: Path,
                          shared_results: Path) -> None:
    out_root.mkdir(exist_ok=True)
    (out_root / "runs").mkdir(exist_ok=True)

    if shared_results.exists() and any(shared_results.iterdir()):
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup = out_root / f"preexisting_results_{stamp}"
        print(f"[INFO] Moving existing results/ to {backup}")
        shutil.move(str(shared_results), str(backup))

    shared_results.mkdir(exist_ok=True)


def _run_single_seed(project_root: Path,
                     shared_results: Path,
                     runs_root: Path,
                     python_exe: str,
                     seed: int,
                     skip_completed: bool) -> None:
    run_dir = runs_root / f"seed_{seed}"
    done_file = run_dir / "main_results.csv"

    if skip_completed and done_file.exists():
        print(f"[SKIP] seed={seed} already completed: {done_file}")
        return

    if run_dir.exists():
        shutil.rmtree(run_dir)

    if shared_results.exists():
        shutil.rmtree(shared_results)
    shared_results.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["RANDOM_SEED"] = str(seed)

    cmd = [python_exe, "main.py"]
    print(f"[RUN] seed={seed} | command={' '.join(cmd)}")
    subprocess.run(cmd, cwd=project_root, env=env, check=True)

    shutil.move(str(shared_results), str(run_dir))
    shared_results.mkdir(exist_ok=True)
    print(f"[DONE] seed={seed} | saved to {run_dir}")


def _aggregate_multi_seed(runs_root: Path,
                          out_root: Path,
                          n_boot: int,
                          alpha: float) -> None:
    per_seed_frames = []

    for run_dir in sorted(runs_root.glob("seed_*")):
        summary_file = run_dir / "main_results.csv"
        if not summary_file.exists():
            continue

        seed = int(run_dir.name.split("_")[-1])
        df = pd.read_csv(summary_file)
        df["seed"] = seed
        per_seed_frames.append(df)

    if not per_seed_frames:
        raise RuntimeError("No completed seed outputs found under results_multi_seed/runs")

    per_seed = pd.concat(per_seed_frames, ignore_index=True)
    per_seed.to_csv(out_root / "per_seed_main_results.csv", index=False)

    method_cols = [c for c in per_seed.columns if c not in {"Benchmark", "seed"}]

    rows = []
    for benchmark in sorted(per_seed["Benchmark"].unique()):
        sub = per_seed[per_seed["Benchmark"] == benchmark]
        for method in method_cols:
            vals = sub[method].astype(float).values
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci_lo, ci_hi = _bootstrap_ci(vals, n_boot=n_boot, alpha=alpha)
            rows.append({
                "benchmark": benchmark,
                "method": method,
                "n_seeds": len(vals),
                "mean_acc": round(mean_val, 3),
                "std_acc": round(std_val, 3),
                "ci_low": round(ci_lo, 3),
                "ci_high": round(ci_hi, 3),
            })

    agg = pd.DataFrame(rows).sort_values(["benchmark", "mean_acc"], ascending=[True, False])
    agg.to_csv(out_root / "multi_seed_summary.csv", index=False)

    pretty = agg.copy()
    pretty["mean±std"] = pretty.apply(
        lambda r: f"{r['mean_acc']:.2f} +- {r['std_acc']:.2f}", axis=1
    )
    pretty["95% CI"] = pretty.apply(
        lambda r: f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]", axis=1
    )
    pretty = pretty[["benchmark", "method", "n_seeds", "mean±std", "95% CI"]]
    pretty.to_csv(out_root / "multi_seed_summary_pretty.csv", index=False)

    print("\n=== Multi-Seed Aggregation Complete ===")
    print(f"Saved: {out_root / 'per_seed_main_results.csv'}")
    print(f"Saved: {out_root / 'multi_seed_summary.csv'}")
    print(f"Saved: {out_root / 'multi_seed_summary_pretty.csv'}")


def main() -> None:
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("No seeds provided")

    project_root = Path(__file__).resolve().parent
    shared_results = project_root / "results"
    out_root = project_root / "results_multi_seed"
    runs_root = out_root / "runs"

    _prepare_results_root(project_root, out_root, shared_results)

    for seed in seeds:
        _run_single_seed(
            project_root=project_root,
            shared_results=shared_results,
            runs_root=runs_root,
            python_exe=args.python,
            seed=seed,
            skip_completed=args.skip_completed,
        )

    _aggregate_multi_seed(
        runs_root=runs_root,
        out_root=out_root,
        n_boot=args.n_boot,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
