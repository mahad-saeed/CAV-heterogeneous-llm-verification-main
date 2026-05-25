import os
import json
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime
from config import (
    BENCHMARKS,
    RANDOM_SEED,
    GROQ_AGENTS,
    OLLAMA_AGENTS,
    USE_LOCAL_OLLAMA,
    REQUEST_TIMEOUT_SEC,
    MAX_RETRIES,
)


def _set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _save_run_metadata(start_time: float) -> None:
    duration_sec = round(time.time() - start_time, 2)
    payload = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "seed": RANDOM_SEED,
        "use_local_ollama": USE_LOCAL_OLLAMA,
        "groq_agents": GROQ_AGENTS,
        "ollama_agents": OLLAMA_AGENTS,
        "benchmarks": BENCHMARKS,
        "request_timeout_sec": REQUEST_TIMEOUT_SEC,
        "max_retries": MAX_RETRIES,
        "duration_sec": duration_sec,
    }
    with open("results/run_metadata.json", "w") as f:
        json.dump(payload, f, indent=2)

def main():
    run_start = time.time()
    os.makedirs("results", exist_ok=True)
    _set_reproducible_seed(RANDOM_SEED)

    print("=" * 55)
    print("  Heterogeneous Multi-Agent Debate Experiment")
    print("  CAV: Capability-Aware Verification")
    print("=" * 55)
    print(f"  Seed={RANDOM_SEED}  timeout={REQUEST_TIMEOUT_SEC}s  retries={MAX_RETRIES}")

    # ── Phase 1 & 2: Load from cache or run fresh ──
    weights_file = "results/capability_weights.json"

    if os.path.exists(weights_file):
        print("\n[CHECKPOINT] Found saved weights — skipping calibration & consistency")
        with open(weights_file) as f:
            data = json.load(f)
        weights            = data["weights"]
        calibration_scores = data["calibration"]
        consistency_scores = data["consistency"]
    else:
        from calibration import run_calibration
        from consistency import measure_consistency
        from capability import compute_weights

        calibration_scores = run_calibration()
        consistency_scores = measure_consistency()
        weights = compute_weights(calibration_scores, consistency_scores)

    # ── Phase 3: Run benchmarks (skip if already done) ──
    from debate import run_benchmark
    all_results = {}

    for benchmark in BENCHMARKS:
        csv_path = f"results/{benchmark}_results.csv"
        if os.path.exists(csv_path):
            print(f"\n[CHECKPOINT] Found {csv_path} — skipping {benchmark}")
            all_results[benchmark] = pd.read_csv(csv_path)
        else:
            df = run_benchmark(benchmark, weights)
            all_results[benchmark] = df
            time.sleep(2)

    # ── Phase 4: Ablation ──
    ablation_path = "results/ablation_results.csv"
    if os.path.exists(ablation_path):
        print("\n[CHECKPOINT] Found ablation results — skipping ablation")
        ablation_df = pd.read_csv(ablation_path)
    else:
        from ablation import run_ablation
        combined    = pd.concat(all_results.values(), ignore_index=True)
        ablation_df = run_ablation(combined, calibration_scores,
                                   consistency_scores)

    # ── Phase 5: Evaluate & plot ──
    from evaluate import run_evaluation
    run_evaluation(all_results, ablation_df, weights)

    _save_run_metadata(run_start)

    print("\n=== DONE ===")
    print("All results saved in the results/ folder.")

if __name__ == "__main__":
    main()
