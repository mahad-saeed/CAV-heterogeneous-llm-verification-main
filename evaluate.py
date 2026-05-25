import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import wilcoxon
from config import ALL_AGENTS, BENCHMARKS, RANDOM_SEED

matplotlib.rcParams.update({"font.size": 12})


def _bootstrap_ci(binary_values: np.ndarray,
                  n_boot: int = 1000,
                  alpha: float = 0.05,
                  seed: int = RANDOM_SEED):
    rng = np.random.default_rng(seed)
    values = np.array(binary_values, dtype=float)
    if len(values) == 0:
        return 0.0, 0.0

    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(sample.mean())

    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return lo, hi


def _holm_bonferroni(pairs):
    m = len(pairs)
    ordered = sorted(pairs, key=lambda x: x[1])
    adjusted = []
    running_max = 0.0
    for i, (name, p) in enumerate(ordered):
        adj = min(1.0, (m - i) * p)
        running_max = max(running_max, adj)
        adjusted.append((name, running_max))
    return dict(adjusted)


def run_evaluation(all_results: dict, ablation_df: pd.DataFrame,
                   weights: dict):
    """
    Generate all tables, statistical tests, and plots.
    all_results: {benchmark_name: DataFrame}
    """
    print("\n=== PHASE 5: Evaluation & Plots ===\n")

    # ── 1. Main accuracy table ──────────────────────────────
    method_cols = {
        "Majority Voting": "mv_correct",
        "CAV-Full (Ours)": "wv_correct",
        "Random Weights": "rv_correct",
        "Best Agent Only": "bv_correct",
    }
    labels = list(method_cols.keys())
    summary = []
    detailed = []

    for bname, df in all_results.items():
        row = {"Benchmark": bname.upper()}
        for label, col in method_cols.items():
            mean_acc = float(df[col].mean())
            ci_lo, ci_hi = _bootstrap_ci(df[col].values)
            row[label] = round(mean_acc * 100, 1)
            detailed.append({
                "benchmark": bname.upper(),
                "method": label,
                "acc": round(mean_acc * 100, 2),
                "ci95_low": round(ci_lo * 100, 2),
                "ci95_high": round(ci_hi * 100, 2),
            })
        summary.append(row)

    # Compute average across benchmarks
    avg_row = {"Benchmark": "AVERAGE"}
    for l in labels:
        avg_row[l] = round(
            sum(r[l] for r in summary) / len(summary), 1
        )
    summary.append(avg_row)
    summary_df = pd.DataFrame(summary)
    print("── Main Results ──")
    print(summary_df.to_string(index=False))
    summary_df.to_csv("results/main_results.csv", index=False)
    pd.DataFrame(detailed).to_csv("results/main_results_with_ci.csv", index=False)

    # ── 2. Statistical significance (Wilcoxon signed-rank) ──
    print("\n── Statistical Significance (Wilcoxon Test) ──")
    pvals = []
    for bname, df in all_results.items():
        diffs = df["wv_correct"] - df["mv_correct"]
        if int((diffs != 0).sum()) == 0:
            p = 1.0
            pvals.append((bname, float(p)))
            continue
        try:
            stat, p = wilcoxon(df["wv_correct"], df["mv_correct"],
                               zero_method="wilcox", alternative="greater")
        except ValueError:
            p = 1.0
        pvals.append((bname, float(p)))

    pvals_adj = _holm_bonferroni(pvals)
    for bname, p in pvals:
        p_adj = pvals_adj[bname]
        sig = "✓ Significant" if p_adj < 0.05 else "✗ Not significant"
        lift = (all_results[bname]["wv_correct"] -
                all_results[bname]["mv_correct"]).mean() * 100
        print(f"  {bname:<15} p={p:.4f}  p_holm={p_adj:.4f}  lift={lift:+.2f}pp  {sig}")

    # ── 3. Main bar chart ───────────────────────────────────
    bnames  = [r["Benchmark"] for r in summary[:-1]]
    x       = np.arange(len(bnames))
    width   = 0.18
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (l, c) in enumerate(zip(labels, colors)):
        vals = [r[l] for r in summary[:-1]]
        bars = ax.bar(x + i * width, vals, width, label=l,
                      color=c, edgecolor="black", linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Verification Strategy Comparison Across Benchmarks\n"
                 "(Heterogeneous 5-Agent Setting)")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(bnames)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("results/main_comparison.png", dpi=150)
    plt.close()
    print("\n  Saved: results/main_comparison.png")

    # ── 4. Capability weights chart ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    agents  = list(weights.keys())
    vals    = [weights[a] for a in agents]
    colors2 = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(agents)))
    bars    = ax.bar(agents, vals, color=colors2, edgecolor="black")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_ylabel("Normalized Capability Weight")
    ax.set_title("Learned Capability Weights per Agent\n"
                 "(No model identity used — derived from behavior only)")
    ax.set_ylim(0, max(vals) * 1.25)
    plt.tight_layout()
    plt.savefig("results/capability_weights.png", dpi=150)
    plt.close()
    print("  Saved: results/capability_weights.png")

    # ── 5. Ablation study chart ─────────────────────────────
    print("\n── Ablation Study ──")
    print(ablation_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    abl_methods = ["majority_acc", "cal_only_acc",
                   "con_only_acc", "full_cav_acc"]
    abl_labels  = ["Majority Voting", "CAV-Calibration Only",
                   "CAV-Consistency Only", "CAV-Full (Ours)"]
    abl_colors  = ["#4C72B0", "#8172B2", "#C44E52", "#55A868"]

    x2    = np.arange(len(ablation_df))
    width = 0.18
    for i, (m, l, c) in enumerate(
            zip(abl_methods, abl_labels, abl_colors)):
        vals = ablation_df[m].values * 100
        bars = ax.bar(x2 + i * width, vals, width, label=l,
                      color=c, edgecolor="black", linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", fontsize=8,
                    fontweight="bold")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Ablation Study: Contribution of Each CAV Component")
    ax.set_xticks(x2 + width * 1.5)
    ax.set_xticklabels(
        [b.upper() for b in ablation_df["benchmark"]])
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig("results/ablation.png", dpi=150)
    plt.close()
    print("  Saved: results/ablation.png")
