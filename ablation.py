import pandas as pd
from config import ALL_AGENTS


def run_ablation(df: pd.DataFrame,
                 calibration_scores: dict,
                 consistency_scores: dict) -> pd.DataFrame:
    """
    Ablation study: compare
      - CAV with calibration only
      - CAV with consistency only
      - CAV with both (full method)
    """
    from verify import weighted_vote
    import numpy as np

    def normalize(raw: dict) -> dict:
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    # Weights using calibration only
    cal_only = normalize({
        a: calibration_scores[a]["calibration_score"]
        for a in ALL_AGENTS
    })

    # Weights using consistency only
    con_only = normalize({
        a: consistency_scores.get(a, 0.5)
        for a in ALL_AGENTS
    })

    results = []
    for benchmark in df["benchmark"].unique():
        sub = df[df["benchmark"] == benchmark]
        questions_data = sub.to_dict("records")

        cal_correct = []
        con_correct = []

        for row in questions_data:
            agent_answers = {
                a: row[f"{a}_answer"] for a in ALL_AGENTS
            }
            true_ans = row["true_answer"]

            cal_ans = weighted_vote(agent_answers, cal_only)
            con_ans = weighted_vote(agent_answers, con_only)

            cal_correct.append(int(cal_ans == true_ans))
            con_correct.append(int(con_ans == true_ans))

        results.append({
            "benchmark":       benchmark,
            "cal_only_acc":    round(sum(cal_correct) / len(cal_correct), 4),
            "con_only_acc":    round(sum(con_correct) / len(con_correct), 4),
            "full_cav_acc":    round(sub["wv_correct"].mean(), 4),
            "majority_acc":    round(sub["mv_correct"].mean(), 4),
        })

    ablation_df = pd.DataFrame(results)
    ablation_df.to_csv("results/ablation_results.csv", index=False)
    return ablation_df
