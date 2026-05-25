import json
from config import ALPHA


def compute_weights(calibration_scores: dict,
                    consistency_scores: dict) -> dict:
    """
    Combine calibration and consistency into final capability weights.
    w_i = ALPHA * calibration_score + (1-ALPHA) * consistency_score
    Normalize so all weights sum to 1.
    """
    print("\n=== PHASE 3: Computing Capability Weights ===")
    raw = {}

    for agent in calibration_scores:
        cal = calibration_scores[agent]["calibration_score"]
        con = consistency_scores.get(agent, 0.5)
        raw[agent] = ALPHA * cal + (1 - ALPHA) * con

    total = sum(raw.values())
    weights = {k: round(v / total, 4) for k, v in raw.items()}

    print("\n  Agent Capability Weights:")
    for agent, w in sorted(weights.items(), key=lambda x: -x[1]):
        cal = calibration_scores[agent]["calibration_score"]
        con = consistency_scores.get(agent, 0.5)
        print(f"    {agent:<12} cal={cal:.2f}  con={con:.2f}  weight={w:.3f}")

    # Save weights to file
    with open("results/capability_weights.json", "w") as f:
        json.dump({
            "weights": weights,
            "calibration": calibration_scores,
            "consistency": consistency_scores
        }, f, indent=2)

    return weights
