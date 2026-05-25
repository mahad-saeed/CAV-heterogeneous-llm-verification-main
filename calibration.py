import json
import os
import numpy as np
from tqdm import tqdm
from agents import ask_agent
from benchmarks import load_benchmark
from config import ALL_AGENTS, CALIBRATION_SIZE, CALIBRATION_BINS


def _expected_calibration_error(correct: list,
                                confidences: list,
                                n_bins: int = CALIBRATION_BINS) -> float:
    if not correct or not confidences:
        return 1.0

    y = np.array(correct, dtype=float)
    c = np.clip(np.array(confidences, dtype=float), 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            mask = (c >= lo) & (c <= hi)
        else:
            mask = (c > lo) & (c <= hi)

        if mask.any():
            bin_acc = y[mask].mean()
            bin_conf = c[mask].mean()
            ece += mask.mean() * abs(bin_acc - bin_conf)

    return float(ece)


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def run_calibration() -> dict:
    """
    Run each agent on CALIBRATION_SIZE GSM8K train questions.
    Measures accuracy and confidence calibration.
    Returns calibration score per agent.
    """
    print("\n=== PHASE 1: Calibration ===")
    os.makedirs("results", exist_ok=True)
    questions = load_benchmark("gsm8k", split="train",
                               start=0, size=CALIBRATION_SIZE)

    progress_path = "results/calibration_progress.json"
    scores_path = "results/calibration_scores.json"

    progress = {}
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)

    scores = {}
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            scores = json.load(f)

    for agent_name in ALL_AGENTS:
        if agent_name in scores:
            print(f"\n  [CHECKPOINT] Calibration already complete for [{agent_name}]")
            continue

        print(f"\n  Calibrating [{agent_name}]...")
        state = progress.get(agent_name, {})
        start_idx = int(state.get("idx", 0))
        correct = list(state.get("correct", []))
        confidences = list(state.get("confidences", []))

        if start_idx > 0:
            print(f"    Resuming from question {start_idx}/{len(questions)}")

        remaining_questions = questions[start_idx:]
        iterator = enumerate(remaining_questions, start=start_idx)

        for q_idx, item in tqdm(iterator,
                                total=len(questions),
                                initial=start_idx,
                                desc=f"  {agent_name}",
                                leave=False):
            result = ask_agent(agent_name, item["question"],
                               ask_confidence=True)
            is_correct = (result["answer"] == item["answer"])
            correct.append(int(is_correct))
            confidences.append(result["confidence"])

            progress[agent_name] = {
                "idx": q_idx + 1,
                "correct": correct,
                "confidences": confidences,
            }
            _write_json(progress_path, progress)

        accuracy     = np.mean(correct)
        avg_conf     = np.mean(confidences)
        calib_error  = abs(avg_conf - accuracy)
        ece          = _expected_calibration_error(correct, confidences)
        brier        = np.mean((np.array(confidences) - np.array(correct)) ** 2)
        calib_score  = accuracy * (1.0 - calib_error) * (1.0 - ece)

        scores[agent_name] = {
            "accuracy":         round(accuracy, 4),
            "avg_confidence":   round(avg_conf, 4),
            "calibration_error": round(calib_error, 4),
            "ece": round(float(ece), 4),
            "brier": round(float(brier), 4),
            "calibration_score": round(calib_score, 4),
        }

        progress.pop(agent_name, None)
        _write_json(progress_path, progress)
        _write_json(scores_path, scores)

        print(f"    accuracy={accuracy:.2%}  "
              f"avg_conf={avg_conf:.2%}  "
              f"ece={ece:.3f}  "
              f"calib_score={calib_score:.2f}")

    if os.path.exists(progress_path) and not progress:
        os.remove(progress_path)

    return scores
