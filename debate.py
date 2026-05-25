import os
import re
import pandas as pd
from tqdm import tqdm
from agents import ask_agent
from benchmarks import load_benchmark
from verify import majority_vote, weighted_vote, random_vote, best_agent_only
from config import (
    ALL_AGENTS,
    TEST_SIZE,
    RANDOM_SEED,
    STRICT_BENCHMARK,
    MAX_AGENT_ANSWER_RETRIES,
    MAX_MISSING_AGENTS_PER_QUESTION,
    FAIL_ON_MISSING_AGENTS,
)
import numpy as np


def _is_mcq_answer(answer) -> bool:
    if answer is None:
        return False
    text = str(answer).strip().upper()
    return len(text) == 1 and "A" <= text <= "Z"


def _infer_mcq_choice_count(question: str) -> int:
    if not isinstance(question, str):
        return 0
    letters = re.findall(r"^\s*([A-Z])\)\s", question, flags=re.MULTILINE)
    if not letters:
        return 0
    unique_letters = sorted(set(letters))
    return len(unique_letters)


def _normalize_answer(raw_answer, true_answer, mcq_num_choices: int = 0):
    if raw_answer is None:
        return None

    true_norm = str(true_answer).strip().upper() if true_answer is not None else ""
    text = str(raw_answer).strip().upper()
    if text == "":
        return None

    if _is_mcq_answer(true_norm):
        n = max(mcq_num_choices, ord(true_norm) - ord("A") + 1)
        n = max(2, min(n, 26))
        allowed = [chr(ord("A") + i) for i in range(n)]

        if text in allowed:
            return text

        m_letter = re.search(r"\b([A-Z])\b", text)
        if m_letter:
            letter = m_letter.group(1)
            if letter in allowed:
                return letter

        m_num = re.search(r"-?\d+(?:\.\d+)?", text)
        if not m_num:
            return None

        try:
            n = float(m_num.group(0))
            if not n.is_integer():
                return None
            n = int(n)
        except Exception:
            return None

        if 1 <= n <= len(allowed):
            return allowed[n - 1]
        if 0 <= n < len(allowed):
            return allowed[n]
        return None

    m_num = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if m_num:
        try:
            val = float(m_num.group(0))
            if val.is_integer():
                return str(int(val))
            return str(val)
        except Exception:
            pass

    return text


def _answers_equal(pred, true_answer) -> bool:
    if pred is None or true_answer is None:
        return False

    p = str(pred).strip().upper()
    t = str(true_answer).strip().upper()
    if p == t:
        return True

    try:
        return abs(float(p.replace(",", "")) - float(t.replace(",", ""))) < 1e-9
    except Exception:
        return False


def _benchmark_seed_offset(name: str) -> int:
    # Deterministic benchmark-specific offset (avoid Python hash randomization)
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(name))


def run_benchmark(benchmark_name: str, weights: dict) -> pd.DataFrame:
    """
    Run all agents and all verification methods on one benchmark.
    Returns a DataFrame of per-question results.
    """
    print(f"\n=== Running Benchmark: {benchmark_name.upper()} ===")
    questions = load_benchmark(benchmark_name, split="test",
                               start=0, size=TEST_SIZE)
    final_csv_path = f"results/{benchmark_name}_results.csv"
    progress_csv_path = f"results/{benchmark_name}_progress.csv"

    rows = []
    start_idx = 0
    if os.path.exists(progress_csv_path) and not os.path.exists(final_csv_path):
        progress_df = pd.read_csv(progress_csv_path)
        if STRICT_BENCHMARK and len(progress_df) > 0:
            if "num_missing_agents" in progress_df.columns:
                bad_mask = progress_df["num_missing_agents"] > MAX_MISSING_AGENTS_PER_QUESTION
            else:
                answer_cols = [c for c in [f"{a}_answer" for a in ALL_AGENTS] if c in progress_df.columns]
                bad_mask = (
                    progress_df[answer_cols].isna().sum(axis=1) > MAX_MISSING_AGENTS_PER_QUESTION
                    if answer_cols else pd.Series([False] * len(progress_df), index=progress_df.index)
                )

            if bad_mask.any():
                first_bad_qid = int(progress_df.loc[bad_mask, "question_id"].min())
                print(f"  [CHECKPOINT] Found rows exceeding missing-answer threshold at question {first_bad_qid}; trimming progress before that point")
                progress_df = progress_df[progress_df["question_id"] < first_bad_qid]
                progress_df.to_csv(progress_csv_path, index=False)
        if len(progress_df) > 0 and "question_id" in progress_df.columns:
            if benchmark_name in {"mmlu", "truthful_qa"}:
                for col in ["true_answer", "majority_vote", "weighted_vote", "random_vote", "best_agent"]:
                    if col in progress_df.columns:
                        invalid = progress_df[col].notna() & ~progress_df[col].astype(str).str.strip().str.upper().str.fullmatch(r"[A-Z]")
                        if invalid.any():
                            first_bad_qid = int(progress_df.loc[invalid, "question_id"].min())
                            print(f"  [CHECKPOINT] Found invalid MCQ row at question {first_bad_qid}; trimming progress before that point")
                            progress_df = progress_df[progress_df["question_id"] < first_bad_qid]
                            progress_df.to_csv(progress_csv_path, index=False)
                            break

            rows = progress_df.to_dict("records")
            for r in rows:
                r.setdefault("num_missing_agents", 0)
                r.setdefault("missing_agents", "")
                r.setdefault("exceeds_missing_threshold", 0)
            start_idx = int(progress_df["question_id"].max()) + 1
            print(f"  [CHECKPOINT] Resuming {benchmark_name} from question {start_idx}/{len(questions)}")

    for i, item in enumerate(tqdm(questions[start_idx:], desc=benchmark_name), start=start_idx):
        question   = item["question"]
        mcq_num_choices = _infer_mcq_choice_count(question)
        expected_format = "mcq" if mcq_num_choices >= 2 else None
        true_ans = _normalize_answer(
            item["answer"],
            item["answer"],
            mcq_num_choices=mcq_num_choices,
        )

        # Collect answers from all agents
        agent_answers = {}
        for agent in ALL_AGENTS:
            parsed = None
            for _ in range(MAX_AGENT_ANSWER_RETRIES):
                result = ask_agent(
                    agent,
                    question,
                    expected_format=expected_format,
                    mcq_num_choices=mcq_num_choices,
                )
                parsed = _normalize_answer(
                    result["answer"],
                    true_ans,
                    mcq_num_choices=mcq_num_choices,
                )
                if parsed is not None:
                    break
            agent_answers[agent] = parsed

        missing = [a for a, v in agent_answers.items() if v is None]
        num_missing = len(missing)
        exceeds_missing_threshold = int(num_missing > MAX_MISSING_AGENTS_PER_QUESTION)
        if STRICT_BENCHMARK and exceeds_missing_threshold:
            msg = (
                f"Incomplete agent responses at {benchmark_name} question_id={i}. "
                f"Missing answers from: {missing} (count={num_missing}, "
                f"allowed<={MAX_MISSING_AGENTS_PER_QUESTION})."
            )
            if FAIL_ON_MISSING_AGENTS:
                raise RuntimeError(
                    msg + " Switch API key/quota and rerun to resume from checkpoint."
                )
            print(f"\n  [WARN] {msg} Continuing due tolerance setting.")

        # Apply all verification methods
        deterministic_rng = np.random.default_rng(
            RANDOM_SEED * 1_000_003 + _benchmark_seed_offset(benchmark_name) + i
        )
        mv  = majority_vote(agent_answers)
        wv  = weighted_vote(agent_answers, weights)
        rv  = random_vote(agent_answers, rng=deterministic_rng)
        bv  = best_agent_only(agent_answers, weights)

        row = {
            "benchmark":     benchmark_name,
            "question_id":   i,
            "true_answer":   true_ans,
            "majority_vote": mv,
            "weighted_vote": wv,
            "random_vote":   rv,
            "best_agent":    bv,
            "mv_correct":    int(_answers_equal(mv, true_ans)),
            "wv_correct":    int(_answers_equal(wv, true_ans)),
            "rv_correct":    int(_answers_equal(rv, true_ans)),
            "bv_correct":    int(_answers_equal(bv, true_ans)),
            "num_missing_agents": num_missing,
            "missing_agents": ",".join(missing),
            "exceeds_missing_threshold": exceeds_missing_threshold,
        }
        # Also store each agent's answer and correctness
        for agent in ALL_AGENTS:
            row[f"{agent}_answer"]  = agent_answers[agent]
            row[f"{agent}_correct"] = int(_answers_equal(agent_answers[agent], true_ans))

        rows.append(row)
        pd.DataFrame(rows).to_csv(progress_csv_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(final_csv_path, index=False)
    if os.path.exists(progress_csv_path):
        os.remove(progress_csv_path)
    print(f"  Saved to results/{benchmark_name}_results.csv")
    return df
