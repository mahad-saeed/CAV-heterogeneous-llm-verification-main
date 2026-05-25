import json
import os
import re
import time
import numpy as np
from tqdm import tqdm
from groq import Groq
from agents import ask_agent
from benchmarks import load_benchmark
from config import GROQ_API_KEY, ALL_AGENTS, CONSISTENCY_SIZE, \
                   NUM_PARAPHRASES, CALIBRATION_SIZE, \
                   REQUEST_TIMEOUT_SEC, MAX_RETRIES, RETRY_BACKOFF_SEC, \
                   PARAPHRASE_MODEL, RATE_LIMIT_MAX_SLEEP_SEC, \
                   STOP_ON_RATE_LIMIT, MAX_TOKENS_PARAPHRASE

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def _call_with_retries(call_fn, label: str):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_fn()
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                err_text = str(exc)
                wait_sec = RETRY_BACKOFF_SEC * attempt
                rate_limit_match = re.search(
                    r"Please try again in (?:(\d+)m)?([0-9]+(?:\.[0-9]+)?)s",
                    err_text,
                )
                if rate_limit_match:
                    mins = int(rate_limit_match.group(1) or 0)
                    secs = float(rate_limit_match.group(2))
                    parsed_wait = mins * 60 + secs + 1.0
                    wait_sec = max(wait_sec, min(parsed_wait, RATE_LIMIT_MAX_SLEEP_SEC))
                print(f"\n  [WARN] {label}: attempt {attempt}/{MAX_RETRIES} failed ({exc}); retrying in {wait_sec:.1f}s")
                time.sleep(wait_sec)
    raise last_error


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def generate_paraphrases(question: str, n: int = NUM_PARAPHRASES) -> list:
    """Generate n paraphrases of a question using a fast model."""
    prompt = (
        f"Rewrite the following question in {n} different ways. "
        f"Keep the exact same meaning. "
        f"Output only the rewritten questions, numbered 1 to {n}.\n\n"
        f"Question: {question}"
    )
    try:
        if groq_client is None:
            raise RuntimeError("GROQ_API_KEY is not set; cannot generate paraphrases.")

        def paraphrase_once():
            req = {
                "model": PARAPHRASE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": MAX_TOKENS_PARAPHRASE,
            }
            try:
                return groq_client.chat.completions.create(
                    timeout=REQUEST_TIMEOUT_SEC,
                    **req,
                )
            except TypeError:
                return groq_client.chat.completions.create(**req)

        response = _call_with_retries(paraphrase_once, "paraphrase")
        text = response.choices[0].message.content
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        paraphrases = []
        for line in lines:
            clean = line.lstrip("0123456789.)- ").strip()
            if len(clean) > 15:
                paraphrases.append(clean)
        return paraphrases[:n]
    except Exception as e:
        err_text = str(e).lower()
        if STOP_ON_RATE_LIMIT and (
            "rate_limit_exceeded" in err_text
            or "rate limit reached" in err_text
        ):
            raise
        print(f"\n  [paraphrase error] {e}")
        return []


def measure_consistency() -> dict:
    """
    For each agent, check if it gives the same answer to
    paraphrased versions of the same question.
    Returns consistency score per agent (0 to 1).
    """
    print("\n=== PHASE 2: Consistency Check ===")
    os.makedirs("results", exist_ok=True)

    # Use questions right after calibration set
    questions = load_benchmark("gsm8k", split="train",
                               start=CALIBRATION_SIZE,
                               size=CONSISTENCY_SIZE)

    progress_path = "results/consistency_progress.json"
    scores_path = "results/consistency_scores.json"
    paraphrase_cache_path = "results/consistency_paraphrases.json"

    progress = {}
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)

    scores = {}
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            scores = json.load(f)

    paraphrase_cache = {}
    if os.path.exists(paraphrase_cache_path):
        with open(paraphrase_cache_path) as f:
            paraphrase_cache = json.load(f)

    print("\n  Preparing paraphrases (shared across agents)...")
    for idx, item in enumerate(tqdm(questions, desc="  paraphrases", leave=False)):
        key = str(idx)
        if key in paraphrase_cache:
            continue
        paraphrase_cache[key] = generate_paraphrases(item["question"])
        _write_json(paraphrase_cache_path, paraphrase_cache)

    for agent_name in ALL_AGENTS:
        if agent_name in scores:
            print(f"\n  [CHECKPOINT] Consistency already complete for [{agent_name}]")
            continue

        print(f"\n  Checking consistency [{agent_name}]...")
        state = progress.get(agent_name, {})
        start_idx = int(state.get("idx", 0))
        consistency_vals = list(state.get("consistency_vals", []))

        if start_idx > 0:
            print(f"    Resuming from question {start_idx}/{len(questions)}")

        remaining_questions = questions[start_idx:]
        iterator = enumerate(remaining_questions, start=start_idx)

        for q_idx, item in tqdm(iterator,
                                total=len(questions),
                                initial=start_idx,
                                desc=f"  {agent_name}",
                                leave=False):
            paraphrases = paraphrase_cache.get(str(q_idx), [])
            if not paraphrases:
                continue

            original = ask_agent(agent_name, item["question"])["answer"]
            para_answers = [
                ask_agent(agent_name, p)["answer"] for p in paraphrases
            ]

            matches = sum(1 for a in para_answers if a == original)
            consistency_vals.append(matches / len(para_answers))

            progress[agent_name] = {
                "idx": q_idx + 1,
                "consistency_vals": consistency_vals,
            }
            _write_json(progress_path, progress)

        score = np.mean(consistency_vals) if consistency_vals else 0.5
        scores[agent_name] = round(score, 4)

        progress.pop(agent_name, None)
        _write_json(progress_path, progress)
        _write_json(scores_path, scores)

        print(f"    consistency score = {score:.2f}")

    if os.path.exists(progress_path) and not progress:
        os.remove(progress_path)
    if os.path.exists(paraphrase_cache_path):
        os.remove(paraphrase_cache_path)

    return scores
