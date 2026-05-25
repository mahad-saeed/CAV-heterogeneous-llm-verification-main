import random
from datasets import load_dataset
from agents import extract_gsm8k_answer, extract_mcq_answer
from config import TEST_SIZE, CALIBRATION_SIZE, CONSISTENCY_SIZE, RANDOM_SEED


def load_benchmark(name: str, split: str = "test", start: int = 0, size: int = TEST_SIZE):
    """
    Load questions and answers from a benchmark.
    Returns list of dicts with 'question' and 'answer'.
    """
    items = []

    if name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split=split)
        ds = ds.select(range(start, min(start + size, len(ds))))
        for row in ds:
            items.append({
                "question": row["question"],
                "answer": extract_gsm8k_answer(row["answer"])
            })

    elif name == "mmlu":
        # Use a single representative MMLU subject
        ds = load_dataset("cais/mmlu", "high_school_mathematics", split="test")
        ds = ds.select(range(start, min(start + size, len(ds))))
        for row in ds:
            choices = row["choices"]
            choice_str = "\n".join(
                [f"{chr(65+i)}) {c}" for i, c in enumerate(choices)]
            )
            question = f"{row['question']}\n{choice_str}"
            items.append({
                "question": question,
                "answer": extract_mcq_answer(row["answer"])
            })

    elif name == "truthful_qa":
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        ds = ds.select(range(start, min(start + size, len(ds))))
        for local_idx, row in enumerate(ds):
            choices = row["mc1_targets"]["choices"]
            labels  = row["mc1_targets"]["labels"]

            # Deterministically shuffle options to avoid positional bias (e.g., always A).
            pairs = list(zip(choices, labels))
            rng = random.Random(RANDOM_SEED + start + local_idx)
            rng.shuffle(pairs)
            choices = [c for c, _ in pairs]
            labels = [l for _, l in pairs]

            choice_str = "\n".join(
                [f"{chr(65+i)}) {c}" for i, c in enumerate(choices)]
            )
            question = f"{row['question']}\n{choice_str}"
            correct_idx = labels.index(1)
            items.append({
                "question": question,
                "answer": chr(65 + correct_idx)
            })

    return items
