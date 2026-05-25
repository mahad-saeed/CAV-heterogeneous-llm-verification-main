# Capability-Aware Verification for Heterogeneous Multi-Agent LLM Systems

This repository accompanies our paper on capability-aware verification (CAV), a reliability-weighted aggregation method for heterogeneous multi-agent large language model systems. The central idea is simple: agents are not treated as equally reliable. Instead, the system estimates each agent's capability from calibration and paraphrase consistency, then uses those learned weights during final answer selection.

## Research Context

Most multi-agent LLM setups rely on majority voting or equally weighted aggregation. That is convenient, but it ignores the fact that different models often have different strengths, calibration behavior, and failure modes. CAV is designed for exactly that setting. It is a static verification method rather than a debate protocol: the focus is on how to combine answers from a heterogeneous pool of agents as reliably as possible.

## What the Project Contains

- `config.py` defines the agent tiers, runtime settings, and experiment constants.
- `agents.py` handles model calls, retries, and answer parsing.
- `calibration.py`, `consistency.py`, and `capability.py` compute the final capability weights.
- `debate.py`, `verify.py`, `ablation.py`, and `evaluate.py` run benchmarking, vote aggregation, ablations, and statistical analysis.
- `main.py` runs the full single-seed pipeline.
- `multi_seed.py` aggregates multiple completed seeds.
- `results/` stores the main outputs reported in the paper.
- `results_multi_seed/` stores the two-seed summary tables.

## Experimental Setup

The paper evaluates CAV on three benchmarks:

- GSM8K for numeric reasoning
- MMLU high-school mathematics for multiple-choice reasoning
- TruthfulQA multiple-choice for factual robustness

The system uses a heterogeneous agent pool with several model tiers, then compares CAV against majority voting, random weighting, and best-agent-only baselines.

## Main Outcome

Across the completed seeds, CAV gives a small overall improvement over majority voting, with the strongest gains on MMLU. The results also show that static weights do not transfer equally well across all tasks, especially when moving from math-oriented calibration data to more fact-sensitive questions.

## Repository Layout

```text
.
├── main.py
├── multi_seed.py
├── calibration.py
├── consistency.py
├── capability.py
├── debate.py
├── verify.py
├── ablation.py
├── evaluate.py
├── results/
├── results_multi_seed/
└── .env.example
```

## Setup

Create a `.env` file from `.env.example` and set the required API key and model settings. The project was developed in Python 3.10.

## Run

Typical entry points are:

```bash
python main.py
python multi_seed.py
```

## Notes

- The repository excludes local environment files, Python cache, and LaTeX build artifacts.
- Experiment outputs are included so the paper can be reproduced from the tracked artifacts.