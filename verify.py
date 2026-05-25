from collections import Counter
import numpy as np


def majority_vote(answers: dict) -> str:
    """Standard majority voting — equal weight for all agents."""
    valid = [a for a in answers.values() if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def weighted_vote(answers: dict, weights: dict) -> str:
    """Capability-weighted voting — weight by reliability score."""
    score_map = {}
    for agent, answer in answers.items():
        if answer is None:
            continue
        w = weights.get(agent, 1.0)
        score_map[answer] = score_map.get(answer, 0.0) + w
    return max(score_map, key=score_map.get) if score_map else None


def random_vote(answers: dict, rng=None) -> str:
    """Baseline: random weights as control condition."""
    valid = {k: v for k, v in answers.items() if v is not None}
    if not valid:
        return None
    agents = list(valid.keys())
    if rng is None:
        rng = np.random.default_rng()
    rand_weights = rng.dirichlet(np.ones(len(agents)))
    score_map = {}
    for agent, w in zip(agents, rand_weights):
        ans = valid[agent]
        score_map[ans] = score_map.get(ans, 0.0) + w
    return max(score_map, key=score_map.get)


def best_agent_only(answers: dict, weights: dict) -> str:
    """Baseline: just use the single highest-weight agent's answer."""
    ranked = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for agent, _ in ranked:
        ans = answers.get(agent)
        if ans is not None:
            return ans
    return None
