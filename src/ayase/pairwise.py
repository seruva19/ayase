"""Bradley–Terry ratings and prompt-level bootstrap intervals."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, cast

import numpy as np

_Edge = tuple[str, str]
_Edges = Mapping[_Edge, list[float]]


def _components(models: set[str], edges: _Edges) -> list[list[str]]:
    neighbors: dict[str, set[str]] = {name: set() for name in models}
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    remaining = set(models)
    groups = []
    while remaining:
        pending = [min(remaining)]
        component = set()
        while pending:
            name = pending.pop()
            if name not in component:
                component.add(name)
                pending.extend(neighbors[name] - component)
        groups.append(sorted(component))
        remaining.difference_update(component)
    return groups


def _fit(models: list[str], edges: _Edges) -> np.ndarray:
    from scipy.optimize import minimize
    from scipy.special import expit

    indices = {name: i for i, name in enumerate(models)}
    pairs = sorted(edges)
    a = np.array([indices[pair[0]] for pair in pairs], dtype=int)
    b = np.array([indices[pair[1]] for pair in pairs], dtype=int)
    # Symmetric half-win priors keep separated comparisons finite.
    wins = np.asarray([edges[pair] for pair in pairs], dtype=float) + 0.5

    def loss(free: np.ndarray) -> tuple[float, np.ndarray]:
        strengths = np.append(free, 0.0)
        difference = strengths[a] - strengths[b]
        value: Any = np.sum(
            wins[:, 0] * np.logaddexp(0.0, -difference) + wins[:, 1] * np.logaddexp(0.0, difference)
        )
        derivative = wins.sum(axis=1) * expit(difference) - wins[:, 0]
        gradient = np.bincount(a, weights=derivative, minlength=len(models)) - np.bincount(
            b, weights=derivative, minlength=len(models)
        )
        return float(value), gradient[:-1]

    fit = minimize(
        loss,
        np.zeros(len(models) - 1),
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-13, "gtol": 1e-8},
    )
    if not fit.success:
        raise ValueError(f"Bradley–Terry optimization did not converge: {fit.message}")
    strengths = np.append(fit.x, 0.0)
    return cast(np.ndarray, strengths - strengths.mean())


def _aggregate(prompts: Iterable[_Edges], multiplicities: Iterable[int]) -> _Edges:
    total: defaultdict[_Edge, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for edges, count in zip(prompts, multiplicities):
        if count:
            for pair, wins in edges.items():
                total[pair][0] += wins[0] * count
                total[pair][1] += wins[1] * count
    return total


def compare(
    rows: Iterable[Mapping[str, Any]], *, draws: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Fit comparison components and bootstrap their centered log-strengths."""
    if draws < 0 or seed < 0:
        raise ValueError("draws and seed must be nonnegative")
    by_prompt: defaultdict[str, defaultdict[_Edge, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0])
    )
    models: set[str] = set()
    count = ties = 0
    for number, row in enumerate(rows, 1):
        try:
            prompt, a, b, outcome = (
                row[key] for key in ("prompt", "model_a", "model_b", "outcome")
            )
        except (TypeError, KeyError) as exc:
            raise ValueError(f"row {number}: expected prompt, model_a, model_b, outcome") from exc
        if not all(isinstance(value, str) and value.strip() for value in (prompt, a, b)):
            raise ValueError(f"row {number}: prompt and model identifiers must be nonempty strings")
        if a == b or outcome not in ("a", "b", "tie"):
            raise ValueError(f"row {number}: use distinct models and outcome a, b, or tie")
        score = 0.5 if outcome == "tie" else float(outcome == "a")
        pair = (a, b) if a < b else (b, a)
        if pair[0] != a:
            score = 1.0 - score
        edge = by_prompt[prompt][pair]
        edge[0] += score
        edge[1] += 1.0 - score
        count += 1
        ties += int(outcome == "tie")
        models.update(pair)
    if not count:
        raise ValueError("at least one comparison is required")

    prompts = [by_prompt[name] for name in sorted(by_prompt)]
    observed = _aggregate(prompts, [1] * len(prompts))
    groups = _components(models, observed)
    owner = {name: i for i, group in enumerate(groups) for name in group}

    def split(edges: _Edges) -> list[dict[_Edge, list[float]]]:
        partition: list[dict[_Edge, list[float]]] = [{} for _ in groups]
        for pair, wins in edges.items():
            partition[owner[pair[0]]][pair] = wins
        return partition

    estimates = [_fit(group, edges) for group, edges in zip(groups, split(observed))]
    bootstrap: list[list[np.ndarray]] = [[] for _ in groups]
    prompt_counts = [0] * len(groups)
    for edges in prompts:
        for index in {owner[pair[0]] for pair in edges}:
            prompt_counts[index] += 1
    rng = np.random.default_rng(seed)
    if max(prompt_counts) > 1:
        for _ in range(draws):
            multiplicities = np.bincount(
                rng.integers(len(prompts), size=len(prompts)), minlength=len(prompts)
            )
            for i, bootstrap_edges in enumerate(split(_aggregate(prompts, multiplicities))):
                group = groups[i]
                if prompt_counts[i] > 1 and len(_components(set(group), bootstrap_edges)) == 1:
                    bootstrap[i].append(_fit(group, bootstrap_edges))

    components = []
    for group, values, replicates, n_prompts in zip(groups, estimates, bootstrap, prompt_counts):
        intervals = (
            np.percentile(replicates, [2.5, 97.5], axis=0).T if len(replicates) >= 2 else None
        )
        order = sorted(range(len(group)), key=lambda i: (-round(float(values[i]), 8), group[i]))
        ratings = []
        previous = None
        rank = 0
        for position, index in enumerate(order, 1):
            rounded = round(float(values[index]), 8)
            if previous != rounded:
                rank = position
                previous = rounded
            ratings.append(
                {
                    "model": group[index],
                    "rank": rank,
                    "log_strength": float(values[index]),
                    "ci95": None if intervals is None else intervals[index].tolist(),
                }
            )
        components.append(
            {"prompt_count": n_prompts, "bootstrap_accepted": len(replicates), "ratings": ratings}
        )
    return {
        "method": "bradley-terry",
        "tie_policy": "half_win",
        "prior_wins_per_side": 0.5,
        "connected": len(groups) == 1,
        "comparison_count": count,
        "tie_count": ties,
        "prompt_count": len(prompts),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "components": components,
    }
