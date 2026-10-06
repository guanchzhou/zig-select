"""Independent reference outputs for zig-select, written to test/golden.json.

Each algorithm is computed here without sharing code with the Zig implementation, and where
possible by a different method:
- isotonic: scikit-learn's IsotonicRegression, evaluated at the fitted points;
- dpp: brute force, choosing at each step the item that maximises det(L_S) directly;
- submodular: gains computed as differences of the full objective F(S + j) - F(S);
- pack, knapsack, mmr, conformal: direct transcriptions of the definitions.
Ties go to the lowest index everywhere, as in the library.

Run: uv run --with numpy --with scikit-learn python test/reference.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

RNG = np.random.default_rng(20261006)
OUT = Path(__file__).resolve().parent / "golden.json"


def pack(order, costs, budget):
    kept, used = [], 0
    for i in order:
        if used + costs[i] <= budget:
            used += costs[i]
            kept.append(int(i))
    return kept, used


def knapsack(values, costs, budget, best_single):
    def density(i):
        if costs[i] == 0:
            return float("inf") if values[i] > 0 else 0.0
        return values[i] / costs[i]
    order = sorted(range(len(values)), key=lambda i: (-density(i), i))
    kept, used = pack(order, costs, budget)
    value = float(sum(values[i] for i in kept))
    if best_single:
        fitting = [i for i in range(len(values)) if costs[i] <= budget]
        if fitting:
            b = max(fitting, key=lambda i: (values[i], -i))
            if values[b] > value:
                return [b], int(costs[b]), float(values[b])
    return kept, used, value


def mmr(rel, sim, lam):
    n = len(rel)
    chosen = []
    while len(chosen) < n:
        best, best_score = None, None
        for i in range(n):
            if i in chosen:
                continue
            penalty = max((sim[i, j] for j in chosen), default=0.0)
            score = lam * rel[i] - (1 - lam) * penalty
            if best is None or score > best_score:
                best, best_score = i, score
        chosen.append(best)
    return chosen


def objective(rel, sim, lam, s):
    if not s:
        return 0.0
    cover = np.maximum(sim[:, s], 0).max(axis=1)
    return float((rel * cover).sum() + lam * rel[s].sum())


def submodular(rel, sim, costs, budget, lam, p):
    n = len(rel)
    s, used = [], 0
    while True:
        base = objective(rel, sim, lam, s)
        best, best_ratio = None, None
        for j in range(n):
            if j in s or used + costs[j] > budget:
                continue
            gain = objective(rel, sim, lam, s + [j]) - base
            if gain <= 1e-15:
                continue
            ratio = float("inf") if costs[j] == 0 else gain / costs[j] ** p
            if best is None or ratio > best_ratio:
                best, best_ratio = j, ratio
        if best is None:
            break
        s.append(best)
        used += costs[best]
    value = objective(rel, sim, lam, s)
    singles = [(objective(rel, sim, lam, [j]), -j) for j in range(n) if costs[j] <= budget]
    if singles:
        v, neg_j = max(singles)
        if v > value:
            return [-neg_j], int(costs[-neg_j]), v
    return s, int(sum(costs[j] for j in s)), value


def dpp_bruteforce(l, max_items, eps):
    n = l.shape[0]
    chosen = [int(np.argmax(np.diag(l)))]
    prev = l[chosen[0], chosen[0]]
    while len(chosen) < min(max_items, n):
        best, best_ratio = None, None
        for i in range(n):
            if i in chosen:
                continue
            idx = chosen + [i]
            ratio = np.linalg.det(l[np.ix_(idx, idx)]) / prev
            if best is None or ratio > best_ratio:
                best, best_ratio = i, ratio
        if best_ratio < eps:
            break
        chosen.append(best)
        prev = np.linalg.det(l[np.ix_(chosen, chosen)])
    return chosen


def conformal(losses, alpha, bound):
    n = losses.shape[0]
    for j in range(losses.shape[1]):
        risk = n / (n + 1) * losses[:, j].mean() + bound / (n + 1)
        if risk <= alpha:
            return j
    return None


def unit_rows(n, d):
    x = RNG.standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    g = {k: [] for k in ("pack", "knapsack", "isotonic", "mmr", "submodular", "dpp", "conformal")}
    for _ in range(40):
        n = int(RNG.integers(1, 30))
        costs = RNG.integers(0, 900, n).tolist()
        order = RNG.permutation(n).tolist()
        budget = int(RNG.integers(0, 6000))
        kept, used = pack(order, costs, budget)
        g["pack"].append({"order": order, "costs": costs, "budget": budget, "kept": kept, "cost": used})

        values = np.round(RNG.random(n), 6).tolist()
        best_single = bool(RNG.integers(0, 2))
        kept, used, value = knapsack(values, costs, budget, best_single)
        g["knapsack"].append({"values": values, "costs": costs, "budget": budget, "best_single": best_single,
                              "kept": kept, "cost": used, "value": value})

        m = int(RNG.integers(2, 60))
        x = np.round(RNG.random(m), 2)
        y = (RNG.random(m) < x).astype(float)
        ir = IsotonicRegression(increasing=True).fit(x, y)
        points = np.unique(x)
        g["isotonic"].append({"x": x.tolist(), "y": y.tolist(), "points": points.tolist(),
                              "expected": ir.predict(points).tolist()})

        k = int(RNG.integers(1, 15))
        f = unit_rows(k, 8)
        sim = f @ f.T
        rel = RNG.random(k)
        lam = float(RNG.choice([0.0, 0.3, 0.5, 0.7, 0.9, 1.0]))
        g["mmr"].append({"rel": rel.tolist(), "sim": sim.ravel().tolist(), "lambda": lam,
                         "order": mmr(rel, sim, lam)})

        ccosts = RNG.integers(1, 700, k).tolist()
        cbudget = int(RNG.integers(100, 3000))
        lam2 = float(RNG.choice([0.0, 0.5, 1.0, 2.0]))
        p = float(RNG.choice([0.5, 1.0]))
        s, scost, sval = submodular(rel, sim, ccosts, cbudget, lam2, p)
        g["submodular"].append({"rel": rel.tolist(), "sim": sim.ravel().tolist(), "costs": ccosts,
                                "budget": cbudget, "lambda": lam2, "p": p,
                                "kept": s, "cost": scost, "value": sval})

        d = int(RNG.integers(2, 10))
        feats = unit_rows(k, d)
        q = np.exp(float(RNG.choice([1.0, 2.0, 5.0])) * RNG.random(k))
        l = (q[:, None] * (feats @ feats.T)) * q[None, :]
        max_items = int(RNG.integers(1, k + 1))
        g["dpp"].append({"features": feats.ravel().tolist(), "d": d, "quality": q.tolist(),
                         "max_items": max_items, "eps": 1e-9,
                         "order": dpp_bruteforce(l, max_items, 1e-9)})

        cn, cm = int(RNG.integers(1, 80)), int(RNG.integers(1, 12))
        thresholds = np.sort(RNG.random(cm))
        scores = RNG.random(cn)
        # Loss 1 while the item's score exceeds the threshold: non-increasing as thresholds rise.
        losses = (scores[:, None] > thresholds[None, :]).astype(float)
        alpha = float(np.round(RNG.random(), 3))
        g["conformal"].append({"losses": losses.ravel().tolist(), "n": cn, "m": cm, "alpha": alpha,
                               "bound": 1.0, "index": conformal(losses, alpha, 1.0)})
    OUT.write_text(json.dumps(g) + "\n")
    print({k: len(v) for k, v in g.items()}, "->", OUT)


if __name__ == "__main__":
    main()
