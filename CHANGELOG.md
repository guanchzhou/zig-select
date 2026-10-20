# Changelog

## 0.1.1 - 2026-10-20

- Guard `dpp.greedy` against rank-0 kernels before selecting the first item and store Cholesky factors in item-major layout for unit-stride inner products.
- Clamp `+Inf` similarities in `mmr.order` so they do not collide with the chosen-item sentinel, filter non-positive values in `knapsack.greedy`, handle `p == 0` zero-cost ratios in `submodular.select`, and guard zero-weight merges in `isotonic.fit`.
- Bound CLI `stdin` reads and add GitHub Actions CI workflow.

## 0.1.0 - 2026-10-06

- First release: `pack`, `knapsack`, `isotonic`, `mmr`, `submodular`, `dpp`
  and `conformal`, each checked against an independent NumPy and
  scikit-learn reference on 40 cases.
- The `zig-select` command runs every selector on JSON from stdin.
- Fuzz targets for the budget, uniqueness and threshold invariants.
