# Changelog

## 0.1.2 - 2026-10-25

- Public entry points return errors instead of asserting on caller input: `error.LengthMismatch` for mismatched slice lengths in every selector, `error.IndexOutOfRange` for a `pack` order entry past `costs`, `error.Empty` for empty `isotonic.fit` and `conformal.calibrate` input, `error.NotANumber` for a NaN x in `isotonic.fit`, and `error.Overflow` when `n * n`, `n * d` or `n * m` overflows.
- `pack.pack`, `mmr.order`, `dpp.kernel` and `conformal.calibrate` now return error unions; callers need `try`.
- NaN losses fail `conformal.isMonotone`, which also returns false when `losses` is not n x m; NaN gains are never selected in `submodular.select`.
- Budget checks in `pack` and `submodular.select` no longer overflow for budgets near the `u64` limit.

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
