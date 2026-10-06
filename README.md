# zig-select

Budgeted selection of evidence in pure Zig 0.17. Given candidate chunks with
scores, token costs and similarities, it chooses which ones to hand a reader
that has a fixed token budget, so the budget goes to relevant, non-redundant
text.

- **Packing and knapsack**: keep units whole, never exceed the budget, and
  rank by value per token. With `best_single`, the knapsack result is at
  least half of the optimum.
- **Calibration**: isotonic regression turns raw retriever scores into
  probabilities of relevance, so values from different retrievers can be
  compared and summed.
- **Diversity**: maximal marginal relevance, a budgeted facility-location
  objective, and greedy determinantal point process inference each penalise
  near-duplicates in a different way.
- **Guarantees**: conformal risk control picks a threshold whose expected
  loss on a new query is at most a target you set.
- No dependencies and no C. Every selector is checked against an
  independent NumPy and scikit-learn reference. MIT licensed.

## Install

```sh
zig fetch --save git+https://github.com/guanchzhou/zig-select
```

```zig
const select = b.dependency("zig_select", .{}).module("select");
exe.root_module.addImport("select", select);
```

## Library

| Module | What it does | Reference |
| --- | --- | --- |
| `pack` | Walk units in a given order and keep each one that fits | |
| `knapsack` | Greedy 0-1 knapsack by value density | Dantzig 1957 |
| `isotonic` | Pool-adjacent-violators calibration | Zadrozny and Elkan 2002 |
| `mmr` | Maximal marginal relevance ordering | Carbonell and Goldstein 1998 |
| `submodular` | Budgeted greedy for facility-location coverage | Lin and Bilmes 2011 |
| `dpp` | Greedy MAP inference with incremental Cholesky | Kulesza and Taskar 2012; Chen, Zhang and Zhou 2018 |
| `conformal` | Conformal risk control of a threshold | Angelopoulos et al. 2024 |

Ties go to the lowest index everywhere, so results are deterministic.
Similarity matrices are row-major `n * n` slices. Units are never truncated:
a unit that does not fit is skipped and later units are still considered.

```zig
const model = try select.isotonic.fit(gpa, &judged_scores, &judged_labels, null);
defer model.deinit(gpa);
for (scores, &p) |s, *q| q.* = model.predict(s);

var kept: [n]u32 = undefined;
const r = try select.submodular.select(gpa, &p, &sim, &costs, 6000, 0, 1, &kept);
// kept[0..r.kept] fits in 6000 tokens and covers the candidates
// without spending the budget on near-duplicates.
```

[`examples/evidence.zig`](examples/evidence.zig) runs this on five
candidates where two are near-duplicates. The knapsack keeps both copies; the
coverage objective keeps one and spends the rest of the budget on two more
chunks.

## Command line

`zig build` installs `zig-out/bin/zig-select`. Each command reads one JSON
object on stdin and writes one JSON object on stdout.

```sh
$ echo '{"order":[2,0,1],"costs":[5,4,3],"budget":8}' | zig-select pack
{"kept":[2,0],"cost":8}
$ echo '{"losses":[[1,0.5,0],[1,0.2,0]],"alpha":0.6}' | zig-select conformal
{"index":1}
```

`zig-select --help` lists every command and its fields. Invalid input exits
with 1; an unknown command or a missing command exits with 2.

## Development

```sh
zig build test              # unit, golden, fuzz-corpus, example, and command-line tests
zig build test-cli          # command-line tests only
zig build test --fuzz       # fuzz the selector invariants (needs .zig-cache/tmp to exist)
zig build docs              # API documentation in zig-out/docs
uv run --with numpy --with scikit-learn python test/reference.py   # regenerate test/golden.json
```

`test/reference.py` computes every selector without sharing code with the
Zig implementation, and by a different method where one exists:
scikit-learn for isotonic regression, brute-force determinants for the DPP,
and full-objective differences for the submodular gains. `test/golden.zig`
requires identical selections and values within 1e-9 on 280 cases.

The fuzz targets check invariants on random input: selections stay within
the budget and never repeat a unit, MMR returns a permutation, the DPP
returns distinct items within its limit, and conformal returns the first
threshold that passes.

## Credits

- G. B. Dantzig, "Discrete-variable extremum problems", *Operations
  Research* 5(2), 266-288 (1957).
- B. Zadrozny and C. Elkan, "Transforming classifier scores into accurate
  multiclass probability estimates", *KDD* (2002).
- J. Carbonell and J. Goldstein, "The use of MMR, diversity-based reranking
  for reordering documents and producing summaries", *SIGIR* (1998).
- H. Lin and J. Bilmes, "A class of submodular functions for document
  summarization", *ACL* (2011).
- A. Kulesza and B. Taskar, "Determinantal point processes for machine
  learning", *Foundations and Trends in Machine Learning* 5(2-3) (2012).
- L. Chen, G. Zhang and H. Zhou, "Fast greedy MAP inference for
  determinantal point process to improve recommendation diversity",
  *NeurIPS* (2018).
- A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei and T. Schuster,
  "Conformal risk control", *ICLR* (2024).

## License

MIT. See [LICENSE](LICENSE).
