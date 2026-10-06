const std = @import("std");
const select = @import("select");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;

    // Retriever scores with relevance labels from a few judged queries.
    const judged_scores = [_]f64{ 0.2, 0.3, 0.5, 0.6, 0.8, 0.9, 0.95 };
    const judged_labels = [_]f64{ 0, 0, 1, 0, 1, 1, 1 };
    const model = try select.isotonic.fit(gpa, &judged_scores, &judged_labels, null);
    defer model.deinit(gpa);

    // Five candidate chunks for one query. Chunks 0 and 1 say nearly the same thing.
    const scores = [_]f64{ 0.92, 0.90, 0.70, 0.55, 0.30 };
    const costs = [_]u32{ 1800, 1700, 1200, 900, 2500 };
    const sim = [_]f64{
        1.00, 0.97, 0.10, 0.05, 0.00,
        0.97, 1.00, 0.10, 0.05, 0.00,
        0.10, 0.10, 1.00, 0.10, 0.60,
        0.05, 0.05, 0.10, 1.00, 0.00,
        0.00, 0.00, 0.60, 0.00, 1.00,
    };
    var p: [scores.len]f64 = undefined;
    for (scores, &p) |s, *q| q.* = model.predict(s);

    const budget = 4000;
    var by_density: [scores.len]u32 = undefined;
    const k = try select.knapsack.greedy(gpa, &p, &costs, budget, true, &by_density);
    std.debug.assert(k.cost <= budget);
    std.debug.assert(contains(by_density[0..k.kept], 0) and contains(by_density[0..k.kept], 1));

    var by_coverage: [scores.len]u32 = undefined;
    const s = try select.submodular.select(gpa, &p, &sim, &costs, budget, 0, 1, &by_coverage);
    std.debug.assert(s.cost <= budget);
    std.debug.assert(!(contains(by_coverage[0..s.kept], 0) and contains(by_coverage[0..s.kept], 1)));
    std.debug.assert(s.kept > k.kept);
}

fn contains(xs: []const u32, x: u32) bool {
    return std.mem.indexOfScalar(u32, xs, x) != null;
}
