//! Maximal marginal relevance (Carbonell and Goldstein 1998): order items so that each next one
//! maximises lambda * relevance - (1 - lambda) * (largest similarity to an item already chosen).
//! The first item is the most relevant. Ties go to the lowest index.

const std = @import("std");

/// `sim` is the row-major n x n similarity matrix. Writes all n indices into `out` in MMR order;
/// `scratch` must hold n values.
pub fn order(rel: []const f64, sim: []const f64, lambda: f64, out: []u32, scratch: []f64) void {
    const n = rel.len;
    std.debug.assert(sim.len == n * n and out.len >= n and scratch.len >= n);
    const max_sim = scratch[0..n];
    // max_sim[i] is item i's largest similarity to the chosen items; +inf marks a chosen item.
    @memset(max_sim, -std.math.inf(f64));
    var chosen: usize = 0;
    while (chosen < n) : (chosen += 1) {
        var best: usize = n;
        var best_score: f64 = -std.math.inf(f64);
        for (0..n) |i| {
            if (max_sim[i] == std.math.inf(f64)) continue; // already chosen
            const penalty: f64 = if (chosen == 0) 0 else max_sim[i];
            const score = lambda * rel[i] - (1 - lambda) * penalty;
            if (best == n or score > best_score) {
                best = i;
                best_score = score;
            }
        }
        out[chosen] = @intCast(best);
        for (0..n) |i| {
            if (max_sim[i] == std.math.inf(f64)) continue;
            const s = @min(sim[i * n + best], std.math.floatMax(f64));
            if (s > max_sim[i]) max_sim[i] = s;
        }
        max_sim[best] = std.math.inf(f64);
    }
}

test "a near duplicate is pushed down" {
    // Items 0 and 1 are duplicates; 2 is less relevant but different.
    const rel = [_]f64{ 0.9, 0.89, 0.6 };
    const sim = [_]f64{
        1.0,  0.99, 0.1,
        0.99, 1.0,  0.1,
        0.1,  0.1,  1.0,
    };
    var out: [3]u32 = undefined;
    var scratch: [3]f64 = undefined;
    order(&rel, &sim, 0.5, &out, &scratch);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 1 }, &out);
}

test "lambda 1 is the relevance order" {
    const rel = [_]f64{ 0.2, 0.9, 0.5 };
    const sim = [_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    var out: [3]u32 = undefined;
    var scratch: [3]f64 = undefined;
    order(&rel, &sim, 1.0, &out, &scratch);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 0 }, &out);
}

test "non-finite similarity does not corrupt chosen sentinel" {
    const inf = std.math.inf(f64);
    const rel = [_]f64{ 1.0, 0.5 };
    const sim = [_]f64{ inf, inf, inf, inf };
    var out: [2]u32 = undefined;
    var scratch: [2]f64 = undefined;
    order(&rel, &sim, 0.5, &out, &scratch);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, &out);
}
