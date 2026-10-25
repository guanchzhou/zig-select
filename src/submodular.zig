//! Budgeted maximisation of a facility-location objective with a relevance term
//! (Lin and Bilmes 2011):
//!
//!     F(S) = sum_i rel[i] * max_{j in S} max(sim(i, j), 0) + lambda * sum_{j in S} rel[j]
//!
//! F is monotone and submodular for rel >= 0. The cost-scaled greedy repeatedly adds the unit
//! with the largest gain / cost^p among those that still fit, and stops when no fitting unit
//! has a positive gain. The result is the better of that set and the best single unit that
//! fits, which is the step that gives the budgeted greedy its constant-factor guarantee.
//! Ties go to the lowest index.

const std = @import("std");

pub const Result = struct {
    kept: usize,
    cost: u64,
    value: f64,
};

fn objectiveOf(rel: []const f64, sim: []const f64, lambda: f64, set: []const u32) f64 {
    const n = rel.len;
    var total: f64 = 0;
    for (0..n) |i| {
        var best: f64 = 0;
        for (set) |j| best = @max(best, sim[i * n + j]);
        total += rel[i] * best;
    }
    for (set) |j| total += lambda * rel[j];
    return total;
}

/// `sim` is row-major n x n. `rel` must be non-negative. Writes the chosen indices, in the order
/// they were added, into `out`.
pub fn select(
    gpa: std.mem.Allocator,
    rel: []const f64,
    sim: []const f64,
    costs: []const u32,
    budget: u64,
    lambda: f64,
    p: f64,
    out: []u32,
) !Result {
    const n = rel.len;
    if (sim.len != try std.math.mul(usize, n, n) or costs.len != n or out.len < n) return error.LengthMismatch;
    const cover = try gpa.alloc(f64, n);
    defer gpa.free(cover);
    @memset(cover, 0);
    const taken = try gpa.alloc(bool, n);
    defer gpa.free(taken);
    @memset(taken, false);

    var kept: usize = 0;
    var used: u64 = 0;
    var value: f64 = 0;
    while (true) {
        var best: ?usize = null;
        var best_ratio: f64 = 0;
        var best_gain: f64 = 0;
        for (0..n) |j| {
            if (taken[j] or costs[j] > budget - used) continue;
            var gain: f64 = lambda * rel[j];
            for (0..n) |i| {
                const s = @max(sim[i * n + j], 0);
                if (s > cover[i]) gain += rel[i] * (s - cover[i]);
            }
            if (!(gain > 0)) continue;
            const c: f64 = @floatFromInt(costs[j]);
            const ratio = if (p == 0) gain else if (c == 0) std.math.inf(f64) else gain / std.math.pow(f64, c, p);
            if (best == null or ratio > best_ratio or (ratio == best_ratio and gain > best_gain)) {
                best = j;
                best_ratio = ratio;
                best_gain = gain;
            }
        }
        const j = best orelse break;
        taken[j] = true;
        out[kept] = @intCast(j);
        kept += 1;
        used += costs[j];
        value += best_gain;
        for (0..n) |i| cover[i] = @max(cover[i], @max(sim[i * n + j], 0));
    }

    var single: ?u32 = null;
    var single_value: f64 = 0;
    for (0..n) |j| {
        if (costs[j] > budget) continue;
        const v = objectiveOf(rel, sim, lambda, &.{@as(u32, @intCast(j))});
        if (single == null or v > single_value) {
            single = @intCast(j);
            single_value = v;
        }
    }
    if (single) |s| if (single_value > value) {
        out[0] = s;
        return .{ .kept = 1, .cost = costs[s], .value = single_value };
    };
    return .{ .kept = kept, .cost = used, .value = value };
}

test "covers two topics instead of repeating one" {
    // Units 0 and 1 cover the same topic; unit 2 covers another. Equal costs.
    const rel = [_]f64{ 0.9, 0.85, 0.5 };
    const sim = [_]f64{
        1.0,  0.95, 0.0,
        0.95, 1.0,  0.0,
        0.0,  0.0,  1.0,
    };
    const costs = [_]u32{ 100, 100, 100 };
    var out: [3]u32 = undefined;
    const r = try select(std.testing.allocator, &rel, &sim, &costs, 200, 0, 1, &out);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2 }, out[0..r.kept]);
}

test "objective of the chosen set equals the accumulated gains" {
    const rel = [_]f64{ 0.7, 0.2, 0.6, 0.4 };
    const sim = [_]f64{
        1.0, 0.3, 0.5, 0.1,
        0.3, 1.0, 0.2, 0.8,
        0.5, 0.2, 1.0, 0.4,
        0.1, 0.8, 0.4, 1.0,
    };
    const costs = [_]u32{ 50, 20, 80, 30 };
    var out: [4]u32 = undefined;
    const r = try select(std.testing.allocator, &rel, &sim, &costs, 120, 0.5, 0.5, &out);
    try std.testing.expectApproxEqAbs(objectiveOf(&rel, &sim, 0.5, out[0..r.kept]), r.value, 1e-12);
}

test "a NaN gain is never selected" {
    const rel = [_]f64{ std.math.nan(f64), 0.5 };
    const sim = [_]f64{ 1, 0, 0, 1 };
    const costs = [_]u32{ 1, 1 };
    var out: [2]u32 = undefined;
    const r = try select(std.testing.allocator, &rel, &sim, &costs, 1, 0, 1, &out);
    try std.testing.expectEqualSlices(u32, &.{1}, out[0..r.kept]);
}

test "rejects mismatched lengths" {
    const gpa = std.testing.allocator;
    const rel = [_]f64{ 1, 0.5 };
    const sim = [_]f64{ 1, 0, 0, 1 };
    var out: [2]u32 = undefined;
    try std.testing.expectError(error.LengthMismatch, select(gpa, &rel, sim[0..3], &.{ 1, 1 }, 2, 0, 1, &out));
    try std.testing.expectError(error.LengthMismatch, select(gpa, &rel, &sim, &.{1}, 2, 0, 1, &out));
    try std.testing.expectError(error.LengthMismatch, select(gpa, &rel, &sim, &.{ 1, 1 }, 2, 0, 1, out[0..1]));
}
