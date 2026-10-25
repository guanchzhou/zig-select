//! Greedy 0-1 knapsack by value density (Dantzig 1957): items in decreasing value per unit of
//! cost, each kept if it fits. With `best_single`, the single most valuable item that fits
//! replaces the greedy set when it is worth more, which makes the result at least half of the
//! optimum.

const std = @import("std");
const pack = @import("pack.zig");

pub const Result = struct {
    kept: usize,
    cost: u64,
    value: f64,
};

const Ctx = struct {
    values: []const f64,
    costs: []const u32,

    fn density(ctx: Ctx, i: u32) f64 {
        const c = ctx.costs[i];
        if (c == 0) return if (ctx.values[i] > 0) std.math.inf(f64) else 0;
        return ctx.values[i] / @as(f64, @floatFromInt(c));
    }

    fn lessThan(ctx: Ctx, a: u32, b: u32) bool {
        const da = ctx.density(a);
        const db = ctx.density(b);
        if (da != db) return da > db;
        return a < b;
    }
};

/// Writes the indices of the chosen items, in the order they were added, into `out`.
/// `values` must be finite and non-negative.
pub fn greedy(
    gpa: std.mem.Allocator,
    values: []const f64,
    costs: []const u32,
    budget: u64,
    best_single: bool,
    out: []u32,
) !Result {
    if (values.len != costs.len or out.len < values.len) return error.LengthMismatch;
    const order = try gpa.alloc(u32, values.len);
    defer gpa.free(order);
    var pos_len: usize = 0;
    for (values, 0..) |v, i| {
        if (v > 0) {
            order[pos_len] = @intCast(i);
            pos_len += 1;
        }
    }
    const ctx: Ctx = .{ .values = values, .costs = costs };
    std.mem.sort(u32, order[0..pos_len], ctx, Ctx.lessThan);

    const p = pack.pack(order[0..pos_len], costs, budget, out) catch unreachable;
    var value: f64 = 0;
    for (out[0..p.kept]) |i| value += values[i];

    if (best_single) {
        var best: ?u32 = null;
        for (values, costs, 0..) |v, c, i| {
            if (c > budget or !(v > 0)) continue;
            if (best == null or v > values[best.?]) best = @intCast(i);
        }
        if (best) |b| if (values[b] > value) {
            out[0] = b;
            return .{ .kept = 1, .cost = costs[b], .value = values[b] };
        };
    }
    return .{ .kept = p.kept, .cost = p.cost, .value = value };
}

test "orders by value per token and fills the budget" {
    const values = [_]f64{ 0.9, 0.5, 0.4 };
    const costs = [_]u32{ 900, 100, 100 };
    var out: [3]u32 = undefined;
    const r = try greedy(std.testing.allocator, &values, &costs, 1000, false, &out);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, out[0..r.kept]);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9), r.value, 1e-12);
}

test "best single item wins when the greedy set is worth less" {
    const values = [_]f64{ 1.0, 0.1 };
    const costs = [_]u32{ 1000, 1 };
    var out: [2]u32 = undefined;
    const r = try greedy(std.testing.allocator, &values, &costs, 1000, true, &out);
    try std.testing.expectEqualSlices(u32, &.{0}, out[0..r.kept]);
}

test "rejects mismatched lengths" {
    var out: [2]u32 = undefined;
    try std.testing.expectError(error.LengthMismatch, greedy(std.testing.allocator, &.{ 1, 2 }, &.{1}, 10, false, &out));
    try std.testing.expectError(error.LengthMismatch, greedy(std.testing.allocator, &.{ 1, 2, 3 }, &.{ 1, 2, 3 }, 10, false, &out));
}
