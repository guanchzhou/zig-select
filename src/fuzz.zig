//! Fuzz targets checking invariants that must hold for every input. `zig build test` runs every
//! input in `src/fuzz-corpus/`; `zig build test --fuzz` explores further. Copy any input that
//! finds a bug back into the corpus.

const std = @import("std");
const pack = @import("pack.zig");
const knapsack = @import("knapsack.zig");
const submodular = @import("submodular.zig");
const mmr = @import("mmr.zig");
const dpp = @import("dpp.zig");
const conformal = @import("conformal.zig");
const Smith = std.testing.Smith;

const corpus = [_][]const u8{
    @embedFile("fuzz-corpus/00"),
    @embedFile("fuzz-corpus/01"),
    @embedFile("fuzz-corpus/02"),
    @embedFile("fuzz-corpus/03"),
    @embedFile("fuzz-corpus/04"),
    @embedFile("fuzz-corpus/05"),
};

const max_n = 24;

fn upTo(smith: *Smith, lo: usize, hi: usize) usize {
    return smith.valueRangeAtMost(u32, @intCast(lo), @intCast(hi));
}

fn unitFloat(smith: *Smith) f64 {
    return @as(f64, @floatFromInt(smith.value(u16))) / 65535.0;
}

fn expectDistinct(xs: []const u32, n: usize) !void {
    var seen: [max_n]bool = @splat(false);
    for (xs) |x| {
        try std.testing.expect(x < n);
        try std.testing.expect(!seen[x]);
        seen[x] = true;
    }
}

fn budgeted(_: void, smith: *Smith) anyerror!void {
    const n = upTo(smith, 1, max_n);
    var costs: [max_n]u32 = undefined;
    var values: [max_n]f64 = undefined;
    var order: [max_n]u32 = undefined;
    for (0..n) |i| {
        costs[i] = smith.valueRangeAtMost(u32, 0, 2000);
        values[i] = unitFloat(smith);
        order[i] = @intCast(i);
    }
    // A random permutation from the input.
    var i: usize = n;
    while (i > 1) : (i -= 1) std.mem.swap(u32, &order[i - 1], &order[upTo(smith, 0, i - 1)]);
    const budget = smith.valueRangeAtMost(u64, 0, 8000);

    var out: [max_n]u32 = undefined;
    const p = try pack.pack(order[0..n], costs[0..n], budget, &out);
    try std.testing.expect(p.cost <= budget);
    try expectDistinct(out[0..p.kept], n);

    const k = try knapsack.greedy(std.testing.allocator, values[0..n], costs[0..n], budget, smith.value(bool), &out);
    try std.testing.expect(k.cost <= budget);
    try expectDistinct(out[0..k.kept], n);

    var sim: [max_n * max_n]f64 = undefined;
    for (0..n) |a| for (a..n) |b| {
        const s = if (a == b) 1 else unitFloat(smith) * 2 - 1;
        sim[a * n + b] = s;
        sim[b * n + a] = s;
    };
    const lambda = unitFloat(smith) * 2;
    const s = try submodular.select(std.testing.allocator, values[0..n], sim[0 .. n * n], costs[0..n], budget, lambda, 0.5 + unitFloat(smith), &out);
    try std.testing.expect(s.cost <= budget);
    try expectDistinct(out[0..s.kept], n);

    var scratch: [max_n]f64 = undefined;
    try mmr.order(values[0..n], sim[0 .. n * n], unitFloat(smith), &out, &scratch);
    try expectDistinct(out[0..n], n);
}

test "fuzz: budgeted selectors stay within budget and never repeat a unit" {
    try std.testing.fuzz({}, budgeted, .{ .corpus = &corpus });
}

fn dppInvariants(_: void, smith: *Smith) anyerror!void {
    const n = upTo(smith, 1, max_n);
    const d = upTo(smith, 1, 8);
    var f: [max_n * 8]f64 = undefined;
    var q: [max_n]f64 = undefined;
    for (0..n) |i| {
        for (0..d) |j| f[i * d + j] = unitFloat(smith) * 2 - 1;
        q[i] = 0.1 + unitFloat(smith) * 5;
    }
    var l: [max_n * max_n]f64 = undefined;
    try dpp.kernel(f[0 .. n * d], d, q[0..n], l[0 .. n * n]);
    const max_items = upTo(smith, 1, n);
    var out: [max_n]u32 = undefined;
    const got = try dpp.greedy(std.testing.allocator, l[0 .. n * n], n, max_items, 1e-9, &out);
    try std.testing.expect(got >= 1 and got <= max_items);
    try expectDistinct(out[0..got], n);
}

test "fuzz: dpp greedy returns distinct items within the limit" {
    try std.testing.fuzz({}, dppInvariants, .{ .corpus = &corpus });
}

fn conformalInvariants(_: void, smith: *Smith) anyerror!void {
    const n = upTo(smith, 1, 20);
    const m = upTo(smith, 1, 10);
    var losses: [20 * 10]f64 = undefined;
    for (0..n) |i| {
        var v: f64 = 1;
        for (0..m) |j| {
            if (smith.boolWeighted(1, 3)) v = @max(0, v - unitFloat(smith));
            losses[i * m + j] = v;
        }
    }
    try std.testing.expect(conformal.isMonotone(losses[0 .. n * m], n, m, 1));
    const alpha = unitFloat(smith);
    const chosen = try conformal.calibrate(losses[0 .. n * m], n, m, alpha, 1);
    const nf: f64 = @floatFromInt(n);
    for (0..m) |j| {
        var sum: f64 = 0;
        for (0..n) |i| sum += losses[i * m + j];
        const passes = nf / (nf + 1) * (sum / nf) + 1 / (nf + 1) <= alpha;
        if (chosen) |c| {
            if (j < c) try std.testing.expect(!passes);
            if (j == c) try std.testing.expect(passes);
        } else try std.testing.expect(!passes);
    }
}

test "fuzz: conformal returns the first threshold that passes" {
    try std.testing.fuzz({}, conformalInvariants, .{ .corpus = &corpus });
}
