//! Replays test/golden.json, the outputs of the independent reference in test/reference.py.
//! Orders and selections must match exactly; values within 1e-9.

const std = @import("std");
const select = @import("select");

const Golden = struct {
    pack: []const struct { order: []const u32, costs: []const u32, budget: u64, kept: []const u32, cost: u64 },
    knapsack: []const struct {
        values: []const f64,
        costs: []const u32,
        budget: u64,
        best_single: bool,
        kept: []const u32,
        cost: u64,
        value: f64,
    },
    isotonic: []const struct { x: []const f64, y: []const f64, points: []const f64, expected: []const f64 },
    mmr: []const struct { rel: []const f64, sim: []const f64, lambda: f64, order: []const u32 },
    submodular: []const struct {
        rel: []const f64,
        sim: []const f64,
        costs: []const u32,
        budget: u64,
        lambda: f64,
        p: f64,
        kept: []const u32,
        cost: u64,
        value: f64,
    },
    dpp: []const struct {
        features: []const f64,
        d: usize,
        quality: []const f64,
        max_items: usize,
        eps: f64,
        order: []const u32,
    },
    conformal: []const struct { losses: []const f64, n: usize, m: usize, alpha: f64, bound: f64, index: ?usize },
};

fn load() !std.json.Parsed(Golden) {
    return std.json.parseFromSlice(Golden, std.testing.allocator, @embedFile("golden.json"), .{});
}

test "pack matches the reference" {
    const g = try load();
    defer g.deinit();
    for (g.value.pack) |c| {
        const out = try std.testing.allocator.alloc(u32, c.order.len);
        defer std.testing.allocator.free(out);
        const p = try select.pack.pack(c.order, c.costs, c.budget, out);
        try std.testing.expectEqualSlices(u32, c.kept, out[0..p.kept]);
        try std.testing.expectEqual(c.cost, p.cost);
    }
}

test "knapsack matches the reference" {
    const g = try load();
    defer g.deinit();
    for (g.value.knapsack) |c| {
        const out = try std.testing.allocator.alloc(u32, c.values.len);
        defer std.testing.allocator.free(out);
        const r = try select.knapsack.greedy(std.testing.allocator, c.values, c.costs, c.budget, c.best_single, out);
        try std.testing.expectEqualSlices(u32, c.kept, out[0..r.kept]);
        try std.testing.expectEqual(c.cost, r.cost);
        try std.testing.expectApproxEqAbs(c.value, r.value, 1e-9);
    }
}

test "isotonic matches scikit-learn at the fitted points" {
    const g = try load();
    defer g.deinit();
    for (g.value.isotonic) |c| {
        const m = try select.isotonic.fit(std.testing.allocator, c.x, c.y, null);
        defer m.deinit(std.testing.allocator);
        for (c.points, c.expected) |x, want| try std.testing.expectApproxEqAbs(want, m.predict(x), 1e-9);
    }
}

test "mmr matches the reference" {
    const g = try load();
    defer g.deinit();
    for (g.value.mmr) |c| {
        const out = try std.testing.allocator.alloc(u32, c.rel.len);
        defer std.testing.allocator.free(out);
        const scratch = try std.testing.allocator.alloc(f64, c.rel.len);
        defer std.testing.allocator.free(scratch);
        try select.mmr.order(c.rel, c.sim, c.lambda, out, scratch);
        try std.testing.expectEqualSlices(u32, c.order, out);
    }
}

test "submodular matches the reference" {
    const g = try load();
    defer g.deinit();
    for (g.value.submodular) |c| {
        const out = try std.testing.allocator.alloc(u32, c.rel.len);
        defer std.testing.allocator.free(out);
        const r = try select.submodular.select(std.testing.allocator, c.rel, c.sim, c.costs, c.budget, c.lambda, c.p, out);
        try std.testing.expectEqualSlices(u32, c.kept, out[0..r.kept]);
        try std.testing.expectEqual(c.cost, r.cost);
        try std.testing.expectApproxEqAbs(c.value, r.value, 1e-9);
    }
}

test "dpp greedy matches brute-force determinant maximisation" {
    const g = try load();
    defer g.deinit();
    for (g.value.dpp) |c| {
        const n = c.quality.len;
        const l = try std.testing.allocator.alloc(f64, n * n);
        defer std.testing.allocator.free(l);
        try select.dpp.kernel(c.features, c.d, c.quality, l);
        const out = try std.testing.allocator.alloc(u32, n);
        defer std.testing.allocator.free(out);
        const got = try select.dpp.greedy(std.testing.allocator, l, n, c.max_items, c.eps, out);
        try std.testing.expectEqualSlices(u32, c.order, out[0..got]);
    }
}

test "conformal matches the reference" {
    const g = try load();
    defer g.deinit();
    for (g.value.conformal) |c| {
        try std.testing.expect(select.conformal.isMonotone(c.losses, c.n, c.m, c.bound));
        try std.testing.expectEqual(c.index, try select.conformal.calibrate(c.losses, c.n, c.m, c.alpha, c.bound));
    }
}
