//! Greedy MAP inference for a determinantal point process (Kulesza and Taskar 2012), with the
//! incremental Cholesky update of Chen, Zhang and Zhou (2018). Each step adds the item with the
//! largest remaining d_i^2, the factor by which it multiplies det(L_S); the run stops after
//! `max_items` items or when the best d_i^2 falls below `eps`. The first item is the one with the
//! largest L_ii. Ties go to the lowest index.

const std = @import("std");

/// Builds the quality-diversity kernel L_ij = q_i * <f_i, f_j> * q_j from n feature rows of
/// length d (row-major) into `l` (n x n, row-major).
pub fn kernel(features: []const f64, d: usize, quality: []const f64, l: []f64) void {
    const n = quality.len;
    std.debug.assert(features.len == n * d and l.len == n * n);
    for (0..n) |i| {
        for (i..n) |j| {
            var dot: f64 = 0;
            for (0..d) |k| dot += features[i * d + k] * features[j * d + k];
            const v = quality[i] * dot * quality[j];
            l[i * n + j] = v;
            l[j * n + i] = v;
        }
    }
}

/// Writes the chosen indices, in selection order, into `out` and returns how many were chosen.
pub fn greedy(gpa: std.mem.Allocator, l: []const f64, n: usize, max_items: usize, eps: f64, out: []u32) !usize {
    std.debug.assert(l.len == n * n and out.len >= @min(max_items, n));
    if (n == 0 or max_items == 0) return 0;
    const k_max = @min(max_items, n);
    const cis = try gpa.alloc(f64, k_max * n);
    defer gpa.free(cis);
    @memset(cis, 0);
    const di2 = try gpa.alloc(f64, n);
    defer gpa.free(di2);
    for (0..n) |i| di2[i] = l[i * n + i];

    var j = argmax(di2);
    if (di2[j] < eps or !(di2[j] > 0)) return 0;
    out[0] = @intCast(j);
    var count: usize = 1;
    while (count < k_max) {
        const k = count - 1;
        const dj = @sqrt(di2[j]);
        const cj = cis[j * k_max ..][0..k];
        const lj = l[j * n ..][0..n];
        for (0..n) |i| {
            const ci = cis[i * k_max ..][0..k];
            var dot: f64 = 0;
            for (cj, ci) |a, b| dot += a * b;
            const e = (lj[i] - dot) / dj;
            cis[i * k_max + k] = e;
            di2[i] -= e * e;
        }
        di2[j] = -std.math.inf(f64);
        j = argmax(di2);
        if (di2[j] < eps or !(di2[j] > 0)) break;
        out[count] = @intCast(j);
        count += 1;
    }
    return count;
}

fn argmax(xs: []const f64) usize {
    var best: usize = 0;
    for (xs, 0..) |x, i| if (x > xs[best]) {
        best = i;
    };
    return best;
}

test "prefers a diverse second item over a near duplicate" {
    const f = [_]f64{
        1.0,   0.0,
        0.999, 0.0447,
        0.0,   1.0,
    };
    const q = [_]f64{ 1.0, 1.0, 0.9 };
    var l: [9]f64 = undefined;
    kernel(&f, 2, &q, &l);
    var out: [3]u32 = undefined;
    const got = try greedy(std.testing.allocator, &l, 3, 2, 1e-10, &out);
    try std.testing.expectEqual(@as(usize, 2), got);
    try std.testing.expectEqual(@as(u32, 2), out[1]);
}

test "stops when the kernel has no more rank" {
    // Three items in a 1-dimensional space: rank 1, so only one item can be chosen.
    const f = [_]f64{ 1.0, 0.5, 0.25 };
    const q = [_]f64{ 1.0, 1.0, 1.0 };
    var l: [9]f64 = undefined;
    kernel(&f, 1, &q, &l);
    var out: [3]u32 = undefined;
    const got = try greedy(std.testing.allocator, &l, 3, 3, 1e-10, &out);
    try std.testing.expectEqual(@as(usize, 1), got);
    try std.testing.expectEqual(@as(u32, 0), out[0]);
}

test "rank-0 kernel returns 0 items without dividing by zero" {
    const l = [_]f64{ 0, 0, 0, 0 };
    var out: [2]u32 = undefined;
    const got = try greedy(std.testing.allocator, &l, 2, 2, 1e-10, &out);
    try std.testing.expectEqual(@as(usize, 0), got);
}
