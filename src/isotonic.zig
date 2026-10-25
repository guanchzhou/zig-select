//! Isotonic regression by pool-adjacent-violators: the non-decreasing step function closest to
//! the observations in weighted squared error. Used to turn a retrieval score into a calibrated
//! probability of relevance (Zadrozny and Elkan 2002).
//!
//! Points with equal x are merged first (weighted mean). A fitted model is a list of blocks,
//! each with the smallest x it covers and its value. `predict` returns the value of the last
//! block whose start is at most x, and the first block's value below the first start.

const std = @import("std");

pub const Model = struct {
    starts: []f64,
    values: []f64,

    pub fn deinit(m: Model, gpa: std.mem.Allocator) void {
        gpa.free(m.starts);
        gpa.free(m.values);
    }

    pub fn predict(m: Model, x: f64) f64 {
        var lo: usize = 0;
        var hi: usize = m.starts.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (m.starts[mid] <= x) lo = mid + 1 else hi = mid;
        }
        return m.values[if (lo == 0) 0 else lo - 1];
    }
};

const Point = struct { x: f64, y: f64, w: f64 };

fn byX(_: void, a: Point, b: Point) bool {
    return a.x < b.x;
}

/// Fits a non-decreasing step function. `weights` may be null (all ones). Needs at least one
/// point, and no x may be NaN.
pub fn fit(gpa: std.mem.Allocator, xs: []const f64, ys: []const f64, weights: ?[]const f64) !Model {
    if (xs.len == 0) return error.Empty;
    if (ys.len != xs.len) return error.LengthMismatch;
    if (weights) |w| if (w.len != xs.len) return error.LengthMismatch;
    for (xs) |x| if (std.math.isNan(x)) return error.NotANumber;
    const pts = try gpa.alloc(Point, xs.len);
    defer gpa.free(pts);
    for (pts, xs, ys, 0..) |*p, x, y, i| p.* = .{ .x = x, .y = y, .w = if (weights) |w| w[i] else 1 };
    std.mem.sort(Point, pts, {}, byX);

    // Merge equal x into one point.
    var n: usize = 0;
    for (pts) |p| {
        if (n > 0 and pts[n - 1].x == p.x) {
            const q = &pts[n - 1];
            const w = q.w + p.w;
            q.y = if (w > 0) (q.y * q.w + p.y * p.w) / w else 0.5 * (q.y + p.y);
            q.w = w;
        } else {
            pts[n] = p;
            n += 1;
        }
    }

    // Pool adjacent violators. Block k covers points [first[k], next block's first).
    const first = try gpa.alloc(usize, n);
    defer gpa.free(first);
    const val = try gpa.alloc(f64, n);
    defer gpa.free(val);
    const wt = try gpa.alloc(f64, n);
    defer gpa.free(wt);
    var blocks: usize = 0;
    for (pts[0..n], 0..) |p, i| {
        first[blocks] = i;
        val[blocks] = p.y;
        wt[blocks] = p.w;
        blocks += 1;
        while (blocks > 1 and val[blocks - 2] > val[blocks - 1]) {
            const w = wt[blocks - 2] + wt[blocks - 1];
            val[blocks - 2] = if (w > 0) (val[blocks - 2] * wt[blocks - 2] + val[blocks - 1] * wt[blocks - 1]) / w else 0.5 * (val[blocks - 2] + val[blocks - 1]);
            wt[blocks - 2] = w;
            blocks -= 1;
        }
    }

    const starts = try gpa.alloc(f64, blocks);
    errdefer gpa.free(starts);
    const values = try gpa.alloc(f64, blocks);
    for (0..blocks) |k| {
        starts[k] = pts[first[k]].x;
        values[k] = val[k];
    }
    return .{ .starts = starts, .values = values };
}

test "pools a violating pair into its mean" {
    const m = try fit(std.testing.allocator, &.{ 0.1, 0.2, 0.3, 0.4 }, &.{ 0, 1, 0, 1 }, null);
    defer m.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 3), m.values.len);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), m.predict(0.25), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), m.predict(-1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), m.predict(9), 1e-12);
}

test "equal x values are averaged before pooling" {
    const m = try fit(std.testing.allocator, &.{ 0.5, 0.5, 0.7 }, &.{ 0, 1, 1 }, null);
    defer m.deinit(std.testing.allocator);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), m.predict(0.5), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), m.predict(0.7), 1e-12);
}

test "rejects empty, mismatched and NaN input" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.Empty, fit(gpa, &.{}, &.{}, null));
    try std.testing.expectError(error.LengthMismatch, fit(gpa, &.{ 0.1, 0.2 }, &.{1}, null));
    try std.testing.expectError(error.LengthMismatch, fit(gpa, &.{ 0.1, 0.2 }, &.{ 0, 1 }, &.{1}));
    try std.testing.expectError(error.NotANumber, fit(gpa, &.{ 0.1, std.math.nan(f64) }, &.{ 0, 1 }, null));
}
