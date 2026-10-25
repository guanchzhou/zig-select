//! The packing rule shared by every selector: walk units in a given order and keep each one
//! whole if it fits in the remaining budget. A unit that does not fit is skipped, and later
//! units are still considered. Units are never truncated.

const std = @import("std");

pub const Packed = struct {
    /// Number of kept units; their indices are the first `kept` entries of the output slice.
    kept: usize,
    /// Sum of the kept units' costs.
    cost: u64,
};

/// Writes the indices of kept units, in walk order, into `out` (at least `order.len` long).
/// Every entry of `order` must index `costs`.
pub fn pack(order: []const u32, costs: []const u32, budget: u64, out: []u32) error{ LengthMismatch, IndexOutOfRange }!Packed {
    if (out.len < order.len) return error.LengthMismatch;
    for (order) |i| if (i >= costs.len) return error.IndexOutOfRange;
    var used: u64 = 0;
    var n: usize = 0;
    for (order) |i| {
        const c = costs[i];
        if (c <= budget - used) {
            used += c;
            out[n] = i;
            n += 1;
        }
    }
    return .{ .kept = n, .cost = used };
}

test "skips a unit that does not fit and keeps a later one" {
    const costs = [_]u32{ 400, 900, 300, 200 };
    const order = [_]u32{ 0, 1, 2, 3 };
    var out: [4]u32 = undefined;
    const p = try pack(&order, &costs, 1000, &out);
    try std.testing.expectEqual(@as(usize, 3), p.kept);
    try std.testing.expectEqual(@as(u64, 900), p.cost);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 3 }, out[0..p.kept]);
}

test "an empty budget keeps only free units" {
    const costs = [_]u32{ 5, 0, 7 };
    const order = [_]u32{ 2, 1, 0 };
    var out: [3]u32 = undefined;
    const p = try pack(&order, &costs, 0, &out);
    try std.testing.expectEqualSlices(u32, &.{1}, out[0..p.kept]);
}

test "a budget near the u64 limit does not overflow" {
    const costs = [_]u32{ std.math.maxInt(u32), std.math.maxInt(u32) };
    var out: [2]u32 = undefined;
    const p = try pack(&.{ 0, 1 }, &costs, std.math.maxInt(u64) - 1, &out);
    try std.testing.expectEqual(@as(usize, 2), p.kept);
}

test "rejects a short output and an index past the costs" {
    const costs = [_]u32{ 1, 2 };
    var out: [2]u32 = undefined;
    try std.testing.expectError(error.LengthMismatch, pack(&.{ 0, 1, 0 }, &costs, 10, &out));
    try std.testing.expectError(error.IndexOutOfRange, pack(&.{ 0, 2 }, &costs, 10, &out));
}
