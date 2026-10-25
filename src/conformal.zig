//! Conformal risk control (Angelopoulos, Bates, Fisch, Lei and Schuster, ICLR 2024).
//!
//! Given n calibration items and m candidate thresholds lambda_0 < ... < lambda_{m-1}, with
//! losses non-increasing in the threshold and bounded above by `bound`, the calibrated choice is
//! the smallest threshold whose inflated empirical risk is at most alpha:
//!
//!     (n / (n + 1)) * mean_i loss[i][j] + bound / (n + 1) <= alpha
//!
//! For a new exchangeable item, the expected loss at that threshold is then at most alpha.

const std = @import("std");

/// `losses` is row-major n x m. Returns the index of the chosen threshold, or null when no
/// threshold reaches the target (the target is below what n items can certify).
pub fn calibrate(losses: []const f64, n: usize, m: usize, alpha: f64, bound: f64) error{ Empty, Overflow, LengthMismatch }!?usize {
    if (n == 0) return error.Empty;
    if (losses.len != try std.math.mul(usize, n, m)) return error.LengthMismatch;
    const nf: f64 = @floatFromInt(n);
    for (0..m) |j| {
        var sum: f64 = 0;
        for (0..n) |i| sum += losses[i * m + j];
        const risk = nf / (nf + 1) * (sum / nf) + bound / (nf + 1);
        if (risk <= alpha) return j;
    }
    return null;
}

/// Checks the precondition: `losses` holds n x m values, and every row is non-increasing and
/// within [0, bound]. NaN fails the check.
pub fn isMonotone(losses: []const f64, n: usize, m: usize, bound: f64) bool {
    const len = std.math.mul(usize, n, m) catch return false;
    if (losses.len != len) return false;
    for (0..n) |i| {
        for (0..m) |j| {
            const v = losses[i * m + j];
            if (!(v >= 0 and v <= bound)) return false;
            if (j > 0 and !(v <= losses[i * m + j - 1])) return false;
        }
    }
    return true;
}

test "picks the first threshold under the inflated risk" {
    // Four items; losses at three thresholds.
    const losses = [_]f64{
        1, 1, 0,
        1, 0, 0,
        0, 0, 0,
        1, 0, 0,
    };
    try std.testing.expect(isMonotone(&losses, 4, 3, 1));
    // Means: 0.75, 0.25, 0. Inflated: 0.8, 0.4, 0.2.
    try std.testing.expectEqual(@as(?usize, 1), try calibrate(&losses, 4, 3, 0.45, 1));
    try std.testing.expectEqual(@as(?usize, 2), try calibrate(&losses, 4, 3, 0.3, 1));
    try std.testing.expectEqual(@as(?usize, null), try calibrate(&losses, 4, 3, 0.1, 1));
}

test "NaN losses fail the precondition" {
    const nan = std.math.nan(f64);
    try std.testing.expect(!isMonotone(&.{ nan, 0 }, 1, 2, 1));
    try std.testing.expect(!isMonotone(&.{ 1, nan }, 1, 2, 1));
    try std.testing.expect(!isMonotone(&.{ 1, 0.5 }, 1, 2, nan));
}

test "shape errors" {
    const losses = [_]f64{ 1, 0, 1, 0 };
    try std.testing.expect(!isMonotone(&losses, 3, 2, 1));
    try std.testing.expect(!isMonotone(&losses, std.math.maxInt(usize), 2, 1));
    try std.testing.expectError(error.Empty, calibrate(&.{}, 0, 2, 0.5, 1));
    try std.testing.expectError(error.LengthMismatch, calibrate(&losses, 3, 2, 0.5, 1));
    try std.testing.expectError(error.Overflow, calibrate(&losses, std.math.maxInt(usize), 2, 0.5, 1));
}
