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
pub fn calibrate(losses: []const f64, n: usize, m: usize, alpha: f64, bound: f64) ?usize {
    std.debug.assert(losses.len == n * m and n > 0);
    const nf: f64 = @floatFromInt(n);
    for (0..m) |j| {
        var sum: f64 = 0;
        for (0..n) |i| sum += losses[i * m + j];
        const risk = nf / (nf + 1) * (sum / nf) + bound / (nf + 1);
        if (risk <= alpha) return j;
    }
    return null;
}

/// Checks the precondition: every row is non-increasing and within [0, bound].
pub fn isMonotone(losses: []const f64, n: usize, m: usize, bound: f64) bool {
    for (0..n) |i| {
        for (0..m) |j| {
            const v = losses[i * m + j];
            if (v < 0 or v > bound) return false;
            if (j > 0 and v > losses[i * m + j - 1]) return false;
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
    try std.testing.expectEqual(@as(?usize, 1), calibrate(&losses, 4, 3, 0.45, 1));
    try std.testing.expectEqual(@as(?usize, 2), calibrate(&losses, 4, 3, 0.3, 1));
    try std.testing.expectEqual(@as(?usize, null), calibrate(&losses, 4, 3, 0.1, 1));
}
