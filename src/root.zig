//! zig-select: budgeted selection of evidence for a reader with a token budget.
//!
//! - `pack`: walk units in order and keep each one that fits.
//! - `knapsack`: greedy 0-1 knapsack by value density.
//! - `isotonic`: pool-adjacent-violators calibration of scores into probabilities.
//! - `mmr`: maximal marginal relevance ordering.
//! - `submodular`: budgeted greedy for a facility-location coverage objective.
//! - `dpp`: greedy MAP inference for a determinantal point process.
//! - `conformal`: conformal risk control of a threshold.

const std = @import("std");

pub const pack = @import("pack.zig");
pub const knapsack = @import("knapsack.zig");
pub const isotonic = @import("isotonic.zig");
pub const mmr = @import("mmr.zig");
pub const submodular = @import("submodular.zig");
pub const dpp = @import("dpp.zig");
pub const conformal = @import("conformal.zig");

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(pack);
    std.testing.refAllDecls(knapsack);
    std.testing.refAllDecls(isotonic);
    std.testing.refAllDecls(mmr);
    std.testing.refAllDecls(submodular);
    std.testing.refAllDecls(dpp);
    std.testing.refAllDecls(conformal);
    _ = @import("fuzz.zig");
}
