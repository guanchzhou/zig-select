//! Command-line cases run by `zig build test-cli`. `stdout` is compared
//! exactly, `stderr` as a substring.

pub const Case = struct {
    name: []const u8,
    args: []const []const u8,
    stdin: ?[]const u8 = null,
    stdout: ?[]const u8 = null,
    stderr: ?[]const u8 = null,
    exit: u8 = 0,
};

pub const cases = [_]Case{
    .{
        .name = "pack: keeps the prefix that fits",
        .args = &.{"pack"},
        .stdin = "{\"order\":[2,0,1],\"costs\":[5,4,3],\"budget\":8}",
        .stdout = "{\"kept\":[2,0],\"cost\":8}\n",
    },
    .{
        .name = "knapsack: by value per token",
        .args = &.{"knapsack"},
        .stdin = "{\"values\":[3,1,2],\"costs\":[3,1,4],\"budget\":5}",
        .stdout = "{\"kept\":[0,1],\"cost\":4,\"value\":4}\n",
    },
    .{
        .name = "isotonic: pools the violating pair",
        .args = &.{"isotonic"},
        .stdin = "{\"x\":[1,2,3,4],\"y\":[0,1,0,1],\"predict\":[2.5]}",
        .stdout = "{\"starts\":[1,2,4],\"values\":[0,0.5,1],\"predicted\":[0.5]}\n",
    },
    .{
        .name = "mmr: skips the near duplicate",
        .args = &.{"mmr"},
        .stdin = "{\"relevance\":[0.9,0.8,0.3],\"similarity\":[[1,0.95,0],[0.95,1,0],[0,0,1]],\"lambda\":0.5}",
        .stdout = "{\"order\":[0,2,1]}\n",
    },
    .{
        .name = "submodular: coverage prefers the distinct unit",
        .args = &.{"submodular"},
        .stdin = "{\"relevance\":[0.9,0.85,0.2],\"similarity\":[[1,0.99,0],[0.99,1,0],[0,0,1]],\"costs\":[1,1,1],\"budget\":2,\"lambda\":0,\"p\":1}",
        .stdout = "{\"kept\":[0,2],\"cost\":2,\"value\":1.9415}\n",
    },
    .{
        .name = "dpp: kernel input",
        .args = &.{"dpp"},
        .stdin = "{\"kernel\":[[2,1.9,0],[1.9,2,0],[0,0,1]],\"max_items\":2,\"eps\":1e-9}",
        .stdout = "{\"order\":[0,2]}\n",
    },
    .{
        .name = "dpp: stops when the features are spanned",
        .args = &.{"dpp"},
        .stdin = "{\"features\":[[1,0],[1,0.01],[0,1]],\"quality\":[1,0.95,0.9],\"max_items\":3,\"eps\":1e-9}",
        .stdout = "{\"order\":[0,2]}\n",
    },
    .{
        .name = "conformal: first threshold that passes",
        .args = &.{"conformal"},
        .stdin = "{\"losses\":[[1,0.5,0],[1,0.2,0]],\"alpha\":0.6}",
        .stdout = "{\"index\":1}\n",
    },
    .{
        .name = "conformal: none passes",
        .args = &.{"conformal"},
        .stdin = "{\"losses\":[[1,1],[1,1]],\"alpha\":0.1}",
        .stdout = "{\"index\":null}\n",
    },
    .{
        .name = "error: truncated JSON",
        .args = &.{"pack"},
        .stdin = "{",
        .stderr = "invalid input",
        .exit = 1,
    },
    .{
        .name = "error: missing field",
        .args = &.{"knapsack"},
        .stdin = "{\"order\":[0],\"costs\":[1],\"budget\":1}",
        .stderr = "MissingField",
        .exit = 1,
    },
    .{
        .name = "error: unknown command",
        .args = &.{"bogus"},
        .stderr = "unknown command: bogus",
        .exit = 2,
    },
    .{
        .name = "error: no command prints usage",
        .args = &.{},
        .stderr = "usage: zig-select COMMAND",
        .exit = 2,
    },
    .{
        .name = "help: exits 0",
        .args = &.{"--help"},
    },
};
