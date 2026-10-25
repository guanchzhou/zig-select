//! zig-select command line. Each command reads one JSON object on stdin and writes one JSON
//! object on stdout. Matrices are JSON arrays of rows.

const std = @import("std");
const select = @import("select");
const build_options = @import("build_options");

const usage =
    \\usage: zig-select COMMAND < input.json
    \\
    \\  pack        {"order":[i..],"costs":[c..],"budget":B}
    \\              -> {"kept":[i..],"cost":C}
    \\  knapsack    {"values":[v..],"costs":[c..],"budget":B,"best_single":bool}
    \\              -> {"kept":[i..],"cost":C,"value":V}
    \\  isotonic    {"x":[..],"y":[..],"weights":[..]?,"predict":[..]}
    \\              -> {"starts":[..],"values":[..],"predicted":[..]}
    \\  mmr         {"relevance":[..],"similarity":[[..]..],"lambda":L}
    \\              -> {"order":[i..]}
    \\  submodular  {"relevance":[..],"similarity":[[..]..],"costs":[c..],"budget":B,"lambda":L,"p":P}
    \\              -> {"kept":[i..],"cost":C,"value":V}
    \\  dpp         {"features":[[..]..],"quality":[..],"max_items":K,"eps":E}
    \\              or {"kernel":[[..]..],"max_items":K,"eps":E}
    \\              -> {"order":[i..]}
    \\  conformal   {"losses":[[..]..],"alpha":A,"bound":B}
    \\              -> {"index":J|null}
    \\  version
    \\
;

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("zig-select: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}

fn flatten(arena: std.mem.Allocator, rows: []const []const f64, cols: ?usize) ![]f64 {
    const width = cols orelse (if (rows.len == 0) 0 else rows[0].len);
    const out = try arena.alloc(f64, rows.len * width);
    for (rows, 0..) |r, i| {
        if (r.len != width) fail("matrix row {d} has {d} values, expected {d}", .{ i, r.len, width });
        @memcpy(out[i * width ..][0..width], r);
    }
    return out;
}

fn square(arena: std.mem.Allocator, rows: []const []const f64, n: usize, name: []const u8) ![]f64 {
    if (rows.len != n) fail("{s} has {d} rows, expected {d}", .{ name, rows.len, n });
    return flatten(arena, rows, n);
}

fn parse(comptime T: type, arena: std.mem.Allocator, text: []const u8) T {
    return std.json.parseFromSliceLeaky(T, arena, text, .{ .ignore_unknown_fields = true }) catch |e|
        fail("invalid input: {s}", .{@errorName(e)});
}

fn emit(out: *std.Io.Writer, v: anytype) !void {
    try std.json.Stringify.value(v, .{}, out);
    try out.writeAll("\n");
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    var buf: [64 * 1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &buf);
    const out = &stdout.interface;
    defer out.flush() catch {};
    if (args.len != 2) {
        std.debug.print("{s}", .{usage});
        std.process.exit(2);
    }
    const cmd: []const u8 = args[1];
    if (std.mem.eql(u8, cmd, "version") or std.mem.eql(u8, cmd, "--version")) {
        try out.print("zig-select {s}\n", .{build_options.version});
        return;
    }
    if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        try out.writeAll(usage);
        return;
    }
    const commands = [_][]const u8{ "pack", "knapsack", "isotonic", "mmr", "submodular", "dpp", "conformal" };
    for (commands) |c| {
        if (std.mem.eql(u8, cmd, c)) break;
    } else {
        std.debug.print("zig-select: unknown command: {s}\n{s}", .{ cmd, usage });
        std.process.exit(2);
    }

    var in_buf: [64 * 1024]u8 = undefined;
    var stdin = std.Io.File.stdin().readerStreaming(init.io, &in_buf);
    const text = stdin.interface.allocRemaining(arena, .limited(256 * 1024 * 1024)) catch |e| switch (e) {
        error.ReadFailed => return stdin.err.?,
        else => |x| return x,
    };

    if (std.mem.eql(u8, cmd, "pack")) {
        const in = parse(struct { order: []const u32, costs: []const u32, budget: u64 }, arena, text);
        for (in.order) |i| if (i >= in.costs.len) fail("order refers to unit {d} of {d}", .{ i, in.costs.len });
        const kept = try arena.alloc(u32, in.order.len);
        const p = try select.pack.pack(in.order, in.costs, in.budget, kept);
        try emit(out, .{ .kept = kept[0..p.kept], .cost = p.cost });
    } else if (std.mem.eql(u8, cmd, "knapsack")) {
        const in = parse(struct { values: []const f64, costs: []const u32, budget: u64, best_single: bool = true }, arena, text);
        if (in.values.len != in.costs.len) fail("values and costs differ in length", .{});
        const kept = try arena.alloc(u32, in.values.len);
        const r = try select.knapsack.greedy(arena, in.values, in.costs, in.budget, in.best_single, kept);
        try emit(out, .{ .kept = kept[0..r.kept], .cost = r.cost, .value = r.value });
    } else if (std.mem.eql(u8, cmd, "isotonic")) {
        const in = parse(struct { x: []const f64, y: []const f64, weights: ?[]const f64 = null, predict: []const f64 = &.{} }, arena, text);
        if (in.x.len == 0 or in.x.len != in.y.len) fail("x and y must be non-empty and equal in length", .{});
        if (in.weights) |w| if (w.len != in.x.len) fail("weights differ in length from x", .{});
        const m = try select.isotonic.fit(arena, in.x, in.y, in.weights);
        const predicted = try arena.alloc(f64, in.predict.len);
        for (in.predict, predicted) |x, *p| p.* = m.predict(x);
        try emit(out, .{ .starts = m.starts, .values = m.values, .predicted = predicted });
    } else if (std.mem.eql(u8, cmd, "mmr")) {
        const in = parse(struct { relevance: []const f64, similarity: []const []const f64, lambda: f64 }, arena, text);
        const n = in.relevance.len;
        const sim = try square(arena, in.similarity, n, "similarity");
        const order = try arena.alloc(u32, n);
        try select.mmr.order(in.relevance, sim, in.lambda, order, try arena.alloc(f64, n));
        try emit(out, .{ .order = order });
    } else if (std.mem.eql(u8, cmd, "submodular")) {
        const in = parse(struct {
            relevance: []const f64,
            similarity: []const []const f64,
            costs: []const u32,
            budget: u64,
            lambda: f64 = 0,
            p: f64 = 1,
        }, arena, text);
        const n = in.relevance.len;
        if (in.costs.len != n) fail("costs differ in length from relevance", .{});
        for (in.relevance) |r| if (r < 0) fail("relevance must be non-negative", .{});
        const sim = try square(arena, in.similarity, n, "similarity");
        const kept = try arena.alloc(u32, n);
        const r = try select.submodular.select(arena, in.relevance, sim, in.costs, in.budget, in.lambda, in.p, kept);
        try emit(out, .{ .kept = kept[0..r.kept], .cost = r.cost, .value = r.value });
    } else if (std.mem.eql(u8, cmd, "dpp")) {
        const in = parse(struct {
            features: ?[]const []const f64 = null,
            quality: ?[]const f64 = null,
            kernel: ?[]const []const f64 = null,
            max_items: usize,
            eps: f64 = 1e-10,
        }, arena, text);
        var n: usize = 0;
        const l = if (in.kernel) |k| blk: {
            n = k.len;
            break :blk try square(arena, k, n, "kernel");
        } else blk: {
            const f = in.features orelse fail("dpp needs features and quality, or kernel", .{});
            const q = in.quality orelse fail("dpp needs quality with features", .{});
            if (f.len != q.len) fail("features and quality differ in length", .{});
            n = q.len;
            const flat = try flatten(arena, f, null);
            const l = try arena.alloc(f64, n * n);
            try select.dpp.kernel(flat, if (n == 0) 0 else f[0].len, q, l);
            break :blk l;
        };
        const order = try arena.alloc(u32, n);
        const got = try select.dpp.greedy(arena, l, n, in.max_items, in.eps, order);
        try emit(out, .{ .order = order[0..got] });
    } else if (std.mem.eql(u8, cmd, "conformal")) {
        const in = parse(struct { losses: []const []const f64, alpha: f64, bound: f64 = 1 }, arena, text);
        if (in.losses.len == 0) fail("losses needs at least one calibration row", .{});
        const m = in.losses[0].len;
        const flat = try flatten(arena, in.losses, m);
        if (!select.conformal.isMonotone(flat, in.losses.len, m, in.bound))
            fail("each losses row must be non-increasing and within [0, bound]", .{});
        try emit(out, .{ .index = try select.conformal.calibrate(flat, in.losses.len, m, in.alpha, in.bound) });
    } else unreachable;
}
