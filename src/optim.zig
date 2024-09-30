const std = @import("std");
const value = @import("value.zig");

pub fn mse(allocator: std.mem.Allocator, preds: std.ArrayList(*value.Value), actuals: std.ArrayList(*value.Value)) !*value.Value {
    var diffs = std.ArrayList(*value.Value).init(allocator);
    defer diffs.deinit();
    for (preds.items, actuals.items) |pred, actual| {
        const diff = try pred.sub(actual);
        try diffs.append(diff);
    }

    var squared_diffs = std.ArrayList(*value.Value).init(allocator);
    defer squared_diffs.deinit();
    for (diffs.items) |diff| {
        const squared = try diff.pow(try value.Value.init(allocator, 2.0));
        try squared_diffs.append(squared);
    }

    var sums = std.ArrayList(*value.Value).init(allocator);
    defer sums.deinit();
    if (squared_diffs.items.len > 0) {
        try sums.append(squared_diffs.items[0]);
        for (squared_diffs.items[1..]) |squared_diff| {
            const sum = try sums.getLast().add(squared_diff);
            try sums.append(sum);
        }
    }

    const mean = if (sums.items.len > 0)
        try sums.getLast().div(try value.Value.init(allocator, @floatFromInt(squared_diffs.items.len)))
    else
        try value.Value.init(allocator, 0.0);

    return mean;
}

test "mse" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var preds = std.ArrayList(*value.Value).init(allocator);
    defer preds.deinit();
    try preds.append(try value.Value.init(allocator, 2.0));
    try preds.append(try value.Value.init(allocator, 3.0));

    var actuals = std.ArrayList(*value.Value).init(allocator);
    defer actuals.deinit();
    try actuals.append(try value.Value.init(allocator, 2.0));
    try actuals.append(try value.Value.init(allocator, 3.0));

    const mean = try mse(allocator, preds, actuals);
    try std.testing.expectEqual(0.0, mean.value);
}
