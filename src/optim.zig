const std = @import("std");
const value = @import("value.zig");

pub fn mse(allocator: std.mem.Allocator, preds: []const *value.Value, actuals: []const *value.Value) !*value.Value {
    const diffs = try allocator.alloc(*value.Value, preds.len);
    for (0..preds.len) |i| {
        diffs[i] = try preds[i].sub(actuals[i]);
    }

    const squared_diffs = try allocator.alloc(*value.Value, diffs.len);
    for (0..diffs.len) |i| {
        squared_diffs[i] = try diffs[i].pow(try value.Value.init(allocator, 2.0));
    }

    const sums = try allocator.alloc(*value.Value, squared_diffs.len);
    sums[0] = squared_diffs[0];
    for (1..squared_diffs.len) |i| {
        sums[i] = try sums[i - 1].add(squared_diffs[i]);
    }

    const mean = try sums[sums.len - 1].div(try value.Value.init(allocator, @floatFromInt(squared_diffs.len)));

    return mean;
}

test "mse" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const preds = try allocator.alloc(*value.Value, 2);
    preds[0] = try value.Value.init(allocator, 2.0);
    preds[1] = try value.Value.init(allocator, 3.0);
    const actuals = try allocator.alloc(*value.Value, 2);
    actuals[0] = try value.Value.init(allocator, 2.0);
    actuals[1] = try value.Value.init(allocator, 3.0);

    const mean = try mse(allocator, preds, actuals);
    try std.testing.expectEqual(0.0, mean.value);
}
