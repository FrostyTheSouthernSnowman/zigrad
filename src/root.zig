const std = @import("std");
pub const value = @import("value.zig");
pub const constants = @import("constants.zig");
pub const nn = @import("nn.zig");
pub const optim = @import("optim.zig");

test "all" {
    std.testing.refAllDecls(value);
    std.testing.refAllDecls(nn);
    std.testing.refAllDecls(optim);
}

test "full neural network" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var xs = std.ArrayList(*value.Value).init(allocator);
    defer xs.deinit();
    try xs.append(try value.Value.init(allocator, 0.0));
    try xs.append(try value.Value.init(allocator, 2.0));
    try xs.append(try value.Value.init(allocator, 3.0));

    var ys = std.ArrayList(*value.Value).init(allocator);
    defer ys.deinit();
    try ys.append(try value.Value.init(allocator, 0.0));
    try ys.append(try value.Value.init(allocator, 4.0));
    try ys.append(try value.Value.init(allocator, 9.0));

    const model = try nn.MLP.init(allocator, 1, &.{ 1, 1 });

    const epochs = 10;

    for (0..epochs) |epoch| {
        var avg_loss: f32 = 0.0;
        for (xs.items, ys.items) |x, y| {
            var input = std.ArrayList(*value.Value).init(allocator);
            defer input.deinit();
            try input.append(x);
            const pred = try model.forward(&input);
            defer pred.deinit();
            var actual = std.ArrayList(*value.Value).init(allocator);
            defer actual.deinit();
            try actual.append(y);
            const loss = try optim.mse(allocator, pred, actual);
            loss.backward();
            model.update(0.1);

            avg_loss += loss.value;
        }

        avg_loss /= @floatFromInt(xs.items.len);

        std.debug.print("Epoch {d}: {d}\n", .{ epoch, avg_loss });
    }
}
