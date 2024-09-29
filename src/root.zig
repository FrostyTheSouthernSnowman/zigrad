const std = @import("std");
pub const value = @import("value.zig");
pub const constants = @import("constants.zig");
pub const nn = @import("nn.zig");

test "all" {
    std.testing.refAllDecls(value);
    std.testing.refAllDecls(nn);
}
