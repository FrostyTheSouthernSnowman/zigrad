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
