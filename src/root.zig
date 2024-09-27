const std = @import("std");

const Op = enum {
    ADD,
    MUL,
    CONST,
};

pub const Value = struct {
    value: f16,
    grad: f16,
    children: std.ArrayList(*Value),
    op: Op,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, value: f16) !*Self {
        const self = try allocator.create(Self);

        self.* = .{
            .value = value,
            .grad = 1.0,
            .children = std.ArrayList(*Value).init(allocator),
            .op = Op.CONST,
            .allocator = allocator,
        };

        return self;
    }

    pub fn backward(self: *Self) void {
        switch (self.op) {
            .ADD => {
                if (self.children.items.len >= 2) {
                    self.children.items[0].add_backward(self.children.items[1], self.grad);
                }
            },
            .MUL => {
                if (self.children.items.len >= 2) {
                    self.children.items[0].mul_backward(self.children.items[1], self.grad);
                }
            },
            .CONST => {
                // do nothing
            },
        }

        for (self.children.items) |child| {
            child.backward();
        }
    }

    pub fn add(self: *Self, other: *Self) !*Self {
        const result = try Value.init(self.allocator, self.value + other.value);

        try result.children.append(self);
        try result.children.append(other);

        result.op = Op.ADD;

        return result;
    }

    fn add_backward(self: *Self, other: *Self, out_grad: f16) void {
        self.grad = out_grad;
        other.grad = out_grad;
    }

    pub fn mul(self: *Self, other: *Self) !*Self {
        const result = try Value.init(self.allocator, self.value * other.value);

        try result.children.append(self);
        try result.children.append(other);

        result.op = Op.MUL;

        return result;
    }

    fn mul_backward(self: *Self, other: *Self, out_grad: f16) void {
        self.grad = other.value * out_grad;
        other.grad = self.value * out_grad;
    }
};

test "Addition" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const a: *Value = try Value.init(allocator, 2.0);
    const b: *Value = try Value.init(allocator, 3.0);

    const c: *Value = try a.add(b);
    defer c.children.deinit();

    try std.testing.expectEqual(5.0, c.value);

    c.backward();

    try std.testing.expectEqual(1.0, c.grad);
    try std.testing.expectEqual(1.0, b.grad);
    try std.testing.expectEqual(1.0, a.grad);
}

test "Multiplication" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const a: *Value = try Value.init(allocator, 2.0);
    const b: *Value = try Value.init(allocator, 3.0);

    const c: *Value = try a.mul(b);
    defer c.children.deinit();

    try std.testing.expectEqual(6.0, c.value);

    c.backward();

    try std.testing.expectEqual(1.0, c.grad);
    try std.testing.expectEqual(2.0, b.grad);
    try std.testing.expectEqual(3.0, a.grad);
}
