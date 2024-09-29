const std = @import("std");
const constants = @import("constants.zig");

const Op = enum {
    ADD,
    SUB,
    MUL,
    POW,
    TANH,
    CONST,
};

pub const Value = struct {
    value: f32,
    grad: f32,
    propagated: bool,
    children: std.ArrayList(*Value),
    op: Op,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, value: f32) !*Self {
        const self = try allocator.create(Self);

        self.* = .{
            .value = value,
            .grad = 0.0,
            .propagated = false,
            .children = std.ArrayList(*Value).init(allocator),
            .op = Op.CONST,
            .allocator = allocator,
        };

        return self;
    }

    pub fn backward(self: *Self) void {
        self.grad = 1.0;
        self.backprop();
    }

    pub fn backprop(self: *Self) void {
        if (self.propagated) {
            return;
        }

        self.propagated = true;

        // By computing the local gradient and then the gradient of the children, we've effectively built a topological sort without actually needing to store the sorted states
        switch (self.op) {
            .ADD => {
                if (self.children.items.len >= 2) {
                    self.children.items[0].add_backward(self.children.items[1], self.grad);
                }
            },
            .SUB => {
                if (self.children.items.len >= 2) {
                    self.children.items[0].sub_backward(self.children.items[1], self.grad);
                }
            },
            .MUL => {
                if (self.children.items.len >= 2) {
                    self.children.items[0].mul_backward(self.children.items[1], self.grad);
                }
            },
            .POW => {
                if (self.children.items.len >= 2) {
                    self.children.items[0].pow_backward(self.children.items[1], self.grad);
                }
            },
            .TANH => {
                if (self.children.items.len >= 1) {
                    self.children.items[0].tanh_backward(self.value, self.grad);
                }
            },
            .CONST => {
                // do nothing
            },
        }

        for (self.children.items) |child| {
            child.backprop();
        }
    }

    pub fn add(self: *Self, other: *Self) !*Self {
        const result = try Value.init(self.allocator, self.value + other.value);

        try result.children.append(self);
        try result.children.append(other);

        result.op = Op.ADD;

        return result;
    }

    fn add_backward(self: *Self, other: *Self, out_grad: f32) void {
        self.grad += out_grad;
        other.grad += out_grad;
    }

    pub fn sub(self: *Self, other: *Self) !*Self {
        const result = try Value.init(self.allocator, self.value - other.value);

        try result.children.append(self);
        try result.children.append(other);

        result.op = Op.SUB;

        return result;
    }

    fn sub_backward(self: *Self, other: *Self, out_grad: f32) void {
        self.grad += out_grad;
        other.grad -= out_grad;
    }

    pub fn mul(self: *Self, other: *Self) !*Self {
        const result = try Value.init(self.allocator, self.value * other.value);

        try result.children.append(self);
        try result.children.append(other);

        result.op = Op.MUL;

        return result;
    }

    fn mul_backward(self: *Self, other: *Self, out_grad: f32) void {
        self.grad += other.value * out_grad;
        other.grad += self.value * out_grad;
    }

    pub fn pow(self: *Self, other: *Self) !*Self {
        const result = try Value.init(self.allocator, std.math.pow(f32, self.value, other.value));

        try result.children.append(self);
        try result.children.append(other);

        result.op = Op.POW;

        return result;
    }

    fn pow_backward(self: *Self, other: *Self, out_grad: f32) void {
        self.grad += other.value * std.math.pow(f32, self.value, other.value - 1) * out_grad;
        other.grad += std.math.pow(f32, self.value, other.value) * std.math.log(f32, constants.e, self.value) * out_grad;
    }

    pub fn tanh(self: *Self) !*Self {
        const doubled_exp = std.math.pow(f32, constants.e, (2 * self.value));

        const tanh_value = (doubled_exp - 1) / (doubled_exp + 1);

        const result = try Value.init(self.allocator, tanh_value);

        try result.children.append(self);

        result.op = Op.TANH;

        return result;
    }

    fn tanh_backward(self: *Self, tanh_value: f32, out_grad: f32) void {
        self.grad += (1 - std.math.pow(f32, tanh_value, 2)) * out_grad;
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

test "Subtraction" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const a: *Value = try Value.init(allocator, 2.0);
    const b: *Value = try Value.init(allocator, 3.0);

    const c: *Value = try a.sub(b);
    defer c.children.deinit();

    try std.testing.expectEqual(-1.0, c.value);

    c.backward();

    try std.testing.expectEqual(1.0, c.grad);
    try std.testing.expectEqual(-1.0, b.grad);
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

test "Power" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const a: *Value = try Value.init(allocator, 2.0);
    const b: *Value = try Value.init(allocator, 3.0);
    const c: *Value = try a.pow(b);
    defer c.children.deinit();
    try std.testing.expectEqual(8.0, c.value);
    c.backward();
    try std.testing.expectEqual(1.0, c.grad);
    try std.testing.expectApproxEqRel(std.math.pow(f32, 2.0, 3.0) * std.math.log(f32, constants.e, 2.0), b.grad, 0.0001);
    try std.testing.expectApproxEqRel(3.0 * std.math.pow(f32, 2.0, 2.0), a.grad, 0.0001);
}

test "Chain Rule" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const a = try Value.init(allocator, 2.0);
    const b = try Value.init(allocator, 3.0);
    const c = try Value.init(allocator, 1.0);

    const d = try a.mul(b);
    const e = try d.add(c);

    try std.testing.expectEqual(6.0, d.value);
    try std.testing.expectEqual(7.0, e.value);

    e.backward();

    try std.testing.expectEqual(1.0, e.grad);
    try std.testing.expectEqual(1.0, d.grad);
    try std.testing.expectEqual(1.0, c.grad);
    try std.testing.expectEqual(2.0, b.grad);
    try std.testing.expectEqual(3.0, a.grad);
}

test "Edge Cases" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const a = try Value.init(allocator, 2.0);

    const b = try a.add(a);

    const c = try b.mul(b);

    try std.testing.expectEqual(16.0, c.value);

    c.backward();

    try std.testing.expectEqual(1.0, c.grad);
    try std.testing.expectEqual(8.0, b.grad);
    try std.testing.expectEqual(16.0, a.grad);
}

test "Tanh" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const a = try Value.init(allocator, 0.8814);

    const b = try a.tanh();

    try std.testing.expectEqual(0.70712, b.value);

    b.backward();

    try std.testing.expectEqual(1.0, b.grad);
    try std.testing.expectApproxEqRel(0.5, a.grad, 0.0001);
}
