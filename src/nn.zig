const std = @import("std");
const value = @import("value.zig");

pub const Neuron = struct {
    weights: std.ArrayList(*value.Value),
    bias: *value.Value,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, weight_count: usize) !*Neuron {
        var neuron = try allocator.create(Neuron);
        neuron.allocator = allocator;
        neuron.weights = std.ArrayList(*value.Value).init(allocator);

        var i: usize = 0;
        while (i < weight_count) : (i += 1) {
            const weight = try value.Value.init(allocator, std.crypto.random.float(f32));
            try neuron.weights.append(weight);
        }

        neuron.bias = try value.Value.init(allocator, std.crypto.random.float(f32));

        return neuron;
    }

    pub fn deinit(self: *Neuron) void {
        self.weights.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *Neuron, inputs: *std.ArrayList(*value.Value)) !*value.Value {
        if (inputs.items.len != self.weights.items.len) {
            return error.InputSizeMismatch;
        }

        var sum = try self.weights.items[0].mul(inputs.items[0]);
        for (self.weights.items[1..], inputs.items[1..]) |weight, input| {
            const prod = try weight.mul(input);
            sum = try sum.add(prod);
        }

        return try sum.add(self.bias);
    }
};

pub const Layer = struct {
    neurons: std.ArrayList(*Neuron),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize) !*Layer {
        var layer = try allocator.create(Layer);
        layer.allocator = allocator;
        layer.neurons = std.ArrayList(*Neuron).init(allocator);

        var i: usize = 0;
        while (i < output_size) : (i += 1) {
            const neuron = try Neuron.init(allocator, input_size);
            try layer.neurons.append(neuron);
        }

        return layer;
    }

    pub fn deinit(self: *Layer) void {
        for (self.neurons.items) |neuron| {
            neuron.deinit();
        }
        self.neurons.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *Layer, inputs: *std.ArrayList(*value.Value)) !std.ArrayList(*value.Value) {
        var outputs = std.ArrayList(*value.Value).init(self.allocator);
        errdefer outputs.deinit();

        for (self.neurons.items) |neuron| {
            const output = try neuron.forward(inputs);
            try outputs.append(output);
        }

        return outputs;
    }
};

pub const MLP = struct {
    layers: std.ArrayList(*Layer),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, output_sizes: []const usize) !*MLP {
        var mlp = try allocator.create(MLP);
        mlp.allocator = allocator;
        mlp.layers = std.ArrayList(*Layer).init(allocator);

        var current_input_size = input_size;

        for (output_sizes) |output_size| {
            const layer = try Layer.init(allocator, current_input_size, output_size);
            try mlp.layers.append(layer);
            current_input_size = output_size;
        }

        return mlp;
    }

    pub fn deinit(self: *MLP) void {
        for (self.layers.items) |layer| {
            layer.deinit();
        }
        self.layers.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *MLP, inputs: *std.ArrayList(*value.Value)) !std.ArrayList(*value.Value) {
        var current_inputs = inputs.*;

        for (self.layers.items) |layer| {
            const layer_output = try layer.forward(&current_inputs);
            if (&current_inputs != inputs) current_inputs.deinit();
            current_inputs = layer_output;
        }

        return current_inputs;
    }
};

test "mlp" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const input_size = 2;
    const output_sizes = [_]usize{ 3, 1 };

    var mlp = try MLP.init(allocator, input_size, &output_sizes);
    defer mlp.deinit();

    var inputs = std.ArrayList(*value.Value).init(allocator);
    try inputs.append(try value.Value.init(allocator, 1.0));
    try inputs.append(try value.Value.init(allocator, 2.0));

    var outputs = try mlp.forward(&inputs);
    defer outputs.deinit();

    try std.testing.expectEqual(outputs.items.len, 1);
    try std.testing.expect(outputs.items[0].value != 0);

    outputs.items[0].backward();

    for (mlp.layers.items) |layer| {
        for (layer.neurons.items) |neuron| {
            for (neuron.weights.items) |weight| {
                try std.testing.expect(weight.grad != 0);
            }
            try std.testing.expect(neuron.bias.grad != 0);
        }
    }
}
