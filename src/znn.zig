const std = @import("std");
const testing = std.testing;

const ArenaAllocator = std.heap.ArenaAllocator;

pub fn Tensor(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .Float, .Int => struct {
            const Self = @This();

            storage: []T,
            shape: []usize,
            strides: []usize,
            alloc: std.mem.Allocator,

            pub fn init(alloc: std.mem.Allocator, dims: anytype) !Self {
                const DimsType = @TypeOf(dims);
                const dims_type_info = @typeInfo(DimsType);
                if (dims_type_info != .Struct) {
                    @compileError("expected tuple dims argument, found " ++ @typeName(DimsType));
                }

                const ndim = dims_type_info.Struct.fields.len;
                const dims_fields = dims_type_info.Struct.fields;

                // Allocate memory for shape and strides
                var shape: []usize = try alloc.alloc(usize, ndim);
                errdefer alloc.free(shape);

                var strides: []usize = try alloc.alloc(usize, ndim);
                errdefer alloc.free(strides);

                // Populate shape and strides
                inline for (dims_fields, 0..) |field, i| {
                    shape[i] = @field(dims, field.name);
                }

                strides[ndim - 1] = 1;
                comptime var i: usize = ndim - 2;
                inline while (i >= 0) : (i -= 1) {
                    strides[i] = shape[i + 1] * strides[i + 1];

                    if (i == 0) {
                        break;
                    }
                }

                // Allocate storage
                var t_size: usize = 1;
                for (shape) |dim| {
                    t_size *= dim;
                }
                const storage = try alloc.alloc(T, t_size);
                errdefer alloc.free(storage);

                return Self{
                    .storage = storage,
                    .shape = shape,
                    .strides = strides,
                    .alloc = alloc,
                };
            }

            pub fn matmul(self: Self, other: Self) !Self {
                if (self.shape.len != 2 or other.shape.len != 2) {
                    // @compileError("Matrix multiplication is only defined for 2D tensors");
                    return error.IncompatibleDimensions;
                }

                if (self.shape[1] != other.shape[0]) {
                    // @compileError("Incompatible tensor shapes for matrix multiplication");
                    return error.IncompatibleShape;
                }

                const output_shape = .{ self.shape[0], other.shape[1] };
                const output_tensor = try Self.init(self.alloc, output_shape);

                var i: usize = 0;
                while (i < self.shape[0]) : (i += 1) {
                    var j: usize = 0;
                    while (j < other.shape[1]) : (j += 1) {
                        var k: usize = 0;
                        while (k < self.shape[1]) : (k += 1) {
                            output_tensor.storage[i * output_tensor.strides[0] + j] +=
                                self.storage[i * self.strides[0] + k] *
                                other.storage[k * other.strides[0] + j];
                        }
                    }
                }

                return output_tensor;
            }
        },
        else => @compileError("Unknown DType: Tensors can only hold Integers and Floats\n"),
    };
}

test "tensor matmul" {
    var arena = ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const a = try Tensor(f32).init(alloc, .{ 1024, 1024 });
    const b = try Tensor(f32).init(alloc, .{ 1024, 1024 });

    const z = try a.matmul(b);

    std.debug.print("z: {any}", .{z});
}
