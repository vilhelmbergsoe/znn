const std = @import("std");
const testing = std.testing;

const ArenaAllocator = std.heap.ArenaAllocator;

pub fn Tensor(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .Float, .Int => struct {
            const Self = @This();

            allocator: std.mem.Allocator,

            storage: []T,
            shape: []usize,
            strides: []usize,

            // assuming grad precision to be float32
            grad: ?*Tensor(f32) = null,
            requires_grad: bool = false,

            pub fn init(allocator: std.mem.Allocator, dims: anytype) !Self {
                const DimsType = @TypeOf(dims);
                const DimsTypeInfo = @typeInfo(DimsType);
                if (DimsTypeInfo != .Struct) {
                    @compileError("Expected 'dims' to be a tuple, but found: " ++ @typeName(DimsType));
                }

                const ndim = DimsTypeInfo.Struct.fields.len;
                const dimsFields = DimsTypeInfo.Struct.fields;

                // Allocate memory for shape and strides
                var shape: []usize = try allocator.alloc(usize, ndim);
                errdefer allocator.free(shape);

                var strides: []usize = try allocator.alloc(usize, ndim);
                errdefer allocator.free(strides);

                // Populate shape and strides
                inline for (dimsFields, 0..) |field, i| {
                    shape[i] = @field(dims, field.name);
                }

                strides[ndim - 1] = 1;
                if (ndim > 1) {
                    comptime var i: usize = ndim - 2;
                    inline while (i >= 0) : (i -= 1) {
                        strides[i] = shape[i + 1] * strides[i + 1];

                        if (i == 0) {
                            break;
                        }
                    }
                }

                // Allocate storage
                var t_size: usize = 1;
                for (shape) |dim| {
                    t_size *= dim;
                }
                const storage = try allocator.alloc(T, t_size);
                errdefer allocator.free(storage);

                return Self{
                    .allocator = allocator,
                    .storage = storage,
                    .shape = shape,
                    .strides = strides,
                };
            }

            // [[1,2], [1,2]]
            pub fn from(allocator: std.mem.Allocator, data: anytype, shape: anytype) !Self {
                const DataType = @TypeOf(data);
                const DataTypeInfo = @typeInfo(DataType);

                // switch (DataTypeInfo) {
                //     .Array => {
                //         @memcpy(tensor.storage, &data);
                //     },
                //     //.Float, .Int, .ComptimeFloat, .ComptimeInt => {
                //     else => return error.InvalidData,
                // }

                const tensor = try Self.init(allocator, shape);

                switch (DataTypeInfo) {
                    .Array => {
                        @memcpy(tensor.storage, &data);
                    },
                    .Float, .Int, .ComptimeFloat, .ComptimeInt => {
                        // Scalar
                        tensor.storage[0] = data;
                        tensor.shape = .{1};
                    },
                    else => @compileError("Unsupported data type provided for 'data'. Expected array, float or int, but found: " ++ @typeName(DataType)),
                }
                return tensor;
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
                const output_tensor = try Self.init(self.allocator, output_shape);

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
        else => @compileError("Unsupported dtype: Tensors can only hold integers and floats.\n"),
    };
}

test "tensor matmul" {
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var arena = ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const a = try Tensor(f32).init(alloc, .{ 3, 3 });
    const b = try Tensor(f32).init(alloc, .{ 3, 3 });

    const z = try a.matmul(b);

    // const a = try Tensor(f32).from(alloc, .{ .{ 1, 2, 3 }, .{ 1, 2, 3 }, .{ 1, 2, 3 } });
    // const b = try Tensor(f32).init(alloc, .{ 3, 3 });
    // [_][1]f32{ [_]f32{1.0}, [_]f32{2.0} });
    // Tensor(f32).from(1.0);

    // const z = try a.matmul(b);
    std.debug.print("{any}\n", .{z});

    // std.debug.print("z: {any}", .{z});
}
