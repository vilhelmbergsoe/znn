const std = @import("std");
const znn = @import("znn.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();
    const alloc = arena.allocator();

    const a = try znn.Tensor(f32).from(alloc, [_]f32{1.0}, .{ 1, 1 });
    const b = try znn.Tensor(f32).from(alloc, [_]f32{ 1.0, 2.0, 3.0, 4.0 }, .{ 1, 4 });
    // const b = try znn.Tensor(f32).init(alloc, .{ 2, 1 });

    const z = try a.matmul(b);

    // std.debug.print("{any}, {any}\n", .{ a.shape, b.shape });
    std.debug.print("{any}\n", .{z});

    // std.debug.print("a: {any}\nb: {any}\n", .{ a, b });
    // std.debug.print("a: {any}\n", .{a});
}
