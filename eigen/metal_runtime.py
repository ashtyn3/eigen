from __future__ import annotations
import Metal
import ctypes

kernel_source = """
#include <metal_stdlib>
using namespace metal;
kernel void log_kernel(device int *in  [[ buffer(0) ]],
                       device int *out [[ buffer(1) ]],
                       uint id [[ thread_position_in_grid ]]) {
    if (id == 0) {
        out[0] = in[0] + in[1];
    }
}
"""

kernel_source_f = """
#include <metal_stdlib>
using namespace metal;

kernel void log_kernel(device float *in  [[ buffer(0) ]],
                       device float *out [[ buffer(1) ]],
                       uint id [[ thread_position_in_grid ]]) {
    out[id] = in[2 * id] + in[2 * id + 1];
}
"""


class Metal_Buffer:
    def __init__(self, dev, dtype: ctypes, a_len: int = 1):
        self.buffer = dev.newBufferWithLength_options_(
            a_len * ctypes.sizeof(ctypes.c_float),
            Metal.MTLResourceStorageModeShared,
        )
        self.t = dtype
        self.len = a_len

    def from_arr(self, values: list | ctypes.Array):
        count = len(values)
        byte_len = ctypes.sizeof(self.t) * count

        buf = self.buffer.contents().as_buffer(byte_len)
        array_view = (self.t * count).from_buffer(buf)

        array_view[:] = values

        return self

    def backward(self):
        raw = self.buffer.contents()
        mv = raw.as_buffer(ctypes.sizeof(self.t) * self.len)
        c_array = (self.t * self.len).from_buffer(mv)
        return list(c_array)


class Metal_Dim:
    thread_p_g: int
    groups: int

    def __init__(self, t: int, g: int):
        self.groups = g
        self.thread_p_g = t

    @property
    def dims(self):
        return (
            Metal.MTLSizeMake(self.thread_p_g, 1, 1),
            Metal.MTLSizeMake(self.groups, 1, 1),
        )


class Runtime:
    def __init__(self):
        self.device = Metal.MTLCreateSystemDefaultDevice()
        self.command_queue = self.device.newCommandQueue()
        self.command_buffer = self.command_queue.commandBuffer()

    def commit(self):
        self.command_buffer.commit()
        self.command_buffer.waitUntilCompleted()


class Kernel:
    inputs: list[Metal_Buffer]
    output: list[Metal_Buffer]
    name: str

    def __init__(self, r, name: str):
        self.command_buffer = r.command_buffer
        self.device = r.device
        self.name = name
        self.encoder = self.command_buffer.computeCommandEncoder()

    def load_source(self, src: str) -> Kernel:
        lib, _ = self.device.newLibraryWithSource_options_error_(
            src, None, None
        )
        self.func = lib.newFunctionWithName_("log_kernel")
        self.pipeline = self.device.newComputePipelineStateWithFunction_error_(
            self.func, None
        )[0]  # noqa: E501

        self.encoder.setComputePipelineState_(self.pipeline)
        return self

    def dims(self, size: Metal_Dim) -> Kernel:
        self.encoder.dispatchThreads_threadsPerThreadgroup_(
            size.dims[0], size.dims[1]
        )
        return self

    def set_buffers(self, *bufs: Metal_Buffer) -> Kernel:
        for [i, buf] in enumerate(*bufs):
            self.encoder.setBuffer_offset_atIndex_(buf.buffer, 0, i)
        return self

    def end(self):
        self.encoder.endEncoding()


r = Runtime()
k = Kernel(r, "log_kernel")
vals = [
    1.5,
    2.5,
    1.5,
    2.5,
    1.5,
    2.5,
    1.5,
    2.5,
    2.5,
    2.5,
]

input_buf = Metal_Buffer(r.device, ctypes.c_float, len(vals)).from_arr(vals)

output_buf = Metal_Buffer(r.device, ctypes.c_float, 5)

k.load_source(kernel_source_f).set_buffers([input_buf, output_buf]).dims(
    Metal_Dim(5, 3)
)
k.end()

r.commit()
print(output_buf.backward())
