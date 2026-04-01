import pyopencl as cl
import numpy as np

# Pick Intel GPU
platform = cl.get_platforms()[0]
gpu_devices = [d for d in platform.get_devices() if d.type == cl.device_type.GPU]
device = gpu_devices[0]

ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

# Data
a = np.random.rand(1024).astype(np.float32)
b = np.random.rand(1024).astype(np.float32)
c = np.empty_like(a)

# Buffers
mf = cl.mem_flags
buf_a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
buf_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
buf_c = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

# Kernel
kernel_code = """
__kernel void add(__global const float* a,
                  __global const float* b,
                  __global float* c)
{
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}
"""

prg = cl.Program(ctx, kernel_code).build()
prg.add(queue, a.shape, None, buf_a, buf_b, buf_c)

cl.enqueue_copy(queue, c, buf_c)

print("First 5 results:", c[:5])
