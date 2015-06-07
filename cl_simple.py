from cl_base import *

class CLNN_Simple(CLNN_Base):
    def __init__(self, ctx, model):
        super(CLNN_Simple, self).__init__(ctx, model)
        self.bw, self.bh = 128, 128
        self.prg = cl.Program(self.ctx, """
__kernel void convolve_many(
    const unsigned int bx,
    const unsigned int by,
    const unsigned int num_inputs,
    const unsigned int num_outputs,
    global const float *in,
    global const float *k,
    global const float *bias,
    global float *out)
{
    int bs = bx*by;
    int xy = get_global_id(0) + get_global_id(1) * bx;
    int z = get_global_id(2);

    float acc = 0;
    int k_off = z * num_inputs * 9;
    for (int i = 0; i < num_inputs; i++) {
        int i_off = xy + i*bs;
        acc += in[i_off + 0*bx + 0] * k[k_off + i*9 + 0];
        acc += in[i_off + 0*bx + 1] * k[k_off + i*9 + 1];
        acc += in[i_off + 0*bx + 2] * k[k_off + i*9 + 2];
        acc += in[i_off + 1*bx + 0] * k[k_off + i*9 + 3];
        acc += in[i_off + 1*bx + 1] * k[k_off + i*9 + 4];
        acc += in[i_off + 1*bx + 2] * k[k_off + i*9 + 5];
        acc += in[i_off + 2*bx + 0] * k[k_off + i*9 + 6];
        acc += in[i_off + 2*bx + 1] * k[k_off + i*9 + 7];
        acc += in[i_off+ 2*bx + 2] * k[k_off + i*9 + 8];
    }
    acc += bias[z];
    out[xy + z*bs] = acc - 0.9 * fmin(acc, 0);
}
""").build()

    def filter_block(self, block):
        mf = cl.mem_flags
        t = time.time()
        buffers = [cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=block)]
        bh, bw = block.shape
        bs = bw * bh
        pad = len(self.steps) * 2

        for _, n_out, _, _ in self.steps:
            buf = cl.Buffer(self.ctx, 0, (4 * bs * n_out))
            buffers.append(buf)

        for i, (n_in, n_out, bias_buf, kern_buf) in enumerate(self.steps):
            self.prg.convolve_many(
                self.queue, (bw-(2*i)-2, bh-(2*i)-2, n_out), None,
                np.int32(bw), np.int32(bh), np.int32(n_in), np.int32(n_out),
                buffers[i], kern_buf, bias_buf, buffers[i+1])
            self.queue.finish()

        o_np = np.empty((bh, bw), np.float32)
        cl.enqueue_copy(self.queue, o_np, buffers[-1])

        self.total_time += time.time() - t
        self.total_pixels += (bw-pad) * (bh-pad)
        return o_np[0:bh-pad,0:bw-pad]
