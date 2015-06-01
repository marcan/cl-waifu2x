#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pyopencl as cl
import numpy as np
import time, json, sys
from scipy import misc, signal
from PIL import Image

class OpenCLNN(object):
    def __init__(self, ctx, model):
        self.ctx = ctx
        self.prg = cl.Program(ctx, """
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
        self.queue = cl.CommandQueue(self.ctx)
        self.load(model)
        self.total_time = 0
        self.total_pixels = 0

    def load(self, model_file):
        mf = cl.mem_flags
        model = json.load(open(model_file))
        self.steps = []
        self.ops_per_pixel = 0
        for i, step in enumerate(model):
            n_in, n_out = step["nInputPlane"], step["nOutputPlane"]
            self.ops_per_pixel += (n_in * n_out)
            assert step["kW"] == step["kH"] == 3
            bias_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=np.float32(step["bias"]))
            kern_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=np.float32(step["weight"]))
            self.steps.append((n_in, n_out, bias_buf, kern_buf))

    def filter_block(self, block):
        mf = cl.mem_flags
        t = time.time()
        buffers = [cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=block)]
        bh, bw = block.shape
        bs = bw * bh
        pad = len(self.steps) * 2

        for _, n_out, _, _ in self.steps:
            buf = cl.Buffer(ctx, 0, (4 * bs * n_out))
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

    def filter_image(self, im, progress_cb=None):
        pad = len(self.steps) * 2
        bw, bh = 128, 128

        dst = np.empty_like(im)
        src = np.pad(im, len(self.steps), "edge")
        h, w = im.shape

        total_pixels = bh * bw * ((h+bh-pad-1)//(bh-pad)) * ((w+bw-pad-1)//(bw-pad))
        done_pixels = 0
        for by in xrange(0, h, bh-pad):
            bh_i = min(bh, h+pad-by)
            bh_o = min(bh-pad, h-by)
            for bx in xrange(0, w, bw-pad):
                bw_i = min(bw, w+pad-bx)
                bw_o = min(bw-pad, w-bx)
                block = src[by:by+bh_i,bx:bx+bw_i]
                xblock = np.ndarray((bh, bw), np.float32)
                xblock[:bh_i,:bw_i] = block
                block = self.filter_block(xblock)
                dst[by:by+bh_o, bx:bx+bw_o] = block[:bh_o, :bw_o]
                done_pixels += bh * bw
                if progress_cb:
                    progress_cb(done_pixels / float(total_pixels))
        return dst

    @property
    def pixels_per_second(self):
        return int(self.total_pixels / self.total_time)

    @property
    def ops_per_second(self):
        return int(self.ops_per_pixel * self.total_pixels / self.total_time)

infile, outfile, modelpath = sys.argv[1:]

ctx = cl.create_some_context()
nn = OpenCLNN(ctx, modelpath)

im = Image.open(infile).convert("YCbCr")
im = misc.fromimage(im.resize((2*im.size[0], 2*im.size[1]), resample=Image.NEAREST)).astype("float32")

in_plane = np.float32(im[:,:,0] / 255.0)
def progress(frac):
    sys.stderr.write("\r%.1f%%..." % (100 * frac))
o_np = nn.filter_image(in_plane, progress)
sys.stderr.write("Done\n")
sys.stderr.write("%d pixels/sec\n" % nn.pixels_per_second)
sys.stderr.write("%d ops/sec\n" % nn.ops_per_second)

im[:,:,0] = np.clip(o_np, 0, 1) * 255
misc.toimage(im, mode="YCbCr").convert("RGB").save(outfile)

