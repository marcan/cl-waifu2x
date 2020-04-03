#!/usr/bin/python
import numpy as np
import pyopencl as cl
import time, json

class CLNN_Base(object):
    FIXED_BLOCKSIZE = False
    def __init__(self, ctx, model):
        self.ctx = ctx
        self.queue = cl.CommandQueue(self.ctx)
        self.load(model)
        self.total_time = 0
        self.total_pixels = 0

    def load(self, model_file):
        mf = cl.mem_flags
        if isinstance(model_file, list):
            model = model_file
        else:
            model = json.load(open(model_file))
        self.steps = []
        self.ops_per_pixel = 0
        for i, step in enumerate(model):
            n_in, n_out = step["nInputPlane"], step["nOutputPlane"]
            self.ops_per_pixel += (n_in * n_out) * 3
            bias_buf = np.float32(step["bias"])
            weight_buf = np.float32(step["weight"])
            assert bias_buf.shape == (n_out,)
            assert weight_buf.shape == (n_out,n_in,3,3)
            assert step["kW"] == step["kH"] == 3
            bias_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=bias_buf)
            kern_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=weight_buf)
            self.steps.append((n_in, n_out, bias_buf, kern_buf))
        self.pad = len(self.steps) * 2

    def filter_image(self, im, progress_cb=None):
        pad = self.pad
        bw, bh = self.bw, self.bh

        dst = np.empty_like(im)
        src = np.pad(im, len(self.steps), "edge")
        h, w = im.shape

        total_pixels = bh * bw * ((h+bh-pad-1)//(bh-pad)) * ((w+bw-pad-1)//(bw-pad))
        done_pixels = 0
        for by in range(0, h, bh-pad):
            bh_i = min(bh, h+pad-by)
            bh_o = min(bh-pad, h-by)
            for bx in range(0, w, bw-pad):
                bw_i = min(bw, w+pad-bx)
                bw_o = min(bw-pad, w-bx)
                block = src[by:by+bh_i,bx:bx+bw_i]
                if self.FIXED_BLOCKSIZE:
                    xblock = np.ndarray((bh, bw), np.float32)
                else:
                    xblock = np.ndarray((bh_i, bw_i), np.float32)
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
