#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pyopencl as cl
import time, sys
from scipy import misc
from PIL import Image

from cl_simple import CLNN_Simple
#from cl_runs import CLNN_Runs

rgb_mode = False
ycbcr_mode = False

def yuv2rgb(yuv):
  A = np.array([[1.,  0.,       1.402],
                [1., -0.34414, -0.71414],
                [1.,  1.772,    0.]])

  return np.clip(np.dot(yuv - [0, 0.5, 0.5], A.T), 0, 1)

def rgb2yuv(rgb):
  A = np.array([[0.299,     0.587,      0.114],
                [-0.168736, -0.331264,   0.5],
                [0.5,      -0.4186881, -0.081312]])

  return np.clip(np.dot(rgb, A.T) + [0, 0.5, 0.5], 0, 1)

while sys.argv[1][0] == "-":
    a = sys.argv[1]
    if a == "-rgb":
        rgb_mode = True
    elif a == "-ycbcr":
        ycbcr_mode = True
    sys.argv = sys.argv[0:1] + sys.argv[2:]

infile, outfile, modelpath = sys.argv[1:]
scale = "scale" in modelpath

ctx = cl.create_some_context()
nn = CLNN_Simple(ctx, modelpath)
#nn = CLNN_Runs(ctx, modelpath)

im = Image.open(infile)

if scale:
    im = im.resize((2*im.size[0], 2*im.size[1]), resample=Image.NEAREST)

im = np.clip(np.asarray(im).astype("float32") / 255.0, 0, 1)

channels = im.shape[2]

if channels == 4:
    im[:,:,0] *= im[:,:,3]
    im[:,:,1] *= im[:,:,3]
    im[:,:,2] *= im[:,:,3]

if not rgb_mode:
    im[:,:,0:3] = rgb2yuv(im[:,:,0:3])

if rgb_mode or ycbcr_mode:
    planes_to_do = list(range(3))
else:
    planes_to_do = [0]

if channels == 4:
    planes_to_do += [3]

for i in planes_to_do:
    plane = im[:,:,i]

    def progress(frac):
        sys.stderr.write("\r%.1f%%..." % (100 * frac))
    o_np = nn.filter_image(plane, progress)
    #o_np = in_plane
    sys.stderr.write("Done\n")
    sys.stderr.write("%d pixels/sec\n" % nn.pixels_per_second)
    sys.stderr.write("%d ops/sec\n" % nn.ops_per_second)

    plane = np.clip(np.nan_to_num(o_np), 0, 1)
    im[:,:,i] = plane

if not rgb_mode:
    im[:,:,0:3] = yuv2rgb(im[:,:,0:3])

if channels == 4:
    with np.errstate(divide='ignore', invalid='ignore'):
        im[:,:,0] /= im[:,:,3]
        im[:,:,1] /= im[:,:,3]
        im[:,:,2] /= im[:,:,3]
        im[ ~ np.isfinite(im)] = 0

im = np.clip(np.round(im * 255), 0, 255).astype("uint8")

im = Image.fromarray(im, mode=("RGBA" if channels == 4 else "RGB"))
im.save(outfile)
