#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pyopencl as cl
import time, sys
from scipy import misc, signal
from PIL import Image

np.set_printoptions(precision=5, suppress=True)

from cl_simple import CLNN_Simple
from cl_runs import CLNN_Runs

ctx = cl.create_some_context()
model = [
    {
        "nInputPlane": 1,
        "nOutputPlane": 2,
        "kW": 3,
        "kH": 3,
        "bias": [0, 0],
        "weight": [
            [
                [[1.0,0.0,0.0],
                 [0.0,0.0,0.0],
                 [0.0,0.0,0.0]],
            ],
            [
                [[2.0,0.0,0.0],
                 [0.0,0.0,0.0],
                 [0.0,0.0,0.0]],
            ],
        ],
    },
    {
        "nInputPlane": 2,
        "nOutputPlane": 1,
        "kW": 3,
        "kH": 3,
        "bias": [100],
        "weight": [
            [
                [[1.0000000,0.0000000,0.0000000],
                 [0.0000000,0.0000000,0.0000000],
                 [0.0000000,0.0000000,0.0000000]],
                [[0.0010000,0.0000000,0.0000000],
                 [0.0000000,0.0000000,0.0000000],
                 [0.0000000,0.0000000,0.0000000]],
            ],
        ],
    },
]

nn = CLNN_Runs(ctx, model)

bw, bh = nn.bw, nn.bh
blk = np.arange(bw*bh).reshape((bh,bw)).astype(np.float32)
print blk[:10,:10]
print
fblk = nn.filter_block(blk)

print fblk[:10,:10]
