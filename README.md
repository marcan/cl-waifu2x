# cl-waifu2x

**WARNING: This software is under active development and not yet inteded for
production or use by end-users. It is not yet optimized. Tread with caution.**

cl-waifu2x is an implementation of the waifu2x algorithm in OpenCL. It focuses
on use of the neural network algorithm, not its training, and therefore it
relies on models produced with the original waifu2x.

cl-waifu2x aims to be compatible with most mainstream OpenCL implementations,
including GPU-based and CPU-based ones from major vendors.

Based on [waifu2x by nagadomi](https://github.com/nagadomi/waifu2x).

## Dependencies

* Python 2.7
* numpy
* scipy
* PIL (or Pillow)
* PyOpenCL

And an OpenCL implementation.

## Usage

    $ python2 cl-waifu2x.py miku_small.png miku_small_cl.png models/scale2.0x_model.json
    Choose platform:
    [0] <pyopencl.Platform 'Intel(R) OpenCL' at 0x7fa4d7f10110>
    [1] <pyopencl.Platform 'NVIDIA CUDA' at 0x7fa4d8026f80>
    Choice [0]:
    Set the environment variable PYOPENCL_CTX='' to avoid being asked again.
    100.0%...Done
    29004 pixels/sec
    925359122 ops/sec

OpenCL implementations that are being used for testing:
* Intel (CPU) (test platform: Intel Core i7 3820QM)
* Nvidia (GPU) (test platform: Nvidia GeForce GTX 660M)

## Performance

The current kernel is very dumb and not yet GPU-optimized. Performance is
currently about equal on CPU and GPU, and about 6 times slower than the original
waifu2x CUDA version on the same GPU, though also several times faster than
the trivial
[single-threaded waifu2x.py](tools/waifu2x.py)
on the same CPU.
