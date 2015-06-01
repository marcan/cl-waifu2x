## export_model.lua

You can use export_model.lua to export the original t7 models from waifu2x
to JSON format. Just place export_model.lua in the root folder of a functional
waifu2x install and then run:

$ th export_model.lua models/source_model.t7 >target_model.json

## waifu2x.py

waifu2x.py is a minimal implementation of the waifu2x algorithm. It is very
slow (single-threaded) and CPU-only, but may be useful due to its small set of
dependencies and also as a learning tool to understand the algorithm.

It currently only supports scaling, not noise reduction mode (though the latter
is trivial to add).
