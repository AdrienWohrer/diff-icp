'''
Handling torch tensor specifications (datatype + device (cpu vs gpu))
'''

# A Wohrer, 2023

# See an example here :
# https://www.kernel-operations.io/keops/_auto_benchmarks/plot_benchmark_convolutions.html#sphx-glr-auto-benchmarks-plot-benchmark-convolutions-py
# basically, we need to add, to every manually created torch tensor, the arguments : dtype=torchdtype, device=compdevice
# For lighter syntax, we store both attributes (dtype,device) in a dictionary that we call a "spec". Then, we simply call
# torch.tensor(...., **myspec)
# to impose both the data type and the device for that tensor

# For the dataype, for the moment we simply use float32 everywhere

import torch
import logging
logging.basicConfig(level=logging.INFO)

# Spec for tensors on Cpu
cpuspec = { "device": "cpu", "dtype": torch.float32 }

# Spec for tensors on Gpu
gpuspec = { "device": "cuda", "dtype": torch.float32 }

# Also define a default spec : Gpu if available, else Cpu
use_cuda = torch.cuda.is_available()
logging.info(f"Can use cuda : {use_cuda}")
defspec = gpuspec if use_cuda else cpuspec

# Helper function : get spec from a given set of torch tensors
# - return as a "spec" dictionary (more convenient)
# - raise error if different specs are found. (Actually we could often be more permissive regarding datatypes, but this is safer.)

def getspec(*T):
    L = [ (t.device,t.dtype) for t in T if t is not None]
    if len(set(L)) != 1:
        raise ValueError("the different input tensors to this function should be on the same device and use the same dtype !")
    return dict(zip(('device','dtype'), L[0]))