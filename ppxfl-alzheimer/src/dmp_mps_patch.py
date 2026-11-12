"""Lets deepmriprep run on the Apple GPU (MPS).

1. Its segmentation models keep two activation constants (factor, half_margin)
   as plain tensor attributes, which model.to(device) does not move; move them.
2. MPS has no kernel for a 3D convolution on very large volumes (the full-brain
   segmentation pass); any model call that hits that runs on a CPU copy of the
   same model instead, everything else stays on the GPU.
"""
import os
import torch
import deepmriprep.register as _reg
import deepmriprep.segment as _seg
from deepbet.utils import load_model as _orig

_CPU = {}


def _fix(model, device):
    for sub in model.modules():
        for name in ("factor", "half_margin"):
            if sub._c.hasattr(name):
                sub._c.setattr(name, getattr(sub, name).to(device))
    return model


class _Auto:
    def __init__(self, model, path):
        self.model, self.path, self.cpu = model, path, None

    def __call__(self, x):
        if self.cpu is None:
            try:
                return self.model(x.contiguous())
            except (RuntimeError, NotImplementedError) as e:
                if "mps" not in str(e).lower():
                    raise
                if self.path not in _CPU:
                    print(f"[mps] {os.path.basename(self.path)} -> CPU | {tuple(x.shape)} | " + ([l for l in str(e).splitlines() if "Error" in l] or [str(e)])[-1][:160], flush=True)
                    _CPU[self.path] = torch.jit.load(self.path, map_location="cpu").eval()
                self.cpu = _CPU[self.path]
        out = self.cpu(x.cpu())
        if isinstance(out, (tuple, list)):
            return type(out)(o.to(x.device) for o in out)
        return out.to(x.device)


def _load_model(path, no_gpu=False):
    model = _orig(path, no_gpu)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return model
    if device.type != "mps":
        return model
    return _Auto(_fix(model, device), path)


_seg.load_model = _load_model
_reg.load_model = _load_model
