"""export_onnx.py — Export a trained checkpoint to quantised ONNX for the web demo.

The browser demo runs the classifier client-side with onnxruntime-web, so no MRI
slice a visitor drops on the page ever leaves their machine. A float32 ResNet50
is ~95MB, past GitHub's comfortable file size; dynamic int8 quantisation of the
Linear/Conv weights brings it to roughly a quarter of that with a small accuracy
cost, which is acceptable for a demonstration (the reported results all come
from the float model).

Usage:
    python -m src.export_onnx --checkpoint results_v2/checkpoints/best_....pth \
        --out ui/public/model/ppxfl_resnet50.onnx
"""

import argparse
import inspect
import json
import os

import torch

from models import get_model

#: Input the exported graph is traced with: one 224x224 single-channel slice.
EXAMPLE_SHAPE = (1, 1, 224, 224)
CLASS_NAMES = ('CN', 'MCI', 'AD')

#: Keys a checkpoint may nest its state dict under, alongside training metadata.
CHECKPOINT_STATE_KEYS = ('model_state_dict', 'state_dict', 'model')


def load_checkpoint_state(checkpoint_path, map_location='cpu'):
    """Return the model state dict from a training checkpoint.

    Checkpoints are saved either as a bare state dict or wrapped alongside the
    epoch and metrics, so accept both shapes.
    """
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        return payload
    nested = [payload[key] for key in CHECKPOINT_STATE_KEYS if isinstance(payload.get(key), dict)]
    return nested[0] if nested else payload


def build_model(checkpoint_path, model_name='resnet50'):
    """Reconstruct the trained network on CPU in eval mode."""
    model = get_model(model_name, num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(load_checkpoint_state(checkpoint_path))
    model.eval()
    return model


def export(model, out_path, opset=17):
    """Trace the model to ONNX with a dynamic batch axis.

    Pinned to the TorchScript exporter. torch 2.6 and later default to
    ``dynamo=True``, which warns that ``dynamic_axes`` is not recommended on that
    path and emits a graph that onnx's version converter cannot then quantise
    (``No initializer or constant input to node found``). Pinning also keeps the
    exported graph identical across torch versions, so the shipped model matches
    the one whose numerics were checked against the float network.

    The keyword did not exist before torch 2.5, so its absence is tolerated:
    those versions only have the TorchScript exporter anyway.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    kwargs = {
        'input_names': ['slice'],
        'output_names': ['logits'],
        'dynamic_axes': {'slice': {0: 'batch'}, 'logits': {0: 'batch'}},
        'opset_version': opset,
    }
    example = torch.zeros(*EXAMPLE_SHAPE)
    if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
        kwargs['dynamo'] = False
    torch.onnx.export(model, example, out_path, **kwargs)
    return out_path


def quantise(src_path, out_path):
    """Dynamic int8 quantisation of the exported graph's weights."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(src_path, out_path, weight_type=QuantType.QInt8)
    return out_path


def write_manifest(out_path, checkpoint_path, model_name, size_bytes):
    """Record what the shipped model is, so the UI can label it honestly."""
    manifest = {
        'model': model_name,
        'source_checkpoint': os.path.basename(checkpoint_path),
        'classes': list(CLASS_NAMES),
        'input_shape': list(EXAMPLE_SHAPE),
        'quantisation': 'dynamic-int8',
        'size_bytes': size_bytes,
    }
    manifest_path = os.path.join(os.path.dirname(out_path), 'model_manifest.json')
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def main():  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description='Export a PPXFL checkpoint to quantised ONNX')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model', default='resnet50')
    parser.add_argument('--out', default=os.path.join('..', 'ui', 'public', 'model',
                                                      'ppxfl_resnet50.onnx'))
    parser.add_argument('--no-quantise', action='store_true')
    args = parser.parse_args()

    model = build_model(args.checkpoint, args.model)
    float_path = args.out.replace('.onnx', '.fp32.onnx')
    export(model, float_path)
    print(f"exported float32: {float_path} ({os.path.getsize(float_path) / 1e6:.1f} MB)")

    if args.no_quantise:
        os.replace(float_path, args.out)
    else:
        quantise(float_path, args.out)
        os.remove(float_path)
    size = os.path.getsize(args.out)
    print(f"wrote {args.out} ({size / 1e6:.1f} MB)")
    print(f"manifest: {write_manifest(args.out, args.checkpoint, args.model, size)}")


if __name__ == '__main__':  # pragma: no cover
    main()
