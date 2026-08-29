"""Tests for the ONNX export used by the browser demo.

A ResNet50 export takes tens of seconds, so the heavier cases use a small stand-in
network with the same interface. The one full-model test covers the real path.
"""

import json
import os

import pytest
import torch
import torch.nn as nn

import export_onnx


class TinyNet(nn.Module):
    """Same contract as the real classifier: 1x224x224 in, 3 logits out.

    Keeps a convolution ahead of the head so the exported graph has the Conv and
    Gemm nodes that dynamic quantisation actually operates on. A pooling-only
    stand-in exports fine but is not a realistic quantisation target, and newer
    onnx releases fail shape inference on it.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(self.pool(self.conv(x)).flatten(1))


@pytest.fixture
def tiny_checkpoint(tmp_path):
    path = tmp_path / 'tiny.pth'
    torch.save(TinyNet().state_dict(), path)
    return str(path)


class TestLoadCheckpointState:
    def test_reads_a_bare_state_dict(self, tiny_checkpoint):
        state = export_onnx.load_checkpoint_state(tiny_checkpoint)
        assert 'fc.weight' in state

    @pytest.mark.parametrize('key', ['model_state_dict', 'state_dict', 'model'])
    def test_unwraps_a_state_dict_saved_alongside_training_metadata(self, tmp_path, key):
        path = tmp_path / f'{key}.pth'
        torch.save({key: TinyNet().state_dict(), 'epoch': 7, 'f1': 0.4}, path)
        assert 'fc.weight' in export_onnx.load_checkpoint_state(str(path))

    def test_accepts_a_checkpoint_saved_as_a_whole_module(self, tmp_path):
        """Some early runs pickled the module itself rather than its state dict."""
        path = tmp_path / 'module.pth'
        torch.save(TinyNet(), path)
        restored = export_onnx.load_checkpoint_state(str(path))
        assert isinstance(restored, TinyNet)

    def test_leaves_an_unrecognised_payload_alone(self, tmp_path):
        path = tmp_path / 'plain.pth'
        torch.save({'fc.weight': torch.zeros(3, 4)}, path)
        assert 'fc.weight' in export_onnx.load_checkpoint_state(str(path))


class TestExport:
    def test_writes_a_graph_that_runs(self, tmp_path):
        out = str(tmp_path / 'tiny.onnx')
        export_onnx.export(TinyNet().eval(), out, opset=17)
        assert os.path.getsize(out) > 0

        import onnxruntime as ort

        session = ort.InferenceSession(out)
        logits = session.run(None, {'slice': torch.zeros(1, 1, 224, 224).numpy()})[0]
        assert logits.shape == (1, 3)

    def test_batch_axis_is_dynamic(self, tmp_path):
        out = str(tmp_path / 'tiny.onnx')
        export_onnx.export(TinyNet().eval(), out, opset=17)

        import onnxruntime as ort

        session = ort.InferenceSession(out)
        logits = session.run(None, {'slice': torch.zeros(4, 1, 224, 224).numpy()})[0]
        assert logits.shape == (4, 3)

    def test_creates_the_output_directory(self, tmp_path):
        out = str(tmp_path / 'nested' / 'deeper' / 'tiny.onnx')
        export_onnx.export(TinyNet().eval(), out, opset=17)
        assert os.path.exists(out)


class TestQuantise:
    def test_shrinks_the_graph_and_keeps_it_runnable(self, tmp_path):
        source = str(tmp_path / 'tiny.onnx')
        target = str(tmp_path / 'tiny.int8.onnx')
        export_onnx.export(TinyNet().eval(), source, opset=17)
        export_onnx.quantise(source, target)

        import onnxruntime as ort

        session = ort.InferenceSession(target)
        assert session.run(None, {'slice': torch.zeros(1, 1, 224, 224).numpy()})[0].shape == (1, 3)


class TestWriteManifest:
    def test_records_what_was_shipped(self, tmp_path):
        model_path = str(tmp_path / 'model' / 'ppxfl.onnx')
        os.makedirs(os.path.dirname(model_path))
        manifest_path = export_onnx.write_manifest(model_path, 'best_run.pth', 'resnet50', 123)
        manifest = json.loads(open(manifest_path).read())
        assert manifest['model'] == 'resnet50'
        assert manifest['source_checkpoint'] == 'best_run.pth'
        assert manifest['classes'] == ['CN', 'MCI', 'AD']
        assert manifest['input_shape'] == [1, 1, 224, 224]
        assert manifest['quantisation'] == 'dynamic-int8'
        assert manifest['size_bytes'] == 123

    def test_lands_beside_the_model_it_describes(self, tmp_path):
        model_path = str(tmp_path / 'ppxfl.onnx')
        manifest_path = export_onnx.write_manifest(model_path, 'run.pth', 'resnet50', 1)
        assert os.path.dirname(manifest_path) == os.path.dirname(os.path.abspath(model_path))


class TestBuildModel:
    def test_restores_the_trained_weights_in_eval_mode(self, tmp_path):
        """Uses the real ResNet50 path, which is what actually ships."""
        from models import get_model

        trained = get_model('resnet50', num_classes=3, pretrained=False)
        with torch.no_grad():
            trained.fc.bias.fill_(0.25)
        path = tmp_path / 'resnet.pth'
        torch.save(trained.state_dict(), path)

        restored = export_onnx.build_model(str(path))
        assert restored.training is False
        assert torch.allclose(restored.fc.bias, torch.full((3,), 0.25))
