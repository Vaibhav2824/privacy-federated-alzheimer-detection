"""Tests for CPU model construction.

Weights are built with ``pretrained=False`` throughout: CI has no network access
and the architecture, not the ImageNet initialisation, is what these tests check.
"""

import pytest
import torch

import models


class TestGetResnet50:
    @pytest.fixture(scope='class')
    @classmethod
    def model(cls):
        return models.get_resnet50(num_classes=3, pretrained=False)

    def test_accepts_single_channel_mri_input(self, model):
        assert model.conv1.in_channels == 1

    def test_head_emits_one_logit_per_class(self, model):
        assert model.fc.out_features == 3
        assert model.fc.in_features == 2048

    def test_forward_pass_shape(self, model):
        with torch.no_grad():
            out = model(torch.zeros(2, 1, 224, 224))
        assert out.shape == (2, 3)

    def test_head_holds_6147_parameters(self, model):
        """The size of the user-level DP head scope, quoted in the paper."""
        assert sum(p.numel() for p in model.fc.parameters()) == 6147

    def test_all_parameters_train_by_default(self, model):
        assert all(p.requires_grad for p in model.parameters())

    def test_freeze_backbone_leaves_the_head_trainable(self):
        frozen = models.get_resnet50(num_classes=3, pretrained=False, freeze_backbone=True)
        assert all(p.requires_grad for p in frozen.fc.parameters())

    def test_freeze_backbone_freezes_most_of_the_network(self):
        frozen = models.get_resnet50(num_classes=3, pretrained=False, freeze_backbone=True)
        trainable = sum(p.numel() for p in frozen.parameters() if p.requires_grad)
        total = sum(p.numel() for p in frozen.parameters())
        assert trainable < total / 4


class TestGetVgg19:
    @pytest.fixture(scope='class')
    @classmethod
    def model(cls):
        return models.get_vgg19(num_classes=3, pretrained=False)

    def test_accepts_single_channel_mri_input(self, model):
        assert model.features[0].in_channels == 1

    def test_head_emits_one_logit_per_class(self, model):
        assert model.classifier[-1].out_features == 3

    def test_forward_pass_shape(self, model):
        with torch.no_grad():
            out = model(torch.zeros(2, 1, 224, 224))
        assert out.shape == (2, 3)

    def test_freeze_backbone_freezes_the_feature_extractor(self):
        frozen = models.get_vgg19(num_classes=3, pretrained=False, freeze_backbone=True)
        assert not any(p.requires_grad for p in frozen.features.parameters())


class TestGetModel:
    def test_dispatches_to_resnet50(self):
        assert isinstance(models.get_model('resnet50', pretrained=False), type(
            models.get_resnet50(pretrained=False)))

    def test_dispatches_to_vgg19(self):
        model = models.get_model('vgg19', pretrained=False)
        assert model.classifier[-1].out_features == 3

    def test_model_name_is_case_insensitive(self):
        assert models.get_model('ResNet50', pretrained=False).fc.out_features == 3

    def test_rejects_an_unknown_model_name(self):
        with pytest.raises(ValueError):
            models.get_model('alexnet', pretrained=False)

    def test_forwards_the_freeze_flag(self):
        frozen = models.get_model('resnet50', pretrained=False, freeze_backbone=True)
        assert not frozen.layer4[0].conv2.weight.requires_grad


class TestPretrainedAdaptation:
    """The pretrained branch reshapes a 3-channel ImageNet stem into a 1-channel
    MRI stem by averaging across the colour axis. Torchvision is stubbed so the
    reshaping logic is exercised without downloading ImageNet weights."""

    @staticmethod
    def _stub_torchvision(monkeypatch, builder, attribute):
        def fake(weights=None):
            model = builder(weights=None)
            with torch.no_grad():
                TestPretrainedAdaptation._stem(model).weight.fill_(1.0)
            return model

        monkeypatch.setattr(models.models, attribute, fake)

    @staticmethod
    def _stem(model):
        return model.conv1 if hasattr(model, 'conv1') else model.features[0]

    def test_resnet_stem_averages_the_rgb_channels(self, monkeypatch):
        import torchvision.models as tv
        self._stub_torchvision(monkeypatch, tv.resnet50, 'resnet50')
        model = models.get_resnet50(num_classes=3, pretrained=True)
        assert model.conv1.in_channels == 1
        assert torch.allclose(model.conv1.weight, torch.ones_like(model.conv1.weight))

    def test_vgg_stem_averages_the_rgb_channels(self, monkeypatch):
        import torchvision.models as tv
        self._stub_torchvision(monkeypatch, tv.vgg19, 'vgg19')
        model = models.get_vgg19(num_classes=3, pretrained=True)
        assert model.features[0].in_channels == 1
        assert torch.allclose(model.features[0].weight, torch.ones_like(model.features[0].weight))


class TestCountParameters:
    def test_reports_total_and_trainable_counts(self):
        model = models.get_resnet50(num_classes=3, pretrained=False)
        total, trainable = models.count_parameters(model)
        assert total == trainable
        assert total > 20_000_000

    def test_frozen_parameters_are_excluded_from_the_trainable_count(self):
        model = models.get_resnet50(num_classes=3, pretrained=False)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        total, trainable = models.count_parameters(model)
        assert trainable == 6147
        assert total > trainable
