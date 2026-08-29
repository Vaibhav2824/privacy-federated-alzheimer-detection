"""Tests for the server-side FedAvg / user-level DP-FedAvg aggregation math."""

from collections import OrderedDict

import pytest
import torch

import dp_aggregation as agg


class TestStateL2Norm:
    def test_sums_squares_across_all_float_tensors(self):
        state = OrderedDict(a=torch.tensor([3.0, 4.0]), b=torch.tensor([[12.0]]))
        assert agg.state_l2_norm(state) == pytest.approx(13.0)

    def test_ignores_integer_tensors(self):
        state = OrderedDict(a=torch.tensor([3.0, 4.0]),
                            counter=torch.tensor(99, dtype=torch.long))
        assert agg.state_l2_norm(state) == pytest.approx(5.0)

    def test_empty_state_has_zero_norm(self):
        assert agg.state_l2_norm(OrderedDict()) == 0.0


class TestClipFactor:
    def test_leaves_updates_inside_the_bound_untouched(self):
        assert agg.clip_factor(0.5, 1.0) == 1.0

    def test_scales_oversized_updates_down_to_the_bound(self):
        factor = agg.clip_factor(4.0, 1.0)
        assert factor == pytest.approx(0.25)
        assert 4.0 * factor == pytest.approx(1.0)

    def test_never_amplifies_a_zero_update(self):
        assert agg.clip_factor(0.0, 1.0) == 1.0


class TestComputeClippedDelta:
    def test_delta_is_local_minus_global(self):
        global_state = OrderedDict(w=torch.zeros(2))
        local_state = OrderedDict(w=torch.tensor([0.3, 0.4]))
        delta, factor = agg.compute_clipped_delta(local_state, global_state, {'w'}, 1.0)
        assert torch.allclose(delta['w'], torch.tensor([0.3, 0.4]))
        assert factor == 1.0

    def test_frozen_keys_are_excluded_from_the_delta(self):
        global_state = OrderedDict(backbone=torch.zeros(2), fc=torch.zeros(2))
        local_state = OrderedDict(backbone=torch.tensor([9.0, 9.0]), fc=torch.tensor([0.1, 0.0]))
        delta, _ = agg.compute_clipped_delta(local_state, global_state, {'fc'}, 1.0)
        assert set(delta) == {'fc'}

    def test_oversized_update_is_clipped_to_the_bound(self):
        global_state = OrderedDict(w=torch.zeros(2))
        local_state = OrderedDict(w=torch.tensor([3.0, 4.0]))
        delta, factor = agg.compute_clipped_delta(local_state, global_state, {'w'}, 1.0)
        scaled_norm = agg.state_l2_norm(OrderedDict(w=delta['w'] * factor))
        assert scaled_norm == pytest.approx(1.0, abs=1e-6)


class TestGaussianNoiseStd:
    def test_scales_with_sigma_and_sensitivity_and_averaging(self):
        assert agg.gaussian_noise_std(2.0, 1.0, 4) == pytest.approx(0.5)

    def test_more_subjects_means_less_noise(self):
        assert agg.gaussian_noise_std(1.0, 1.0, 100) < agg.gaussian_noise_std(1.0, 1.0, 10)

    def test_rejects_non_positive_subject_counts(self):
        with pytest.raises(ValueError, match='must be positive'):
            agg.gaussian_noise_std(1.0, 1.0, 0)


class TestApplyUserlevelUpdate:
    def _round_inputs(self, simple_state):
        summed_delta = OrderedDict(weight=torch.full((2, 2), 2.0), bias=torch.full((2,), 2.0))
        summed_buffers = OrderedDict(running_mean=torch.full((2,), 8.0))
        return summed_delta, summed_buffers

    def test_trainable_params_move_by_the_averaged_delta(self, simple_state):
        summed_delta, summed_buffers = self._round_inputs(simple_state)
        new_state = agg.apply_userlevel_update(
            simple_state, summed_delta, summed_buffers,
            trainable_keys={'weight', 'bias'}, buffer_keys={'running_mean'},
            num_subjects=2, sigma=0.0, max_grad_norm=1.0)
        assert torch.allclose(new_state['weight'], torch.full((2, 2), 1.0))
        assert torch.allclose(new_state['bias'], torch.full((2,), 1.0))

    def test_buffers_are_averaged_not_treated_as_deltas(self, simple_state):
        summed_delta, summed_buffers = self._round_inputs(simple_state)
        new_state = agg.apply_userlevel_update(
            simple_state, summed_delta, summed_buffers,
            trainable_keys={'weight', 'bias'}, buffer_keys={'running_mean'},
            num_subjects=2, sigma=0.0, max_grad_norm=1.0)
        assert torch.allclose(new_state['running_mean'], torch.full((2,), 4.0))

    def test_frozen_params_are_copied_through_unperturbed(self, simple_state):
        summed_delta, summed_buffers = self._round_inputs(simple_state)
        new_state = agg.apply_userlevel_update(
            simple_state, summed_delta, summed_buffers,
            trainable_keys={'bias'}, buffer_keys={'running_mean'},
            num_subjects=2, sigma=10.0, max_grad_norm=1.0)
        assert torch.equal(new_state['weight'], simple_state['weight'])

    def test_integer_counters_are_never_noised(self, simple_state):
        summed_delta, summed_buffers = self._round_inputs(simple_state)
        summed_delta['num_batches_tracked'] = torch.tensor(4.0)
        new_state = agg.apply_userlevel_update(
            simple_state, summed_delta, summed_buffers,
            trainable_keys={'weight', 'bias', 'num_batches_tracked'},
            buffer_keys={'running_mean'},
            num_subjects=2, sigma=10.0, max_grad_norm=1.0)
        assert new_state['num_batches_tracked'].dtype == torch.long
        assert int(new_state['num_batches_tracked']) == 2

    def test_integer_buffers_are_carried_over_unchanged(self):
        global_state = OrderedDict(counter=torch.tensor(7, dtype=torch.long))
        new_state = agg.apply_userlevel_update(
            global_state, OrderedDict(), OrderedDict(),
            trainable_keys=set(), buffer_keys={'counter'},
            num_subjects=2, sigma=1.0, max_grad_norm=1.0)
        assert int(new_state['counter']) == 7

    def test_noise_is_added_when_sigma_is_positive(self, simple_state):
        summed_delta, summed_buffers = self._round_inputs(simple_state)
        generator = torch.Generator().manual_seed(0)
        new_state = agg.apply_userlevel_update(
            simple_state, summed_delta, summed_buffers,
            trainable_keys={'weight', 'bias'}, buffer_keys={'running_mean'},
            num_subjects=2, sigma=5.0, max_grad_norm=1.0, generator=generator)
        assert not torch.allclose(new_state['weight'], torch.full((2, 2), 1.0))

    def test_noise_is_reproducible_for_a_fixed_generator_seed(self, simple_state):
        summed_delta, summed_buffers = self._round_inputs(simple_state)
        states = []
        for _ in range(2):
            states.append(agg.apply_userlevel_update(
                simple_state, summed_delta, summed_buffers,
                trainable_keys={'weight', 'bias'}, buffer_keys={'running_mean'},
                num_subjects=2, sigma=5.0, max_grad_norm=1.0,
                generator=torch.Generator().manual_seed(1234)))
        assert torch.equal(states[0]['weight'], states[1]['weight'])

    def test_head_scope_perturbs_far_less_than_full_scope(self):
        """The reason head scope exists: noise energy scales with perturbed dimension."""
        global_state = OrderedDict(backbone=torch.zeros(64, 64), fc=torch.zeros(8))
        summed_delta = OrderedDict(backbone=torch.zeros(64, 64), fc=torch.zeros(8))

        def total_noise(trainable_keys):
            new_state = agg.apply_userlevel_update(
                global_state, summed_delta, OrderedDict(),
                trainable_keys=trainable_keys, buffer_keys=set(),
                num_subjects=1, sigma=1.0, max_grad_norm=1.0,
                generator=torch.Generator().manual_seed(7))
            return sum(float((v ** 2).sum()) for v in new_state.values())

        assert total_noise({'fc'}) < total_noise({'backbone', 'fc'}) / 100


class TestFedavgAverage:
    def test_uniform_average_of_two_clients(self):
        states = [OrderedDict(w=torch.zeros(2)), OrderedDict(w=torch.full((2,), 4.0))]
        averaged = agg.fedavg_average(states)
        assert torch.allclose(averaged['w'], torch.full((2,), 2.0))

    def test_weights_by_client_sample_count(self):
        states = [OrderedDict(w=torch.zeros(1)), OrderedDict(w=torch.ones(1))]
        averaged = agg.fedavg_average(states, weights=[3, 1])
        assert averaged['w'].item() == pytest.approx(0.25)

    def test_integer_entries_are_taken_from_the_first_client(self):
        states = [OrderedDict(n=torch.tensor(2, dtype=torch.long)),
                  OrderedDict(n=torch.tensor(8, dtype=torch.long))]
        averaged = agg.fedavg_average(states)
        assert int(averaged['n']) == 2

    def test_preserves_the_reference_dtype(self):
        states = [OrderedDict(w=torch.zeros(2, dtype=torch.float64)),
                  OrderedDict(w=torch.ones(2, dtype=torch.float64))]
        assert agg.fedavg_average(states)['w'].dtype == torch.float64

    def test_rejects_an_empty_client_list(self):
        with pytest.raises(ValueError, match='at least one client'):
            agg.fedavg_average([])

    def test_rejects_a_weight_count_mismatch(self):
        with pytest.raises(ValueError, match='weights for'):
            agg.fedavg_average([OrderedDict(w=torch.zeros(1))], weights=[1, 2])

    def test_rejects_weights_that_sum_to_zero(self):
        states = [OrderedDict(w=torch.zeros(1)), OrderedDict(w=torch.ones(1))]
        with pytest.raises(ValueError, match='positive value'):
            agg.fedavg_average(states, weights=[0, 0])
