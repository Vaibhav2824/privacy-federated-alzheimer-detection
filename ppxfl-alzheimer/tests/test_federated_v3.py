"""Tests for federated averaging with subject-level differential privacy."""

import numpy as np
import pytest

from src.federated_v3 import (
    FedConfig,
    calibrate_noise,
    clip_to_norm,
    compute_epsilon,
    noise_to_signal_law,
    predict_proba,
    softmax,
    summarise_trace,
    train_federated,
)


@pytest.fixture
def separable():
    """A small three-class problem a linear model can solve."""
    rng = np.random.default_rng(0)
    n_per_class = 60
    features = 12
    X, y = [], []
    for klass in range(3):
        centre = np.zeros(features)
        centre[klass * 3:(klass + 1) * 3] = 2.5
        X.append(rng.normal(centre, 1.0, size=(n_per_class, features)))
        y.append(np.full(n_per_class, klass))
    return np.vstack(X), np.concatenate(y)


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]])
    probabilities = softmax(logits)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities > 0).all()


def test_softmax_is_shift_invariant():
    logits = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(softmax(logits), softmax(logits + 100.0))


def test_clip_leaves_short_vectors_untouched():
    vector = np.array([0.3, 0.4])
    assert np.allclose(clip_to_norm(vector, 1.0), vector)


def test_clip_rescales_long_vectors_to_the_bound():
    vector = np.array([3.0, 4.0])
    clipped = clip_to_norm(vector, 1.0)
    assert np.linalg.norm(clipped) == pytest.approx(1.0)
    assert np.allclose(clipped, vector / 5.0)


def test_clip_handles_the_zero_vector():
    zero = np.zeros(4)
    assert np.allclose(clip_to_norm(zero, 1.0), zero)


def test_training_without_noise_learns(separable):
    X, y = separable
    clients = [np.arange(0, 90), np.arange(90, 180)]
    result = train_federated(X, y, clients, FedConfig(rounds=120, learning_rate=0.5))
    accuracy = float(np.mean(np.argmax(predict_proba(X, result), axis=1) == y))
    assert accuracy > 0.8
    assert result.epsilon is None
    assert result.perturbed_dimension == X.shape[1] * 3 + 3


def test_noise_free_trace_reports_zero_noise(separable):
    X, y = separable
    result = train_federated(X, y, [np.arange(len(y))], FedConfig(rounds=5))
    assert all(step.noise_norm == 0.0 for step in result.trace)
    assert summarise_trace(result.trace)["mean_noise_to_signal"] == 0.0


def test_noise_degrades_accuracy_and_reports_epsilon(separable):
    X, y = separable
    clients = [np.arange(0, 90), np.arange(90, 180)]
    config = FedConfig(rounds=40, learning_rate=0.5, noise_multiplier=5.0,
                       subject_sample_rate=0.5, seed=1)
    private = train_federated(X, y, clients, config)
    clean = train_federated(X, y, clients, FedConfig(rounds=40, learning_rate=0.5))
    private_accuracy = float(np.mean(np.argmax(predict_proba(X, private), 1) == y))
    clean_accuracy = float(np.mean(np.argmax(predict_proba(X, clean), 1) == y))
    assert private.epsilon is not None and private.epsilon > 0
    assert private_accuracy <= clean_accuracy


def test_subject_sampling_can_skip_every_subject(separable):
    """A round with no participants is skipped rather than dividing by zero."""
    X, y = separable
    config = FedConfig(rounds=3, subject_sample_rate=1e-12, seed=2)
    result = train_federated(X, y, [np.arange(len(y))], config)
    assert result.trace == []


def test_summarise_empty_trace():
    assert summarise_trace([]) == {}


def test_epsilon_decreases_with_more_noise():
    high_noise = compute_epsilon(10.0, 0.5, 50, 1e-3)
    low_noise = compute_epsilon(2.0, 0.5, 50, 1e-3)
    assert high_noise < low_noise


def test_epsilon_increases_with_more_rounds():
    assert compute_epsilon(5.0, 0.5, 100, 1e-3) > compute_epsilon(5.0, 0.5, 10, 1e-3)


def test_calibrated_noise_meets_the_target_budget():
    target = 2.0
    sigma = calibrate_noise(target, 0.5, 40, 1e-3)
    assert compute_epsilon(sigma, 0.5, 40, 1e-3) <= target + 1e-2


def test_tighter_budget_needs_more_noise():
    assert calibrate_noise(1.0, 0.5, 40, 1e-3) > calibrate_noise(10.0, 0.5, 40, 1e-3)


def test_noise_to_signal_grows_as_sqrt_of_dimension():
    """Quadrupling the dimension should double the noise-to-signal ratio."""
    small = noise_to_signal_law(1_000, 5.0, 1.0, 100)
    large = noise_to_signal_law(4_000, 5.0, 1.0, 100)
    assert large["worst_case_ratio"] / small["worst_case_ratio"] == pytest.approx(2.0)


def test_noise_to_signal_falls_with_more_participants():
    few = noise_to_signal_law(1_000, 5.0, 1.0, 50)
    many = noise_to_signal_law(1_000, 5.0, 1.0, 200)
    assert many["worst_case_ratio"] < few["worst_case_ratio"]


def test_noise_to_signal_with_no_participants_is_infinite():
    law = noise_to_signal_law(1_000, 5.0, 1.0, 0)
    assert law["worst_case_ratio"] == float("inf")


def test_dump_result_writes_json_and_creates_the_directory(tmp_path):
    from src.federated_v3 import dump_result

    path = tmp_path / "nested" / "result.json"
    dump_result(str(path), {"accuracy": 0.5})
    import json

    assert json.loads(path.read_text(encoding="utf-8")) == {"accuracy": 0.5}


def test_calibration_converges_within_the_tolerance():
    """The bisection must stop on the tolerance, not only on the iteration cap."""
    sigma = calibrate_noise(2.0, 0.5, 40, 1e-3, low=0.3, high=200.0, tolerance=1e-3)
    assert 0.3 < sigma < 200.0
    tight = calibrate_noise(2.0, 0.5, 40, 1e-3, low=0.3, high=200.0, tolerance=1e-6)
    assert sigma == pytest.approx(tight, abs=1e-2)


def test_calibration_stops_at_the_iteration_cap_when_no_tolerance_is_given():
    """With tolerance 0 the bisection never breaks early and still returns a bound."""
    sigma = calibrate_noise(2.0, 0.5, 40, 1e-3, tolerance=0.0)
    assert compute_epsilon(sigma, 0.5, 40, 1e-3) <= 2.0 + 1e-6


def test_partition_changes_the_result(separable):
    """FedAvg must depend on how subjects are grouped, or the partition study is empty."""
    X, y = separable
    config = FedConfig(rounds=25, local_epochs=3, learning_rate=0.05, seed=0)
    # The fixture stacks the classes in order, so an interleaved partition is
    # the balanced one and a contiguous one is the extreme label skew.
    balanced = [np.arange(len(y))[k::3] for k in range(3)]
    skewed = [np.flatnonzero(y == k) for k in range(3)]
    balanced_weights = train_federated(X, y, balanced, config).weights
    skewed_weights = train_federated(X, y, skewed, config).weights
    assert not np.allclose(balanced_weights, skewed_weights)


def test_local_epochs_change_the_result(separable):
    X, y = separable
    clients = [np.arange(0, 90), np.arange(90, 180)]
    one = train_federated(X, y, clients,
                          FedConfig(rounds=10, local_epochs=1, learning_rate=0.05))
    five = train_federated(X, y, clients,
                           FedConfig(rounds=10, local_epochs=5, learning_rate=0.05))
    assert not np.allclose(one.weights, five.weights)


def test_private_training_clips_every_subject_update(separable):
    """The recorded signal norm cannot exceed clip_norm times the participants."""
    X, y = separable
    clients = [np.arange(0, 90), np.arange(90, 180)]
    config = FedConfig(rounds=5, local_epochs=2, learning_rate=0.05,
                       clip_norm=0.5, noise_multiplier=1e-9,
                       subject_sample_rate=1.0, seed=3)
    result = train_federated(X, y, clients, config)
    assert result.trace
    for step in result.trace:
        assert step.signal_norm <= 0.5 * len(y) + 1e-6
