"""
federated_v3.py — federated averaging with subject-level differential privacy,
parameterised by model dimension.

The mechanism is the standard one: each round every participating subject
produces one update, that update is clipped to L2 norm ``C``, the clipped
updates are summed, Gaussian noise of scale ``sigma * C`` is added to the sum,
and the result is averaged.  Clipping per subject rather than per sample is
what makes the guarantee subject-level: it bounds the influence of an entire
patient, which is the unit a hospital is accountable for, not the influence of
one MRI slice.

What this module is built to measure is the cost of that guarantee as a
function of the perturbed dimension.  The noise added to the summed update is
isotropic in ``d`` dimensions, so its expected norm grows as ``sigma * C *
sqrt(d)`` while the signal — the summed update — does not grow with ``d`` at
all.  The ratio between the two is therefore the quantity that decides whether
a private federated model is usable, and it can be reported directly rather
than inferred from an accuracy drop.  A 23.5M-parameter convolutional network
and a few-hundred-parameter model over anatomically standardised region
features sit at opposite ends of that ratio.

Privacy accounting uses Opacus' RDP accountant over the subsampled Gaussian
mechanism, composed across rounds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FedConfig:
    rounds: int = 60
    local_epochs: int = 1
    learning_rate: float = 0.5
    l2: float = 1e-3
    clip_norm: float = 1.0
    noise_multiplier: float = 0.0
    subject_sample_rate: float = 1.0
    seed: int = 42
    delta: float = 1e-3


@dataclass
class RoundTrace:
    round_index: int
    signal_norm: float
    noise_norm: float
    noise_to_signal: float


@dataclass
class FedResult:
    weights: np.ndarray
    bias: np.ndarray
    trace: list[RoundTrace] = field(default_factory=list)
    epsilon: float | None = None
    perturbed_dimension: int = 0


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _subject_gradient(x: np.ndarray, y: int, weights: np.ndarray, bias: np.ndarray,
                      n_classes: int, l2: float):
    """Gradient of the multinomial logistic loss for a single subject.

    One subject contributes exactly one scan in this cohort, so a per-subject
    gradient is well defined without having to group samples first.
    """
    logits = x @ weights + bias
    probabilities = softmax(logits[None, :])[0]
    target = np.zeros(n_classes)
    target[y] = 1.0
    error = probabilities - target
    grad_w = np.outer(x, error) + l2 * weights
    return grad_w, error


def _flatten(grad_w: np.ndarray, grad_b: np.ndarray) -> np.ndarray:
    return np.concatenate([grad_w.ravel(), grad_b.ravel()])


def _unflatten(vector: np.ndarray, shape_w, shape_b):
    size_w = int(np.prod(shape_w))
    return vector[:size_w].reshape(shape_w), vector[size_w:].reshape(shape_b)


def clip_to_norm(vector: np.ndarray, clip_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= clip_norm or norm == 0.0:
        return vector
    return vector * (clip_norm / norm)


def _local_train(X, y, indices, weights, bias, config: FedConfig, n_classes: int,
                 rng) -> np.ndarray:
    """Run local SGD from the current global model and return the update.

    The update, not the gradient, is what a federated round exchanges: a client
    takes several passes over its own subjects before reporting, which is what
    makes the client's data distribution matter to the result.  With a single
    global gradient step per round the partition cancels out entirely and
    ``natural``, ``iid`` and ``dirichlet`` produce identical numbers.
    """
    local_w = weights.copy()
    local_b = bias.copy()
    order = np.asarray(indices)
    for _ in range(config.local_epochs):
        for index in rng.permutation(order):
            grad_w, error = _subject_gradient(
                X[index], int(y[index]), local_w, local_b, n_classes, config.l2
            )
            local_w -= config.learning_rate * grad_w
            local_b -= config.learning_rate * error
    return _flatten(local_w - weights, local_b - bias)


def train_federated(X, y, clients, config: FedConfig, n_classes: int = 3) -> FedResult:
    """FedAvg over ``clients``, with optional subject-level Gaussian noise.

    Without noise this is ordinary FedAvg: each client trains locally from the
    global model and the server averages the client updates in proportion to
    client size.

    With noise the unit of both clipping and participation becomes the subject,
    because that is what the guarantee is about.  Each sampled subject trains
    locally on its own scan, its update is clipped to ``clip_norm``, the clipped
    updates are summed across all clients, Gaussian noise of scale
    ``noise_multiplier * clip_norm`` is added once at the server, and the sum is
    averaged.  With a trusted aggregator this makes the private result depend on
    which subjects took part rather than on how they were grouped, which is why
    the partition comparison is reported on the non-private arm.
    """
    rng = np.random.default_rng(config.seed)
    n_features = X.shape[1]
    weights = np.zeros((n_features, n_classes))
    bias = np.zeros(n_classes)
    dimension = weights.size + bias.size
    private = config.noise_multiplier > 0.0

    trace: list[RoundTrace] = []
    for round_index in range(config.rounds):
        summed = np.zeros(dimension)
        participants = 0

        for members in clients:
            if private:
                for index in members:
                    if config.subject_sample_rate < 1.0 and \
                            rng.random() > config.subject_sample_rate:
                        continue
                    update = _local_train(
                        X, y, [index], weights, bias, config, n_classes, rng
                    )
                    summed += clip_to_norm(update, config.clip_norm)
                    participants += 1
            else:
                selected = [
                    index for index in members
                    if config.subject_sample_rate >= 1.0
                    or rng.random() <= config.subject_sample_rate
                ]
                if not selected:
                    continue
                update = _local_train(
                    X, y, selected, weights, bias, config, n_classes, rng
                )
                # Weighted by client size, as in FedAvg.
                summed += update * len(selected)
                participants += len(selected)

        if participants == 0:
            continue

        signal_norm = float(np.linalg.norm(summed))
        if private:
            noise = rng.normal(
                0.0, config.noise_multiplier * config.clip_norm, size=dimension
            )
            noise_norm = float(np.linalg.norm(noise))
            summed = summed + noise
        else:
            noise_norm = 0.0

        step = summed / participants
        step_w, step_b = _unflatten(step, weights.shape, bias.shape)
        weights += step_w
        bias += step_b

        trace.append(RoundTrace(
            round_index=round_index,
            signal_norm=signal_norm,
            noise_norm=noise_norm,
            noise_to_signal=noise_norm / signal_norm if signal_norm > 0 else float("inf"),
        ))

    epsilon = None
    if config.noise_multiplier > 0.0:
        epsilon = compute_epsilon(
            config.noise_multiplier, config.subject_sample_rate,
            config.rounds, config.delta
        )
    return FedResult(weights=weights, bias=bias, trace=trace,
                     epsilon=epsilon, perturbed_dimension=dimension)


def predict_proba(X, result: FedResult) -> np.ndarray:
    return softmax(X @ result.weights + result.bias)


# Opacus' default alpha grid tops out at 64, and at the noise levels a
# few-hundred-subject cohort needs the optimum sits at that endpoint, which
# leaves the reported epsilon loose — and a loose bound means calibrating to
# more noise than the budget actually requires.
RDP_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 512))


def compute_epsilon(noise_multiplier: float, sample_rate: float, steps: int,
                    delta: float) -> float:
    """RDP accounting for the subsampled Gaussian mechanism over ``steps`` rounds."""
    from opacus.accountants import RDPAccountant

    accountant = RDPAccountant()
    accountant.history = [(noise_multiplier, sample_rate, steps)]
    return float(accountant.get_epsilon(delta=delta, alphas=RDP_ALPHAS))


def calibrate_noise(target_epsilon: float, sample_rate: float, steps: int,
                    delta: float, low: float = 0.3, high: float = 200.0,
                    tolerance: float = 1e-3) -> float:
    """Smallest noise multiplier whose accounted epsilon meets the target."""
    for _ in range(80):
        mid = 0.5 * (low + high)
        if compute_epsilon(mid, sample_rate, steps, delta) > target_epsilon:
            low = mid
        else:
            high = mid
        if high - low < tolerance:
            break
    return high


def noise_to_signal_law(dimension: int, noise_multiplier: float, clip_norm: float,
                        participants: int) -> dict:
    """The closed-form ratio the experiments are meant to confirm.

    The summed update has norm at most ``participants * clip_norm`` and the
    Gaussian noise vector has expected norm ``noise_multiplier * clip_norm *
    sqrt(dimension)``, so the worst-case ratio depends on the model only
    through ``sqrt(dimension)``.
    """
    expected_noise = noise_multiplier * clip_norm * np.sqrt(dimension)
    max_signal = participants * clip_norm
    return {
        "dimension": int(dimension),
        "expected_noise_norm": float(expected_noise),
        "max_signal_norm": float(max_signal),
        "worst_case_ratio": float(expected_noise / max_signal) if max_signal else float("inf"),
    }


def summarise_trace(trace: list[RoundTrace]) -> dict:
    if not trace:
        return {}
    ratios = [t.noise_to_signal for t in trace]
    return {
        "rounds": len(trace),
        "mean_noise_to_signal": float(np.mean(ratios)),
        "median_noise_to_signal": float(np.median(ratios)),
        "final_signal_norm": float(trace[-1].signal_norm),
        "final_noise_norm": float(trace[-1].noise_norm),
    }


def dump_result(path: str, payload: dict) -> None:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
