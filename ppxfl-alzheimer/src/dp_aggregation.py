"""dp_aggregation.py — Server-side aggregation math for FedAvg and user-level DP-FedAvg.

These are the arithmetic steps the FL server performs once per round, pulled out
of the training loop so they can be exercised on tiny CPU tensors: weighted
FedAvg averaging, per-subject delta clipping, and the Gaussian mechanism applied
to the averaged update.

The user-level mechanism guarantees that one subject's entire contribution is
bounded by ``max_grad_norm`` before noise is added, which is what makes the
released model subject-level (not sample-level) differentially private.

Three invariants are enforced here and relied on by ``fl_server``:

1. Integer tensors are never noised (noise on an int64 buffer is meaningless).
2. BatchNorm buffers sit outside the privacy mechanism entirely — they are
   running statistics, not learned parameters, so their local values are
   averaged directly instead of being treated as clipped deltas.
3. Parameters frozen for the round receive no delta and no noise. They carry no
   subject information, so perturbing them would only add noise to the model.
"""

from collections import OrderedDict

import torch


def state_l2_norm(delta):
    """L2 norm of a state-dict-shaped update, over floating-point entries only."""
    total = 0.0
    for tensor in delta.values():
        if tensor.dtype.is_floating_point:
            total += float((tensor.float() ** 2).sum())
    return total ** 0.5


def clip_factor(norm, max_norm):
    """Scale that brings an update of size ``norm`` within ``max_norm``.

    Updates already inside the bound are left alone (factor 1.0), so clipping
    only ever shrinks a subject's influence, never amplifies it.
    """
    return min(1.0, max_norm / (norm + 1e-12))


def compute_clipped_delta(local_state, global_state, trainable_keys, max_grad_norm):
    """One subject's clipped update: ``clip(local - global, max_grad_norm)``.

    Keys outside ``trainable_keys`` are skipped: the local optimizer never
    touched them, so their delta is exactly zero.

    Returns ``(delta, factor)`` where ``delta`` is unscaled and ``factor`` is the
    clipping scale to apply when accumulating it.
    """
    delta = OrderedDict()
    for key, value in local_state.items():
        if key not in trainable_keys:
            continue
        delta[key] = value.cpu().float() - global_state[key].cpu().float()
    return delta, clip_factor(state_l2_norm(delta), max_grad_norm)


def gaussian_noise_std(sigma, max_grad_norm, num_subjects):
    """Per-coordinate std of the Gaussian mechanism on the *averaged* update.

    Each subject's update is clipped to ``max_grad_norm``, so the sensitivity of
    the sum is ``max_grad_norm``; dividing by ``num_subjects`` carries that
    through the averaging step.
    """
    if num_subjects <= 0:
        raise ValueError(f'num_subjects must be positive, got {num_subjects}')
    return sigma * max_grad_norm / num_subjects


def apply_userlevel_update(global_state, summed_delta, summed_buffers, trainable_keys,
                           buffer_keys, num_subjects, sigma, max_grad_norm, generator=None):
    """Build the next global state from one round's accumulated subject updates.

    ``summed_delta`` holds the sum of already-clipped per-subject deltas and
    ``summed_buffers`` the sum of raw local buffer values.
    """
    noise_std = gaussian_noise_std(sigma, max_grad_norm, num_subjects)
    new_state = OrderedDict()

    for key, value in global_state.items():
        if key in buffer_keys:
            # Running statistics: average the local values directly. Accumulating
            # them as unclipped, unnoised deltas let a single subject's stats blow
            # up the global model over rounds (observed as exploding loss).
            new_state[key] = ((summed_buffers[key] / num_subjects).to(value.dtype)
                              if value.dtype.is_floating_point else value.clone())
            continue
        if key not in trainable_keys:
            new_state[key] = value.clone()
            continue

        avg_delta = summed_delta[key] / num_subjects
        if value.dtype.is_floating_point:
            noise = torch.randn(avg_delta.shape, generator=generator,
                                dtype=avg_delta.dtype) * noise_std
            new_state[key] = value + avg_delta + noise
        else:
            new_state[key] = value + avg_delta.to(value.dtype)
    return new_state


def fedavg_average(client_states, weights=None):
    """Sample-count-weighted FedAvg over client state dicts.

    Integer entries (``num_batches_tracked``) are carried over from the first
    client rather than averaged, since a fractional count is not representable.
    """
    if not client_states:
        raise ValueError('fedavg_average needs at least one client state')
    if weights is None:
        weights = [1.0] * len(client_states)
    if len(weights) != len(client_states):
        raise ValueError(f'{len(weights)} weights for {len(client_states)} clients')

    total = float(sum(weights))
    if total <= 0:
        raise ValueError('client weights must sum to a positive value')
    normalised = [w / total for w in weights]

    averaged = OrderedDict()
    for key, reference in client_states[0].items():
        if not reference.dtype.is_floating_point:
            averaged[key] = reference.clone()
            continue
        acc = torch.zeros_like(reference, dtype=torch.float32)
        for weight, state in zip(normalised, client_states):
            acc += state[key].float() * weight
        averaged[key] = acc.to(reference.dtype)
    return averaged
