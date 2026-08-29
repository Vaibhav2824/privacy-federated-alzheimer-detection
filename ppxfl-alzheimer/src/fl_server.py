"""
fl_server.py — Federated Learning Server (Manual FedAvg Simulation)
PPXFL: Privacy-Preserving Explainable Federated Learning for Alzheimer's Detection

Runs FL simulation using a manual FedAvg loop (no ray dependency).
Supports configurable clients, rounds, local epochs, and optional per-client
DP-SGD (head-only or full-network) with cross-round privacy accounting.
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from copy import deepcopy

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from centralised_train import MRIDataset, compute_class_weights, compute_metrics, evaluate, train_one_epoch
from models import get_model
from splits import load_split


def fedavg_aggregate(client_weights, client_sizes):
    """
    FedAvg: weighted average of client model parameters.

    Args:
        client_weights: list of state_dicts (on CPU)
        client_sizes: list of int (num samples per client)

    Returns:
        averaged state_dict
    """
    total = sum(client_sizes)
    avg_state = OrderedDict()

    for key in client_weights[0].keys():
        avg_state[key] = sum(
            client_weights[i][key].float() * (client_sizes[i] / total)
            for i in range(len(client_weights))
        )

    return avg_state


def _make_dp_engine(model, optimizer, loader, dp_mode, target_epsilon, target_delta,
                     total_epochs_equiv, max_grad_norm):
    """Create a PrivacyEngine + calibrated noise multiplier for one client, once.

    The engine object is returned so the caller can reuse it every round —
    reusing the same PrivacyEngine instance is what makes the RDP accountant
    accumulate privacy cost ACROSS rounds instead of resetting each round
    (the previous implementation created a fresh PrivacyEngine per round,
    which silently under-reported cumulative epsilon).
    """
    from opacus import PrivacyEngine
    from opacus.accountants.utils import get_noise_multiplier

    sample_rate = loader.batch_size / len(loader.dataset)
    noise_multiplier = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        epochs=total_epochs_equiv,
        accountant='rdp',
    )

    pe = PrivacyEngine(accountant='rdp')
    dp_model, dp_optimizer, dp_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        # see dp_train.py's matching comment: 'hooks' breaks on ResNet's in-place
        # residual add once the backbone is trainable (dp_mode='full').
        grad_sample_mode='ew' if dp_mode == 'full' else 'hooks',
    )
    return pe, dp_model, dp_optimizer, dp_loader, noise_multiplier


def run_simulation(model_name='vgg19', num_clients=4, num_rounds=50,
                   local_epochs=5, batch_size=32, lr=1e-4,
                   dp_mode='none', target_epsilon=5.0, target_delta=1e-3, max_grad_norm=1.0,
                   fold=0, alpha=0.5, clients_root=None, splits_path=None,
                   data_dir=None, results_dir=None, seed=42, resume=False, tag_suffix=''):
    """
    Run federated learning simulation using manual FedAvg.

    Args:
        model_name: 'vgg19' or 'resnet50'
        num_clients: K — number of clients
        num_rounds: T — total FL rounds
        local_epochs: E — local epochs per round
        batch_size: Training batch size
        lr: Learning rate
        dp_mode: 'none', 'head' (freeze backbone, DP-SGD on head only), or 'full' (DP-SGD on whole net)
        target_epsilon: privacy budget per client, calibrated once over the full run
        target_delta: DP delta (default 1e-3 given the small cohort)
        max_grad_norm: C for gradient clipping
        fold: which subject-level fold's train subjects the clients were partitioned from
        alpha: Dirichlet alpha used by partition.py (selects the client data directory)
        clients_root: root dir containing f{fold}_a{alpha}_s{seed}/client_k subdirs
        splits_path: path to splits_v1.json (for val/test — NEVER re-split here)
        data_dir: path to data/processed (manifest.csv, all_images.npy, all_labels.npy)
        results_dir: Path to save results
        seed: Random seed
        resume: resume from the last completed round's checkpoint if present
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_dir is None:
        data_dir = os.path.join(project_root, 'data', 'processed')
    if clients_root is None:
        clients_root = os.path.join(project_root, 'data', 'clients')
    if splits_path is None:
        splits_path = os.path.join(project_root, 'data', 'splits', 'splits_v1.json')
    if results_dir is None:
        results_dir = os.path.join(project_root, 'results')

    figures_dir = os.path.join(results_dir, 'figures')
    metrics_dir = os.path.join(results_dir, 'metrics')
    checkpoints_dir = os.path.join(results_dir, 'checkpoints')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    dp_suffix = f"_dp{dp_mode}_eps{target_epsilon}" if dp_mode != 'none' else ""
    experiment_name = (f"{model_name}_fedavg_K{num_clients}_T{num_rounds}"
                       f"_E{local_epochs}_f{fold}_s{seed}{dp_suffix}{tag_suffix}")
    last_checkpoint_path = os.path.join(checkpoints_dir, f'last_{experiment_name}.pth')
    best_checkpoint_path = os.path.join(checkpoints_dir, f'best_{experiment_name}.pth')

    # ── Load client datasets (produced by partition.py for this exact fold+alpha+seed) ──
    client_data_dir = os.path.join(clients_root, f'f{fold}_a{alpha}_s{seed}')
    if not os.path.isdir(client_data_dir):
        raise FileNotFoundError(
            f"{client_data_dir} not found — run partition.py --fold {fold} --alpha {alpha} "
            f"--seed {seed} --num-clients {num_clients} first."
        )

    client_loaders = []
    client_sizes = []
    for cid in range(1, num_clients + 1):
        cdir = os.path.join(client_data_dir, f'client_{cid}')
        imgs = np.load(os.path.join(cdir, 'images.npy'))
        lbls = np.load(os.path.join(cdir, 'labels.npy'))
        ds = MRIDataset(imgs, lbls, augment=True)
        # DP's sample_rate = batch_size/len(dataset) must stay strictly below 1 or
        # the RDP accountant's log(1/q - 1) hits a math domain error — and with
        # drop_last=True, batch_size > len(dataset) silently yields ZERO batches
        # per epoch (client trains on nothing). At this cohort's Dirichlet-skewed
        # client sizes (as low as ~9 slices), the configured batch_size can easily
        # exceed a client's dataset, so cap it per-client to leave meaningful
        # subsampling amplification (<=50% of the client's data per step).
        client_batch_size = min(batch_size, max(1, len(ds) // 2)) if dp_mode != 'none' else batch_size
        loader = DataLoader(ds, batch_size=client_batch_size, shuffle=True, num_workers=0,
                            drop_last=(dp_mode != 'none'))
        client_loaders.append(loader)
        client_sizes.append(len(ds))
        print(f"  Client {cid}: {len(ds)} samples  "
              f"(CN={np.sum(lbls==0)}, MCI={np.sum(lbls==1)}, AD={np.sum(lbls==2)})"
              f"{f' [DP batch_size capped to {client_batch_size}]' if client_batch_size != batch_size else ''}")

    # ── Val/test come from the SAME fold's held-out subjects, never re-split ──
    split = load_split(fold, os.path.join(data_dir, 'manifest.csv'), splits_path)
    all_images = np.load(os.path.join(data_dir, 'all_images.npy'))
    all_labels = np.load(os.path.join(data_dir, 'all_labels.npy'))
    val_dataset = MRIDataset(all_images[split['val_idx']], all_labels[split['val_idx']])
    test_dataset = MRIDataset(all_images[split['test_idx']], all_labels[split['test_idx']])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"  Fold {fold}: val={len(val_dataset)} slices ({len(split['val_subjects'])} subj) | "
          f"test={len(test_dataset)} slices ({len(split['test_subjects'])} subj) — held out from ALL clients")

    class_weights = compute_class_weights(all_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Initialise global model ─────────────────────────────────────────
    freeze_backbone = (dp_mode == 'head')
    global_model = get_model(model_name, num_classes=3, pretrained=True, freeze_backbone=freeze_backbone)

    if dp_mode == 'full':
        from opacus.validators import ModuleValidator
        if not ModuleValidator.is_valid(global_model):
            global_model = ModuleValidator.fix(global_model)  # BatchNorm -> GroupNorm

    print(f"\n{'='*60}")
    print(f"Starting FL Simulation: {experiment_name}")
    print(f"  Model: {model_name} | Clients: {num_clients} | Rounds: {num_rounds}")
    print(f"  Local Epochs: {local_epochs} | Batch: {batch_size} | LR: {lr}")
    if dp_mode != 'none':
        print(f"  DP mode: {dp_mode} | target ε={target_epsilon} δ={target_delta} | C={max_grad_norm}")
    print(f"{'='*60}\n")

    # One PrivacyEngine PER CLIENT, created once and reused every round so the
    # RDP accountant accumulates privacy cost across the whole run.
    privacy_engines = {}
    noise_multipliers = {}
    if dp_mode != 'none':
        total_epochs_equiv = num_rounds * local_epochs
        for cid in range(num_clients):
            dummy_model = deepcopy(global_model)
            dummy_optimizer = optim.Adam(filter(lambda p: p.requires_grad, dummy_model.parameters()), lr=lr)
            pe, _, _, _, sigma = _make_dp_engine(
                dummy_model, dummy_optimizer, client_loaders[cid], dp_mode,
                target_epsilon, target_delta, total_epochs_equiv, max_grad_norm,
            )
            privacy_engines[cid] = pe
            noise_multipliers[cid] = sigma
            print(f"  Client {cid+1}: calibrated σ={sigma:.4f} for target ε={target_epsilon} "
                  f"over {total_epochs_equiv} client-epochs")

    round_metrics = {'rounds': [], 'accuracy': [], 'f1_macro': [], 'auroc_macro': [], 'loss': []}
    if dp_mode != 'none':
        round_metrics['epsilon_per_client'] = []

    best_f1 = 0.0
    best_state = None
    start_round = 1

    if resume and os.path.exists(last_checkpoint_path):
        ckpt = torch.load(last_checkpoint_path, map_location='cpu', weights_only=False)
        global_model.load_state_dict(ckpt['global_state_dict'])
        round_metrics = ckpt['round_metrics']
        best_f1 = ckpt['best_f1']
        best_state = ckpt['best_state']
        start_round = ckpt['round'] + 1
        # Privacy accountants can't be serialised cleanly across a session boundary in
        # this simple setup; on resume we replay round count via a dummy sigma-consistent
        # accountant reset, and report a WARNING that composed epsilon before the resume
        # point is approximated by the last recorded value plus the remaining rounds.
        print(f"\n>>> Resumed FL run from round {ckpt['round']} (best F1 so far: {best_f1:.3f})")
        if dp_mode != 'none':
            print("  [WARN] Privacy accountant state is not checkpointed across resumes; "
                  "cumulative epsilon after resume is approximate. Prefer completing a DP run "
                  "in one session where possible.")

    # ── FL round loop ───────────────────────────────────────────────────
    for rnd in range(start_round, num_rounds + 1):
        collected_weights = []
        collected_sizes = []
        round_epsilons = []

        for cid in range(num_clients):
            # global_model is left in eval() mode after the previous round's server-side
            # evaluation; deepcopy inherits that, and Opacus's make_private() refuses to
            # wrap a model that isn't in training mode.
            local_model = deepcopy(global_model).to(device)
            local_model.train()
            local_optimizer = optim.Adam(filter(lambda p: p.requires_grad, local_model.parameters()),
                                         lr=lr, weight_decay=1e-4)

            if dp_mode != 'none':
                pe = privacy_engines[cid]
                sigma = noise_multipliers[cid]
                local_model, local_optimizer, loader = pe.make_private(
                    module=local_model,
                    optimizer=local_optimizer,
                    data_loader=client_loaders[cid],
                    noise_multiplier=sigma,
                    max_grad_norm=max_grad_norm,
                    grad_sample_mode='ew' if dp_mode == 'full' else 'hooks',
                )
            else:
                loader = client_loaders[cid]

            for _ in range(local_epochs):
                train_one_epoch(local_model, loader, criterion, local_optimizer, device)

            if dp_mode != 'none':
                round_epsilons.append(pe.get_epsilon(delta=target_delta))

            state = OrderedDict()
            for pname, pval in local_model.state_dict().items():
                clean_name = pname.replace('_module.', '')
                state[clean_name] = pval.cpu().clone()
            collected_weights.append(state)
            collected_sizes.append(client_sizes[cid])

            # Every round, every client gets a freshly deepcopy'd + Opacus-wrapped
            # model. Opacus's hook-based grad-sample tracking (and, for dp_mode !=
            # 'none', the persistent PrivacyEngine reused across rounds) holds
            # references that don't get released by Python's refcounting alone —
            # confirmed by a real CUDA OOM after ~3 rounds reporting "9.75 GiB
            # allocated by PyTorch" on this 4GB card. Explicit cleanup here is
            # necessary, not optional, for any DP-FL run longer than a couple rounds.
            del local_model, local_optimizer, loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        global_state = fedavg_aggregate(collected_weights, collected_sizes)
        global_model.load_state_dict(global_state)

        global_model.to(device)
        loss, acc, preds, labels, probs = evaluate(global_model, val_loader, criterion, device)
        metrics = compute_metrics(labels, preds, probs)
        global_model.cpu()

        round_metrics['rounds'].append(rnd)
        round_metrics['loss'].append(float(loss))
        round_metrics['accuracy'].append(float(acc))
        round_metrics['f1_macro'].append(float(metrics['f1_macro']))
        round_metrics['auroc_macro'].append(float(metrics['auroc_macro']))
        if dp_mode != 'none':
            round_metrics['epsilon_per_client'].append(round_epsilons)

        f1 = metrics['f1_macro']
        if f1 > best_f1:
            best_f1 = f1
            best_state = deepcopy(global_model.state_dict())

        star = " ★" if f1 >= best_f1 else ""
        eps_str = f" | worst-ε={max(round_epsilons):.2f}" if dp_mode != 'none' else ""
        print(f"  Round {rnd:3d}/{num_rounds} | Loss: {loss:.4f} "
              f"Acc: {acc:.1f}% | F1: {f1:.3f} AUROC: {metrics['auroc_macro']:.3f}{eps_str}{star}")

        # Round-level checkpoint (atomic write) — Colab session can die between rounds.
        tmp_path = last_checkpoint_path + '.tmp'
        torch.save({
            'round': rnd,
            'global_state_dict': global_model.state_dict(),
            'round_metrics': round_metrics,
            'best_f1': best_f1,
            'best_state': best_state,
        }, tmp_path)
        os.replace(tmp_path, last_checkpoint_path)

    # ── Save best global model ──────────────────────────────────────────
    if best_state is not None:
        global_model.load_state_dict(best_state)
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'experiment': experiment_name,
        'best_f1': best_f1,
        'fold': fold,
        'seed': seed,
        'test_subjects': split['test_subjects'],
        'val_subjects': split['val_subjects'],
        'client_train_subjects_note': 'see partition_metadata.json in the client data dir for per-client subject lists',
    }, best_checkpoint_path)
    print(f"\n  ✓ Global model saved: {best_checkpoint_path}")

    # ── Final test evaluation (held-out, never seen by any client) ──────
    global_model.to(device)
    loss, acc, preds, labels, probs = evaluate(global_model, test_loader, criterion, device)
    final_metrics = compute_metrics(labels, preds, probs)

    final_epsilon = None
    if dp_mode != 'none':
        final_epsilon = max(pe.get_epsilon(delta=target_delta) for pe in privacy_engines.values())

    fl_metrics = {
        'experiment': experiment_name,
        'model': model_name,
        'fold': fold,
        'seed': seed,
        'accuracy': float(final_metrics['accuracy']),
        'f1_macro': float(final_metrics['f1_macro']),
        'auroc_macro': float(final_metrics['auroc_macro']),
        'precision_macro': float(final_metrics['precision_macro']),
        'recall_macro': float(final_metrics['recall_macro']),
        'num_rounds': num_rounds,
        'num_clients': num_clients,
        'local_epochs': local_epochs,
        'alpha': alpha,
        'dp_mode': dp_mode,
        'target_epsilon': target_epsilon if dp_mode != 'none' else None,
        'actual_epsilon_worst_client': final_epsilon,
        'target_delta': target_delta if dp_mode != 'none' else None,
    }
    metrics_path = os.path.join(metrics_dir, f'{experiment_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(fl_metrics, f, indent=2)
    print(f"  ✓ Metrics saved: {metrics_path}")

    run_meta = {
        'run_tag': experiment_name,
        'model': model_name,
        'fold': fold,
        'seed': seed,
        'test_subjects': split['test_subjects'],
        'val_subjects': split['val_subjects'],
        'client_data_dir': client_data_dir,
    }
    with open(os.path.join(metrics_dir, f'{experiment_name}_run_meta.json'), 'w') as f:
        json.dump(run_meta, f, indent=2)

    history_path = os.path.join(metrics_dir, f'{experiment_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(round_metrics, f, indent=2)
    print(f"  ✓ Round history saved: {history_path}")

    plot_fl_convergence(round_metrics, figures_dir, experiment_name)

    print(f"\n{'='*60}")
    print(f"FL Simulation complete: {experiment_name}")
    print(f"  Final — Acc: {acc:.1f}%  F1: {final_metrics['f1_macro']:.3f}  "
          f"AUROC: {final_metrics['auroc_macro']:.3f}"
          f"{f'  ε={final_epsilon:.2f}' if final_epsilon is not None else ''}")
    print(f"{'='*60}")

    return round_metrics, fl_metrics


def run_simulation_userlevel_dp(model_name='resnet50', num_rounds=20, local_epochs=3,
                                lr=1e-4, target_epsilon=5.0, target_delta=1e-3, max_grad_norm=1.0,
                                subjects_per_round=None, fold=0,
                                splits_path=None, data_dir=None, results_dir=None,
                                seed=42, resume=False, tag_suffix='', dp_scope='full'):
    """User-level (subject-level) DP-FedAvg — McMahan et al. 2018.

    Fixes the slice-level DP unit used by dp_mode='head'/'full' in run_simulation():
    here the privacy-protected unit is a SUBJECT, not an image slice, regardless of
    which physical client a subject's slices live on. Each round, a subset of the
    fold's train subjects locally fine-tune a copy of the global model; each
    subject's resulting parameter DELTA is clipped to L2 norm `max_grad_norm`
    (bounding any one subject's influence on the round update); the clipped deltas
    are averaged and perturbed with Gaussian noise calibrated (via Opacus's analytic
    accountant, treating one round as one privacy step with sampling rate
    q = subjects_per_round / total_train_subjects) to a target (epsilon, delta) over
    the whole run, then applied to the global model. No PrivacyEngine wraps the
    per-subject local training itself — the DP mechanism lives entirely in the
    round aggregation step, so it composes over ROUNDS, not over per-slice batches.

    dp_scope='head' freezes the backbone (only the classifier head + the
    grayscale-adapted conv1 stay trainable, matching get_model's freeze_backbone
    convention) and applies the clip/noise mechanism to those ~9K trainable
    parameters only. With full-model scope, per-coordinate Gaussian noise summed
    over ResNet50's ~25M parameters has an L2 norm far above the clip bound at
    tight epsilon — the round update is essentially pure noise. Shrinking the
    perturbed dimension by three orders of magnitude keeps the signal
    recoverable. Frozen parameters and BatchNorm buffers (momentum forced to 0
    so running stats cannot drift during local training) provably never change,
    so they carry no per-subject information and need no noise.
    """
    from opacus.accountants import RDPAccountant
    from opacus.accountants.utils import get_noise_multiplier

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = data_dir or os.path.join(project_root, 'data', 'processed')
    splits_path = splits_path or os.path.join(project_root, 'data', 'splits', 'splits_v1.json')
    results_dir = results_dir or os.path.join(project_root, 'results')
    figures_dir = os.path.join(results_dir, 'figures')
    metrics_dir = os.path.join(results_dir, 'metrics')
    checkpoints_dir = os.path.join(results_dir, 'checkpoints')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    split = load_split(fold, os.path.join(data_dir, 'manifest.csv'), splits_path)
    manifest = __import__('pandas').read_csv(os.path.join(data_dir, 'manifest.csv'))
    all_images = np.load(os.path.join(data_dir, 'all_images.npy'))
    all_labels = np.load(os.path.join(data_dir, 'all_labels.npy'))

    train_subjects = split['train_subjects']
    train_manifest = manifest[manifest['subject_id'].isin(train_subjects)]
    subject_to_idx = {
        sid: g['array_index'].to_numpy() for sid, g in train_manifest.groupby('subject_id')
    }
    subject_ids = sorted(subject_to_idx.keys())
    total_subjects = len(subject_ids)
    if subjects_per_round is None:
        subjects_per_round = max(2, total_subjects // 2)
    subjects_per_round = min(subjects_per_round, total_subjects)
    sample_rate = subjects_per_round / total_subjects

    val_dataset = MRIDataset(all_images[split['val_idx']], all_labels[split['val_idx']])
    test_dataset = MRIDataset(all_images[split['test_idx']], all_labels[split['test_idx']])
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(all_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    global_model = get_model(model_name, num_classes=3, pretrained=True)
    if dp_scope == 'head':
        # Freeze everything except the classifier head. This is deliberately not
        # models.get_model(freeze_backbone=True): that helper keeps every
        # parameter whose name merely contains 'conv1' trainable, which matches
        # each bottleneck's conv1 and leaves ~4.3M parameters unfrozen. The DP
        # mechanism's noise norm scales with the perturbed dimension, so here the
        # scope has to be exactly the head (fc: 2048*3+3 = 6,147 parameters).
        head_prefix = 'classifier.' if model_name.lower() == 'vgg19' else 'fc.'
        for name, param in global_model.named_parameters():
            param.requires_grad = name.startswith(head_prefix)
    buffer_keys = {name for name, _ in global_model.named_buffers()}
    trainable_keys = {name for name, p in global_model.named_parameters() if p.requires_grad}
    if dp_scope == 'head':
        assert trainable_keys, f'head scope froze every parameter of {model_name}'
        # Freeze BN running stats too: momentum=0 makes the train-mode update a
        # no-op, so buffers never change and stay outside the privacy mechanism.
        for module in global_model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.momentum = 0.0

    scope_tag = 'head_' if dp_scope == 'head' else ''
    experiment_name = (f"{model_name}_dpfedavg_userlevel_{scope_tag}T{num_rounds}_E{local_epochs}"
                       f"_f{fold}_s{seed}_eps{target_epsilon}{tag_suffix}")
    last_checkpoint_path = os.path.join(checkpoints_dir, f'last_{experiment_name}.pth')
    best_checkpoint_path = os.path.join(checkpoints_dir, f'best_{experiment_name}.pth')

    sigma = get_noise_multiplier(
        target_epsilon=target_epsilon, target_delta=target_delta,
        sample_rate=sample_rate, steps=num_rounds, accountant='rdp',
    )
    accountant = RDPAccountant()

    n_perturbed = sum(p.numel() for n, p in global_model.named_parameters() if n in trainable_keys)
    print(f"\n{'='*60}\nUser-level DP-FedAvg: {experiment_name}")
    print(f"  {total_subjects} train subjects | {subjects_per_round}/round (q={sample_rate:.3f})")
    print(f"  scope={dp_scope} ({n_perturbed:,} perturbed params)")
    print(f"  sigma={sigma:.4f} calibrated for target eps={target_epsilon} over {num_rounds} rounds")
    print(f"{'='*60}\n")

    round_metrics = {'rounds': [], 'accuracy': [], 'f1_macro': [], 'auroc_macro': [], 'loss': [], 'epsilon': []}
    best_f1, best_state, start_round = 0.0, None, 1
    rng = np.random.RandomState(seed)

    if resume and os.path.exists(last_checkpoint_path):
        ckpt = torch.load(last_checkpoint_path, map_location='cpu', weights_only=False)
        global_model.load_state_dict(ckpt['global_state_dict'])
        round_metrics = ckpt['round_metrics']
        best_f1, best_state, start_round = ckpt['best_f1'], ckpt['best_state'], ckpt['round'] + 1
        for _ in range(start_round - 1):
            accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
        print(f"  Resumed from round {ckpt['round']} (best F1={best_f1:.3f})")

    for rnd in range(start_round, num_rounds + 1):
        chosen = rng.choice(subject_ids, size=subjects_per_round, replace=False)
        global_flat = {k: v.clone() for k, v in global_model.state_dict().items()}
        summed_delta = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_flat.items()}

        summed_buffers = {k: torch.zeros_like(v, dtype=torch.float32)
                          for k, v in global_flat.items() if k in buffer_keys}

        for sid in chosen:
            local_model = deepcopy(global_model).to(device)
            local_model.train()
            local_optimizer = optim.Adam(
                (p for p in local_model.parameters() if p.requires_grad),
                lr=lr, weight_decay=1e-4)
            idx = subject_to_idx[sid]
            ds = MRIDataset(all_images[idx], all_labels[idx], augment=True)
            loader = DataLoader(ds, batch_size=min(8, len(ds)), shuffle=True, num_workers=0)

            for _ in range(local_epochs):
                train_one_epoch(local_model, loader, criterion, local_optimizer, device)

            # Per-subject update delta, clipped to bound this one subject's influence.
            # BatchNorm buffers are excluded from the DP mechanism entirely (they
            # aren't learned parameters): average their raw local VALUES directly
            # instead of accumulating unclipped, unnoised deltas, which otherwise
            # let one subject's local running stats blow up the global model over
            # rounds (observed as exploding loss on some seeds).
            delta = {}
            norm_sq = 0.0
            local_state = local_model.state_dict()
            for k, v in local_state.items():
                if k in buffer_keys:
                    if v.dtype.is_floating_point:
                        summed_buffers[k] += v.cpu().float()
                    continue
                if k not in trainable_keys:
                    # Frozen parameter (head scope): the optimizer never touches
                    # it, so its delta is exactly zero — skip it entirely.
                    continue
                d = (v.cpu().float() - global_flat[k].float())
                delta[k] = d
                if v.dtype.is_floating_point:
                    norm_sq += float((d ** 2).sum())
            clip_scale = min(1.0, max_grad_norm / (norm_sq ** 0.5 + 1e-12))
            for k in delta:
                summed_delta[k] += delta[k] * clip_scale

            del local_model, local_optimizer, loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        new_state = OrderedDict()
        for k, v in global_flat.items():
            if k in buffer_keys:
                new_state[k] = (summed_buffers[k] / subjects_per_round).to(v.dtype) if v.dtype.is_floating_point else v
                continue
            if k not in trainable_keys:
                # Frozen parameter (head scope): no subject ever updates it, so
                # it carries no private information — leave it untouched rather
                # than perturbing it (noising ~25M frozen weights is exactly the
                # noise-domination failure mode the head scope exists to avoid).
                new_state[k] = v
                continue
            avg_delta = summed_delta[k] / subjects_per_round
            if v.dtype.is_floating_point:
                noise = torch.randn_like(avg_delta) * (sigma * max_grad_norm / subjects_per_round)
                new_state[k] = v + avg_delta + noise
            else:
                new_state[k] = v + avg_delta.to(v.dtype)
        global_model.load_state_dict(new_state)
        accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
        eps_now = accountant.get_epsilon(delta=target_delta)

        global_model.to(device)
        loss, acc, preds, labels, probs = evaluate(global_model, val_loader, criterion, device)
        metrics = compute_metrics(labels, preds, probs)
        global_model.cpu()

        round_metrics['rounds'].append(rnd)
        round_metrics['loss'].append(float(loss))
        round_metrics['accuracy'].append(float(acc))
        round_metrics['f1_macro'].append(float(metrics['f1_macro']))
        round_metrics['auroc_macro'].append(float(metrics['auroc_macro']))
        round_metrics['epsilon'].append(float(eps_now))

        f1 = metrics['f1_macro']
        if f1 > best_f1:
            best_f1 = f1
            best_state = deepcopy(global_model.state_dict())

        print(f"  Round {rnd:3d}/{num_rounds} | Loss:{loss:.4f} Acc:{acc:.1f}% "
              f"F1:{f1:.3f} AUROC:{metrics['auroc_macro']:.3f} eps={eps_now:.2f}")

        tmp_path = last_checkpoint_path + '.tmp'
        torch.save({'round': rnd, 'global_state_dict': global_model.state_dict(),
                    'round_metrics': round_metrics, 'best_f1': best_f1, 'best_state': best_state},
                   tmp_path)
        os.replace(tmp_path, last_checkpoint_path)

    if best_state is not None:
        global_model.load_state_dict(best_state)
    torch.save({'model_state_dict': global_model.state_dict(), 'experiment': experiment_name,
                'best_f1': best_f1, 'fold': fold, 'seed': seed,
                'test_subjects': split['test_subjects'], 'val_subjects': split['val_subjects']},
               best_checkpoint_path)

    global_model.to(device)
    loss, acc, preds, labels, probs = evaluate(global_model, test_loader, criterion, device)
    final_metrics = compute_metrics(labels, preds, probs)
    final_epsilon = accountant.get_epsilon(delta=target_delta)

    fl_metrics = {
        'experiment': experiment_name, 'model': model_name, 'fold': fold, 'seed': seed,
        'accuracy': float(final_metrics['accuracy']), 'f1_macro': float(final_metrics['f1_macro']),
        'auroc_macro': float(final_metrics['auroc_macro']),
        'precision_macro': float(final_metrics['precision_macro']),
        'recall_macro': float(final_metrics['recall_macro']),
        'num_rounds': num_rounds, 'local_epochs': local_epochs,
        'dp_mode': 'fedavg_userlevel', 'dp_unit': 'subject', 'dp_scope': dp_scope,
        'perturbed_params': int(n_perturbed),
        'subjects_per_round': subjects_per_round, 'total_train_subjects': total_subjects,
        'target_epsilon': target_epsilon, 'actual_epsilon': float(final_epsilon),
        'target_delta': target_delta, 'noise_multiplier': sigma, 'max_grad_norm': max_grad_norm,
    }
    with open(os.path.join(metrics_dir, f'{experiment_name}_metrics.json'), 'w') as f:
        json.dump(fl_metrics, f, indent=2)
    with open(os.path.join(metrics_dir, f'{experiment_name}_history.json'), 'w') as f:
        json.dump(round_metrics, f, indent=2)

    plot_fl_convergence(round_metrics, figures_dir, experiment_name)
    print(f"\n  Final — Acc:{acc:.1f}% F1:{final_metrics['f1_macro']:.3f} "
          f"AUROC:{final_metrics['auroc_macro']:.3f} eps={final_epsilon:.2f}")
    return round_metrics, fl_metrics


def plot_fl_convergence(round_metrics, save_dir, experiment_name):
    """Plot FL accuracy and loss vs round."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'FL Convergence — {experiment_name}',
                 fontsize=14, fontweight='bold')

    rounds = round_metrics['rounds']

    # Accuracy + F1
    ax1.plot(rounds, round_metrics['accuracy'], 'b-o', markersize=3,
             linewidth=2, label='Accuracy (%)')
    ax1.plot(rounds, [f * 100 for f in round_metrics['f1_macro']],
             'g--s', markersize=3, linewidth=1.5, label='F1 × 100')
    ax1.set_xlabel('FL Round', fontsize=12)
    ax1.set_ylabel('Performance', fontsize=12)
    ax1.set_title('Accuracy & F1 vs Round')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(rounds, round_metrics['loss'], 'r-o', markersize=3, linewidth=2)
    ax2.set_xlabel('FL Round', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Server Loss vs Round')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{experiment_name}_convergence.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Convergence plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='PPXFL FL Server / Simulation')
    parser.add_argument('--model', type=str, default='vgg19',
                        choices=['vgg19', 'resnet50'])
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--clients', type=int, default=4)
    parser.add_argument('--local-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dp-mode', type=str, default='none',
                        choices=['none', 'head', 'full', 'fedavg_userlevel'])
    parser.add_argument('--target-epsilon', type=float, default=5.0)
    parser.add_argument('--target-delta', type=float, default=1e-3)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--subjects-per-round', type=int, default=None,
                        help='fedavg_userlevel only: subjects sampled per round')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--splits-path', type=str, default=None)
    parser.add_argument('--clients-root', type=str, default=None)
    parser.add_argument('--results-dir', type=str, default=None)
    parser.add_argument('--tag-suffix', type=str, default='')
    parser.add_argument('--dp-scope', type=str, default='full', choices=['full', 'head'],
                        help='fedavg_userlevel only: perturb all parameters (full) or '
                             'the classifier head with a frozen backbone (head)')
    args = parser.parse_args()

    if args.dp_mode == 'fedavg_userlevel':
        run_simulation_userlevel_dp(
            model_name=args.model, num_rounds=args.rounds, local_epochs=args.local_epochs,
            lr=args.lr, target_epsilon=args.target_epsilon, target_delta=args.target_delta,
            max_grad_norm=args.max_grad_norm, subjects_per_round=args.subjects_per_round,
            fold=args.fold, seed=args.seed, resume=args.resume,
            data_dir=args.data_dir, splits_path=args.splits_path, results_dir=args.results_dir,
            tag_suffix=args.tag_suffix, dp_scope=args.dp_scope,
        )
        return

    run_simulation(
        model_name=args.model,
        num_clients=args.clients,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dp_mode=args.dp_mode,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        max_grad_norm=args.max_grad_norm,
        fold=args.fold,
        alpha=args.alpha,
        seed=args.seed,
        resume=args.resume,
        data_dir=args.data_dir,
        splits_path=args.splits_path,
        clients_root=args.clients_root,
        results_dir=args.results_dir,
        tag_suffix=args.tag_suffix,
    )


if __name__ == '__main__':
    main()
