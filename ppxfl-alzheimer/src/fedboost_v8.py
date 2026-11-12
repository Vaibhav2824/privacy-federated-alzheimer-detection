"""
fedboost_v8.py -- federated gradient boosting over the real ADNI sites.

Horizontal, histogram-based federated GBDT (the scheme behind XGBoost's
federated mode and FedTree), written in numpy so it plugs into the same folds,
clients and metrics as the FedAvg / FedProx logistic regression in
``federated_v3`` / ``experiments_v3``.

What each party sees
--------------------
* Raw region volumes and biomarkers never leave a site.
* Once, before training: each site sends per-feature quantiles of its own data
  (a fixed number of summary points, not subject rows) and its class counts.
  The server merges them into global bin edges, class weights and the initial
  score.
* Every tree node: each site sends, per feature and bin, the summed gradient,
  summed hessian and count over its subjects in that node.  The server adds the
  site histograms, picks the split, and broadcasts it.  Only the smaller child's
  histogram is requested; the sibling is parent minus child.
* Leaf values are computed by the server from the summed histograms.

Because histograms are additive, the summed histogram equals the one a central
server would build, so the federated model is the centralised histogram model
up to the quantile sketch used for the bin edges: federation costs (almost) no
accuracy.  The privacy protection is that subject rows are never shared; site
histograms are aggregates.  This arm carries no formal DP guarantee -- secure
aggregation would hide individual site histograms from the server, and DP on
histograms is future work.  The DP results stay on the logistic-regression arm.

Usage
-----
    python -m src.fedboost_v8 check   --features data/v8/features_v8_ext.npz
    python -m src.fedboost_v8 federated --features data/v8/features_v8_ext.npz \
        --out-dir results_v8/metrics_fedboost --seeds 42 123 2024
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class FedBoostConfig:
    rounds: int = 400               # boosting iterations (one tree per class each)
    learning_rate: float = 0.04
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 20
    l2: float = 1.0
    n_bins: int = 32                # non-missing bins per feature (+1 missing bin)
    sketch_points: int = 64         # quantiles each site shares per feature
    class_weight: str = "balanced"
    seed: int = 42


@dataclass
class Tree:
    feature: list = field(default_factory=list)
    threshold: list = field(default_factory=list)
    missing_left: list = field(default_factory=list)
    left: list = field(default_factory=list)
    right: list = field(default_factory=list)
    value: list = field(default_factory=list)
    count: list = field(default_factory=list)
    hess: list = field(default_factory=list)
    gain: list = field(default_factory=list)

    def add(self, count, hess, value=0.0) -> int:
        for name, v in (("feature", -1), ("threshold", np.nan), ("missing_left", True),
                        ("left", -1), ("right", -1), ("value", value),
                        ("count", count), ("hess", hess), ("gain", 0.0)):
            getattr(self, name).append(v)
        return len(self.feature) - 1

    def arrays(self) -> dict:
        return {k: np.asarray(v) for k, v in asdict(self).items()}


@dataclass
class FedBoostModel:
    trees: list            # trees[round][class] -> dict of arrays
    init: np.ndarray       # initial raw score per class
    edges: list            # per-feature bin edges
    n_classes: int
    comm: dict

    def raw_score(self, X: np.ndarray) -> np.ndarray:
        F = np.tile(self.init, (len(X), 1)).astype(float)
        for round_trees in self.trees:
            for k, t in enumerate(round_trees):
                F[:, k] += _tree_predict(t, X)
        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _softmax(self.raw_score(X))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _tree_predict(t: dict, X: np.ndarray) -> np.ndarray:
    node = np.zeros(len(X), dtype=int)
    while True:
        f = t["feature"][node]
        active = f >= 0
        if not active.any():
            return t["value"][node]
        idx = np.where(active)[0]
        x = X[idx, f[idx]]
        go_left = np.where(np.isnan(x), t["missing_left"][node[idx]],
                           x <= t["threshold"][node[idx]])
        node[idx] = np.where(go_left, t["left"][node[idx]], t["right"][node[idx]])


# ----------------------------------------------------------------------------
# set-up messages: quantile sketch and class counts
# ----------------------------------------------------------------------------

def _site_sketch(Xc: np.ndarray, points: int):
    """Per feature: ``points`` quantiles of this site's non-missing values + count."""
    q = np.linspace(0, 1, points)
    out = np.full((Xc.shape[1], points), np.nan)
    counts = np.sum(~np.isnan(Xc), axis=0)
    for j in range(Xc.shape[1]):
        v = Xc[:, j][~np.isnan(Xc[:, j])]
        if len(v):
            out[j] = np.quantile(v, q)
    return out, counts


def _merge_sketches(sketches, n_bins: int) -> list:
    """Weighted quantiles of the pooled site quantile points -> global bin edges."""
    n_features = sketches[0][0].shape[0]
    edges = []
    targets = np.linspace(0, 1, n_bins + 1)[1:-1]
    for j in range(n_features):
        vals, wts = [], []
        for pts, counts in sketches:
            if counts[j] == 0:
                continue
            vals.append(pts[j])
            wts.append(np.full(len(pts[j]), counts[j] / len(pts[j])))
        if not vals:
            edges.append(np.array([]))
            continue
        v, w = np.concatenate(vals), np.concatenate(wts)
        o = np.argsort(v)
        v, w = v[o], np.cumsum(w[o])
        w = w / w[-1]
        e = np.unique(v[np.minimum(np.searchsorted(w, targets), len(v) - 1)])
        edges.append(e[:-1] if len(e) and e[-1] >= v[-1] else e)
    return edges


def _bin(X: np.ndarray, edges: list, n_bins: int) -> np.ndarray:
    codes = np.empty(X.shape, dtype=np.int16)
    for j, e in enumerate(edges):
        c = np.searchsorted(e, X[:, j], side="left")
        c[np.isnan(X[:, j])] = n_bins          # dedicated missing bin
        codes[:, j] = c
    return codes


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------

def _best_split(hist, l2, min_leaf, n_bins):
    """hist: (F, B+1, 3) summed [G, H, N]; last bin = missing.  Returns split."""
    G, H, N = hist[..., 0], hist[..., 1], hist[..., 2]
    Gt, Ht = G.sum(1, keepdims=True), H.sum(1, keepdims=True)
    Nt = N.sum(1, keepdims=True)
    parent = (Gt ** 2 / (Ht + l2))[:, 0]
    cG = np.cumsum(G[:, :n_bins], 1)[:, :-1]
    cH = np.cumsum(H[:, :n_bins], 1)[:, :-1]
    cN = np.cumsum(N[:, :n_bins], 1)[:, :-1]
    mG, mH, mN = G[:, n_bins:], H[:, n_bins:], N[:, n_bins:]
    best = (0.0, None)
    for miss_left in (False, True):
        gl = cG + (mG if miss_left else 0)
        hl = cH + (mH if miss_left else 0)
        nl = cN + (mN if miss_left else 0)
        gr, hr, nr = Gt - gl, Ht - hl, Nt - nl
        gain = gl ** 2 / (hl + l2) + gr ** 2 / (hr + l2) - parent[:, None]
        gain[(nl < min_leaf) | (nr < min_leaf)] = -np.inf
        f, b = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[f, b] > best[0] + 1e-12:
            best = (float(gain[f, b]), (int(f), int(b), miss_left))
    return best


class _Server:
    """Holds only aggregates; asks sites for histograms of named nodes."""

    def __init__(self, codes, site_of, n_sites, n_bins):
        self.codes, self.site_of, self.n_sites, self.n_bins = codes, site_of, n_sites, n_bins
        self.messages = 0
        self.floats = 0

    def histogram(self, members, g, h):
        """Each site computes its own (F, B+1, 3) histogram; the server sums them."""
        F, B1 = self.codes.shape[1], self.n_bins + 1
        site = self.site_of[members]
        flat = (site[:, None] * F * B1 + np.arange(F)[None, :] * B1
                + self.codes[members]).ravel()
        size = self.n_sites * F * B1
        per_site = np.stack([
            np.bincount(flat, weights=np.repeat(g[members], F), minlength=size),
            np.bincount(flat, weights=np.repeat(h[members], F), minlength=size),
            np.bincount(flat, minlength=size).astype(float),
        ], axis=-1).reshape(self.n_sites, F, B1, 3)
        sending = np.unique(site)
        self.messages += len(sending)
        self.floats += len(sending) * F * B1 * 3
        return per_site.sum(0)


def _grow_tree(server, members_all, g, h, cfg, edges):
    tree = Tree()
    hist = server.histogram(members_all, g, h)
    G, H = hist[0, :, 0].sum(), hist[0, :, 1].sum()
    root = tree.add(len(members_all), H, -cfg.learning_rate * G / (H + cfg.l2))
    leaves = {root: (members_all, hist, _best_split(hist, cfg.l2, cfg.min_samples_leaf, cfg.n_bins))}
    n_leaves = 1
    while n_leaves < cfg.max_leaf_nodes:
        cand = {k: v for k, v in leaves.items() if v[2][1] is not None}
        if not cand:
            break
        node = max(cand, key=lambda k: cand[k][2][0])
        members, hist, (gain, (f, b, miss_left)) = leaves.pop(node)
        code = server.codes[members, f]
        go_left = np.where(code == cfg.n_bins, miss_left, code <= b)
        L, R = members[go_left], members[~go_left]
        small, big = (L, R) if len(L) <= len(R) else (R, L)
        h_small = server.histogram(small, g, h)
        h_big = hist - h_small
        hL, hR = (h_small, h_big) if small is L else (h_big, h_small)
        # b past the last edge = "measured vs missing" split
        tree.feature[node] = f
        tree.threshold[node] = float(edges[f][b]) if b < len(edges[f]) else np.inf
        tree.missing_left[node], tree.gain[node] = bool(miss_left), gain
        for side, mem, hh in (("left", L, hL), ("right", R, hR)):
            Gs, Hs = hh[0, :, 0].sum(), hh[0, :, 1].sum()
            child = tree.add(len(mem), Hs, -cfg.learning_rate * Gs / (Hs + cfg.l2))
            getattr(tree, side)[node] = child
            leaves[child] = (mem, hh, _best_split(hh, cfg.l2, cfg.min_samples_leaf, cfg.n_bins))
        n_leaves += 1
    leaf_of = {}
    for leaf, (mem, _, _) in leaves.items():
        leaf_of[leaf] = mem
    return tree, leaf_of


def train_fedboost(X, y, clients, cfg: FedBoostConfig, n_classes: int) -> FedBoostModel:
    """X may contain NaN (unmeasured biomarkers): routed through a missing bin."""
    t0 = time.time()
    n = len(y)
    site_of = np.full(n, -1, dtype=int)
    for s, members in enumerate(clients):
        site_of[members] = s
    if (site_of < 0).any():
        raise ValueError("every training subject must belong to a client")

    # set-up round: sketches + class counts (aggregates only)
    sketches = [_site_sketch(X[m], cfg.sketch_points) for m in clients]
    edges = _merge_sketches(sketches, cfg.n_bins)
    class_counts = sum(np.bincount(y[m], minlength=n_classes) for m in clients)
    if cfg.class_weight == "balanced":
        cw = n / (n_classes * np.maximum(class_counts, 1))
    else:
        cw = np.ones(n_classes)
    sw = cw[y]
    prior = class_counts * cw
    init = np.log(prior / prior.sum())

    codes = _bin(X, edges, cfg.n_bins)
    server = _Server(codes, site_of, len(clients), cfg.n_bins)
    Y = np.eye(n_classes)[y]
    raw = np.tile(init, (n, 1))
    members_all = np.arange(n)
    trees = []
    for _ in range(cfg.rounds):
        p = _softmax(raw)                      # computed locally by each site
        round_trees = []
        for k in range(n_classes):
            g = sw * (p[:, k] - Y[:, k])
            h = sw * p[:, k] * (1 - p[:, k])
            tree, leaf_of = _grow_tree(server, members_all, g, h, cfg, edges)
            arr = tree.arrays()
            for leaf, mem in leaf_of.items():
                raw[mem, k] += arr["value"][leaf]
            round_trees.append(arr)
        trees.append(round_trees)
    comm = {"histogram_messages": server.messages, "floats_sent": server.floats,
            "sketch_floats": int(sum(s[0].size for s in sketches)),
            "n_sites": len(clients), "seconds": round(time.time() - t0, 1)}
    return FedBoostModel(trees, init, edges, n_classes, comm)


def feature_importance(model: FedBoostModel, n_features: int) -> np.ndarray:
    """Total split gain per feature (normalised), summed over all trees."""
    imp = np.zeros(n_features)
    for rt in model.trees:
        for t in rt:
            inner = t["feature"] >= 0
            np.add.at(imp, t["feature"][inner], t["gain"][inner])
    return imp / max(imp.sum(), 1e-12)


# ----------------------------------------------------------------------------
# differentially private federated boosting
# ----------------------------------------------------------------------------
#
# Mechanism (subject-level, same trust model as DP-FedAvg in federated_v3: the
# server -- or a secure aggregator in front of it -- adds the noise once to the
# summed site histograms):
#   * every tree samples subjects with probability q (each site flips the coins
#     for its own subjects);
#   * gradients are clipped to |g| <= clip_g and hessians to [0, clip_h];
#   * trees grow level by level to depth d.  Nodes on one level hold disjoint
#     subjects, so a subject changes one node histogram per level: one bin per
#     feature in G and in H, L2 sensitivity sqrt(F (clip_g^2 + clip_h^2));
#   * leaf (G, H) sums: sensitivity sqrt(clip_g^2 + clip_h^2);
#   * each of these d + 1 releases gets Gaussian noise of sigma = z * sensitivity,
#     so one tree is a single Gaussian mechanism with multiplier z / sqrt(d + 1);
#     trees are composed with the PRV (or RDP) accountant like DP-SGD steps;
#   * split choice and leaf values use only the noisy sums (post-processing).
# Bin edges come from the per-feature mean and standard deviation, and the class
# weights from the class counts -- the same non-private aggregates the DP-FedAvg
# arm uses to standardise its inputs.

# Fixed a priori from the AD literature (medial-temporal / AD-signature cortex,
# ventricular expansion, core AD biomarkers), not selected on this data, so the
# choice costs no privacy budget.
DP_REGIONS_GM = ["Hippocampus", "Amygdala", "Ent entorhinal area", "PHG parahippocampal gyrus",
                 "ITG inferior temporal gyrus", "MTG middle temporal gyrus",
                 "FuG fusiform gyrus", "PCu precuneus"]
DP_REGIONS_CSF = ["Lateral Ventricle", "Inf Lat Vent"]
DP_OTHER = ["bio::apoe4", "bio::amyloid_cl", "bio::csf_ptau_ab42", "bio::education",
            "demo::age", "demo::sex"]
DP_PLASMA = ["bio::plasma_ptau217", "bio::plasma_nfl", "bio::plasma_gfap"]


def dp_feature_index(feature_names) -> np.ndarray:
    names = [str(n) for n in feature_names]
    want = [f"neuromorphometrics:gm::{side} {r}" for r in DP_REGIONS_GM for side in ("Left", "Right")]
    want += [f"neuromorphometrics:cs::{side} {r}" for r in DP_REGIONS_CSF for side in ("Left", "Right")]
    want += DP_OTHER
    missing = [w for w in want if w not in names]
    if missing:
        raise ValueError(f"DP feature set missing: {missing}")
    return np.asarray([names.index(w) for w in want])


@dataclass
class DPBoostConfig:
    epsilon: float | None = 1.0     # None = same algorithm without noise
    delta: float = 1e-3
    rounds: int = 20                # trees = rounds x classes
    max_depth: int = 3
    learning_rate: float = 0.3
    n_bins: int = 8
    l2: float = 5.0
    clip_g: float = 1.0
    clip_h: float = 0.25
    min_hess: float = 2.0           # noisy hessian sum a child must keep
    sample_rate: float = 0.5
    colsample: int = 0              # features per tree (0 = all); chosen at random,
                                    # independent of the data, so it costs no budget
    accountant: str = "prv"
    seed: int = 42


def _gauss_edges(X, n_bins):
    from scipy.stats import norm
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    z = norm.ppf(np.linspace(0, 1, n_bins + 1)[1:-1])
    return [np.unique(m + s * z) if s > 0 else np.array([]) for m, s in zip(mu, sd)]


def _dp_split(hist, cfg):
    """Best split from a noisy (F, B+1, 2) [G, H] histogram, or None."""
    G, H = hist[..., 0], hist[..., 1]
    B = cfg.n_bins
    Gt, Ht = G.sum(1, keepdims=True), H.sum(1, keepdims=True)
    parent = Gt ** 2 / np.maximum(Ht + cfg.l2, 1e-6)
    cG, cH = np.cumsum(G[:, :B], 1)[:, :-1], np.cumsum(H[:, :B], 1)[:, :-1]
    best = (0.0, None)
    for miss_left in (False, True):
        gl = cG + (G[:, B:] if miss_left else 0)
        hl = cH + (H[:, B:] if miss_left else 0)
        gr, hr = Gt - gl, Ht - hl
        ok = (hl > cfg.min_hess) & (hr > cfg.min_hess)
        gain = np.where(ok, gl ** 2 / (hl + cfg.l2) + gr ** 2 / (np.maximum(hr, 0) + cfg.l2)
                        - parent, -np.inf)
        f, b = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[f, b] > best[0]:
            best = (float(gain[f, b]), (int(f), int(b), miss_left))
    return best[1]


def train_dp_fedboost(X, y, clients, cfg: DPBoostConfig, n_classes: int) -> FedBoostModel:
    from .federated_v3 import calibrate_noise, compute_epsilon
    rng = np.random.default_rng(cfg.seed)
    n, F = X.shape
    site_of = np.full(n, -1, dtype=int)
    for s, m in enumerate(clients):
        site_of[m] = s
    edges = _gauss_edges(X, cfg.n_bins)
    codes = _bin(X, edges, cfg.n_bins)
    server = _Server(codes, site_of, len(clients), cfg.n_bins)

    counts = sum(np.bincount(y[m], minlength=n_classes) for m in clients)
    cw = n / (n_classes * np.maximum(counts, 1))
    sw = cw[y]
    prior = counts * cw
    init = np.log(prior / prior.sum())

    steps = cfg.rounds * n_classes
    releases = cfg.max_depth + 1
    if cfg.epsilon is None:
        z = 0.0
        eps = None
    else:
        z_tree = calibrate_noise(cfg.epsilon, cfg.sample_rate, steps, cfg.delta,
                                 accountant=cfg.accountant)
        z = z_tree * np.sqrt(releases)          # per release
        eps = compute_epsilon(z_tree, cfg.sample_rate, steps, cfg.delta, cfg.accountant)
    unit = np.sqrt(cfg.clip_g ** 2 + cfg.clip_h ** 2)
    k_feat = cfg.colsample if 0 < cfg.colsample < F else F
    sd_hist, sd_leaf = z * unit * np.sqrt(k_feat), z * unit

    Y = np.eye(n_classes)[y]
    raw = np.tile(init, (n, 1))
    trees = []
    for _ in range(cfg.rounds):
        p = _softmax(raw)
        round_trees = []
        for k in range(n_classes):
            take = rng.random(n) < cfg.sample_rate
            feats = np.sort(rng.choice(F, k_feat, replace=False))
            g = np.clip(sw * (p[:, k] - Y[:, k]), -cfg.clip_g, cfg.clip_g) * take
            h = np.clip(sw * p[:, k] * (1 - p[:, k]), 0, cfg.clip_h) * take
            tree = Tree()
            root = tree.add(int(take.sum()), 0.0)
            frontier = [(root, np.where(take)[0])]
            for _depth in range(cfg.max_depth):
                nxt = []
                for node, mem in frontier:
                    hist = server.histogram(mem, g, h)[feats][..., :2]
                    hist = hist + rng.normal(0, sd_hist, hist.shape) if z else hist
                    split = _dp_split(hist, cfg)
                    if split is None:
                        continue
                    f, b, miss_left = split
                    f = int(feats[f])
                    tree.feature[node] = f
                    tree.threshold[node] = float(edges[f][b]) if b < len(edges[f]) else np.inf
                    tree.missing_left[node] = bool(miss_left)
                    code = codes[mem, f]
                    go_left = np.where(code == cfg.n_bins, miss_left, code <= b)
                    for side, part in (("left", mem[go_left]), ("right", mem[~go_left])):
                        child = tree.add(len(part), 0.0)
                        getattr(tree, side)[node] = child
                        nxt.append((child, part))
                frontier = nxt
            # leaf values from noisy (G, H) sums over every leaf (disjoint subjects)
            arr = tree.arrays()
            leaf_of = _tree_leaves(arr, X)
            for leaf in np.where(arr["feature"] < 0)[0]:
                mem = np.where(leaf_of == leaf)[0]
                Gs, Hs = float(g[mem].sum()), float(h[mem].sum())
                if z:
                    Gs += rng.normal(0, sd_leaf)
                    Hs += rng.normal(0, sd_leaf)
                v = -cfg.learning_rate * Gs / (max(Hs, 0.0) + cfg.l2)
                arr["value"][leaf] = float(np.clip(v, -cfg.learning_rate, cfg.learning_rate))
            raw[:, k] += arr["value"][leaf_of]
            round_trees.append(arr)
        trees.append(round_trees)
    comm = {"epsilon": eps, "delta": cfg.delta, "noise_multiplier_per_tree":
            (float(z / np.sqrt(releases)) if z else 0.0), "trees": steps,
            "sigma_histogram": float(sd_hist), "sigma_leaf": float(sd_leaf),
            "n_features": F, "n_sites": len(clients)}
    return FedBoostModel(trees, init, edges, n_classes, comm)


def _tree_leaves(t: dict, X: np.ndarray) -> np.ndarray:
    node = np.zeros(len(X), dtype=int)
    while True:
        f = t["feature"][node]
        active = f >= 0
        if not active.any():
            return node
        idx = np.where(active)[0]
        x = X[idx, f[idx]]
        go_left = np.where(np.isnan(x), t["missing_left"][node[idx]],
                           x <= t["threshold"][node[idx]])
        node[idx] = np.where(go_left, t["left"][node[idx]], t["right"][node[idx]])


def run_dp_configuration(X, y, sites, label_set, split_scheme, cfg: DPBoostConfig,  # pragma: no cover
                         feature_names, n_splits=5):
    from .experiments_v3 import _metrics
    from .splits_v3 import make_clients, make_folds
    fi = dp_feature_index(feature_names)
    Xs, ys, ss, K = _subset(X[:, fi], y, sites, label_set)
    folds = make_folds(split_scheme, ys, ss, n_splits=n_splits, seed=cfg.seed)
    prob = np.zeros((len(ys), K))
    info = []
    for fold in folds:
        clients = make_clients("natural", np.arange(len(fold.train)), ys[fold.train],
                               [ss[i] for i in fold.train], seed=cfg.seed)
        m = train_dp_fedboost(Xs[fold.train], ys[fold.train], clients, cfg, K)
        prob[fold.test] = m.predict_proba(Xs[fold.test])
        info.append(m.comm)
    return {"label_set": label_set, "split_scheme": split_scheme, "config": asdict(cfg),
            "metrics": _metrics(ys, prob.argmax(1), prob, K), "per_fold": info}


def command_dp(a):  # pragma: no cover
    from .evaluate_v3 import load_features
    X, y, _, sites, names = load_features(a.features)
    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for label_set in a.label_sets:
        for eps in a.epsilons:
            for seed in a.seeds:
                cfg = DPBoostConfig(epsilon=None if eps <= 0 else eps, rounds=a.rounds,
                                    max_depth=a.max_depth, learning_rate=a.learning_rate,
                                    sample_rate=a.sample_rate, colsample=a.colsample,
                                    n_bins=a.n_bins, seed=seed)
                r = run_dp_configuration(X, y, sites, label_set, a.scheme, cfg, names)
                tag = f"dpboost_{label_set}_{a.scheme}_eps{eps:g}_s{seed}{a.tag}"
                with open(os.path.join(a.out_dir, f"{tag}.json"), "w") as h:
                    json.dump(r, h, indent=2, default=float)
                mt = r["metrics"]
                e = r["per_fold"][0]["epsilon"]
                print(f"  {tag}: acc={mt['accuracy']:.3f} bal={mt['balanced_accuracy']:.3f} "
                      f"auroc={mt['macro_auroc']:.3f} eps={e if e is None else round(e, 2)} "
                      f"sigma_hist={r['per_fold'][0]['sigma_histogram']:.2f}", flush=True)
                rows.append({"tag": tag, **mt})
    with open(os.path.join(a.out_dir, "dpboost_summary.json"), "w") as h:
        json.dump(rows, h, indent=2)


# ----------------------------------------------------------------------------
# experiments (same folds / clients / metrics as experiments_v3)
# ----------------------------------------------------------------------------
# The run_* / command_* drivers below load the ADNI feature matrix and are
# exercised by scripts/run_v8.sh, not by the CPU test suite (see .coveragerc).

def _subset(X, y, sites, label_set):
    from .evaluate_v3 import LABEL_SETS
    classes = LABEL_SETS[label_set]["classes"]
    keep = np.isin(y, classes)
    remap = {c: i for i, c in enumerate(classes)}
    return (X[keep], np.asarray([remap[v] for v in y[keep]]),
            [s for s, k in zip(sites, keep) if k], len(classes))


def run_configuration(X, y, sites, label_set, split_scheme, cfg: FedBoostConfig,  # pragma: no cover
                      n_splits=5, with_fedprox=True, prox_mu=1.0, feature_names=None):
    from .experiments_v3 import _metrics, _standardise
    from .federated_v3 import FedConfig, predict_proba, train_federated
    from .splits_v3 import make_clients, make_folds

    Xs, ys, ss, K = _subset(X, y, sites, label_set)
    folds = make_folds(split_scheme, ys, ss, n_splits=n_splits, seed=cfg.seed)
    p_boost = np.zeros((len(ys), K))
    p_prox = np.zeros((len(ys), K))
    comms, imps = [], []
    for fold in folds:
        tr_sites = [ss[i] for i in fold.train]
        clients = make_clients("natural", np.arange(len(fold.train)), ys[fold.train],
                               tr_sites, seed=cfg.seed)
        model = train_fedboost(Xs[fold.train], ys[fold.train], clients, cfg, K)
        p_boost[fold.test] = model.predict_proba(Xs[fold.test])
        comms.append(model.comm)
        imps.append(feature_importance(model, Xs.shape[1]))
        if with_fedprox:
            a, b = _standardise(Xs[fold.train], Xs[fold.test])
            res = train_federated(a, ys[fold.train], clients,
                                  FedConfig(rounds=60, local_epochs=3, learning_rate=0.5,
                                            seed=cfg.seed, prox_mu=prox_mu), n_classes=K)
            p_prox[fold.test] = predict_proba(b, res)
    out = {"label_set": label_set, "split_scheme": split_scheme, "seed": cfg.seed,
           "n_subjects": int(len(ys)), "n_features": int(Xs.shape[1]),
           "config": asdict(cfg), "fedboost": _metrics(ys, p_boost.argmax(1), p_boost, K),
           "comm_per_fold": comms}
    if with_fedprox:
        p_ens = (p_boost + p_prox) / 2
        out["fedprox_logreg"] = _metrics(ys, p_prox.argmax(1), p_prox, K)
        out["fedboost_plus_fedprox"] = _metrics(ys, p_ens.argmax(1), p_ens, K)
    imp = np.mean(imps, 0)
    if feature_names is not None:
        top = np.argsort(imp)[::-1][:15]
        out["top_features"] = [[str(feature_names[i]), round(float(imp[i]), 4)] for i in top]
    return out


def command_federated(a):  # pragma: no cover
    from .evaluate_v3 import load_features
    X, y, _, sites, names = load_features(a.features)
    os.makedirs(a.out_dir, exist_ok=True)
    summary = []
    for label_set in a.label_sets:
        for scheme in a.schemes:
            for seed in a.seeds:
                cfg = FedBoostConfig(rounds=a.rounds, seed=seed)
                r = run_configuration(X, y, sites, label_set, scheme, cfg, a.n_splits,
                                      with_fedprox=not a.no_fedprox, prox_mu=a.prox_mu,
                                      feature_names=names)
                tag = f"fedboost_{label_set}_{scheme}_s{seed}"
                with open(os.path.join(a.out_dir, f"{tag}.json"), "w") as h:
                    json.dump(r, h, indent=2)
                line = f"  {tag}: boost acc={r['fedboost']['accuracy']:.3f} " \
                       f"bal={r['fedboost']['balanced_accuracy']:.3f} " \
                       f"auroc={r['fedboost']['macro_auroc']:.3f}"
                if "fedprox_logreg" in r:
                    line += f" | fedprox acc={r['fedprox_logreg']['accuracy']:.3f}" \
                            f" | boost+prox acc={r['fedboost_plus_fedprox']['accuracy']:.3f}"
                print(line, flush=True)
                summary.append({"tag": tag, **{k: r[k] for k in r
                                               if k in ("fedboost", "fedprox_logreg",
                                                        "fedboost_plus_fedprox")}})
    with open(os.path.join(a.out_dir, "fedboost_summary.json"), "w") as h:
        json.dump(summary, h, indent=2)


def command_check(a):  # pragma: no cover
    """Federated vs centralised (sklearn HGB, same hyper-parameters) on identical folds."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score

    from .evaluate_v3 import load_features
    from .splits_v3 import make_clients, make_folds

    X, y, _, sites, _ = load_features(a.features)
    Xs, ys, ss, K = _subset(X, y, sites, "cn-mci-ad")
    cfg = FedBoostConfig(rounds=a.rounds)
    folds = make_folds("subject", ys, ss, n_splits=5, seed=42)
    pf, pc = np.zeros(len(ys), int), np.zeros(len(ys), int)
    for fold in folds:
        clients = make_clients("natural", np.arange(len(fold.train)), ys[fold.train],
                               [ss[i] for i in fold.train])
        m = train_fedboost(Xs[fold.train], ys[fold.train], clients, cfg, K)
        pf[fold.test] = m.predict_proba(Xs[fold.test]).argmax(1)
        c = HistGradientBoostingClassifier(
            max_iter=cfg.rounds, learning_rate=cfg.learning_rate,
            max_leaf_nodes=cfg.max_leaf_nodes, min_samples_leaf=cfg.min_samples_leaf,
            l2_regularization=cfg.l2, max_bins=cfg.n_bins, class_weight="balanced",
            early_stopping=False, random_state=0).fit(Xs[fold.train], ys[fold.train])
        pc[fold.test] = c.predict(Xs[fold.test])
        print(f"  fold: {len(clients)} sites, {m.comm['seconds']} s, "
              f"{m.comm['histogram_messages']} site messages", flush=True)
    print(f"federated boosting  acc={accuracy_score(ys, pf):.3f}")
    print(f"centralised sklearn acc={accuracy_score(ys, pc):.3f}")
    print(f"same prediction on {np.mean(pf == pc):.1%} of subjects")


def main(argv=None) -> int:  # pragma: no cover
    p = argparse.ArgumentParser(description="federated gradient boosting (v8)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("check", "federated", "dp"):
        s = sub.add_parser(name)
        s.add_argument("--features", default="data/v8/features_v8_ext.npz")
        s.add_argument("--rounds", type=int, default=400)
    f = sub.choices["federated"]
    f.add_argument("--out-dir", default="results_v8/metrics_fedboost")
    f.add_argument("--label-sets", nargs="+", default=["cn-mci-ad", "cn-ad"])
    f.add_argument("--schemes", nargs="+", default=["subject", "site"])
    f.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    f.add_argument("--n-splits", type=int, default=5)
    f.add_argument("--prox-mu", type=float, default=1.0)
    f.add_argument("--no-fedprox", action="store_true")
    d = sub.choices["dp"]
    d.set_defaults(rounds=20)
    d.add_argument("--out-dir", default="results_v8/metrics_dpboost")
    d.add_argument("--label-sets", nargs="+", default=["cn-mci-ad", "cn-ad"])
    d.add_argument("--scheme", default="subject")
    d.add_argument("--epsilons", nargs="+", type=float, default=[0, 1, 2, 5, 10],
                   help="0 = no noise")
    d.add_argument("--seeds", nargs="+", type=int, default=[42])
    d.add_argument("--max-depth", type=int, default=3)
    d.add_argument("--learning-rate", type=float, default=0.3)
    d.add_argument("--sample-rate", type=float, default=0.5)
    d.add_argument("--colsample", type=int, default=0)
    d.add_argument("--n-bins", type=int, default=8)
    d.add_argument("--tag", default="")
    a = p.parse_args(argv)
    {"check": command_check, "federated": command_federated, "dp": command_dp}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
