/**
 * The arithmetic of one live DP-FedAvg federation, shared by the solo
 * simulation and by a phone taking part in a room.
 *
 * The model is the paper's: a multinomial logistic regression over 27
 * features, 27 x 3 weights plus 3 biases = 84 parameters. The patients are
 * synthetic, drawn from a fixed generator with a disease axis that loads on
 * the same features the real model uses; no ADNI data reaches the browser.
 *
 * Secure aggregation uses pairwise additive masks over 32-bit integers: each
 * pair of hospitals shares a seed, one adds the mask stream and the other
 * subtracts it, so the masks cancel in the sum and the server only ever sees
 * the total of the clipped updates.
 */

export const CLASSES = ['CN', 'MCI', 'AD'] as const;
export const K = CLASSES.length;

/** Feature name and how strongly it moves with disease severity (z units). */
export const FEATURES: readonly { name: string; load: number }[] = [
  { name: 'Left hippocampus', load: -1.0 },
  { name: 'Right hippocampus', load: -1.0 },
  { name: 'Left amygdala', load: -0.8 },
  { name: 'Right amygdala', load: -0.8 },
  { name: 'Left entorhinal', load: -0.9 },
  { name: 'Right entorhinal', load: -0.9 },
  { name: 'Left parahippocampal', load: -0.6 },
  { name: 'Right parahippocampal', load: -0.6 },
  { name: 'Left inferior temporal', load: -0.6 },
  { name: 'Right inferior temporal', load: -0.6 },
  { name: 'Left middle temporal', load: -0.5 },
  { name: 'Right middle temporal', load: -0.5 },
  { name: 'Left fusiform', load: -0.5 },
  { name: 'Right fusiform', load: -0.5 },
  { name: 'Left precuneus', load: -0.4 },
  { name: 'Right precuneus', load: -0.4 },
  { name: 'Posterior cingulate', load: -0.4 },
  { name: 'Angular gyrus', load: -0.3 },
  { name: 'Left lateral ventricle', load: 0.6 },
  { name: 'Right lateral ventricle', load: 0.6 },
  { name: 'APOE ε4 alleles', load: 0.6 },
  { name: 'Amyloid PET', load: 1.0 },
  { name: 'Tau composite', load: 0.9 },
  { name: 'Neurodegeneration composite', load: 0.7 },
  { name: 'Education', load: -0.2 },
  { name: 'Age', load: 0.3 },
  { name: 'Sex', load: 0.0 },
];
export const D = FEATURES.length;
export const P = D * K + K;

/* ---- seeded randomness ---- */

/** mulberry32: a small, fast, seedable generator returning [0, 1). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal by Box-Muller. */
export function gauss(r: () => number): number {
  const u = 1 - r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
}

/* ---- synthetic hospitals ---- */

export interface Subject { x: number[]; y: number }
export interface Hospital { train: Subject[]; test: Subject[]; mix: number[] }

const SEVERITY = [0, 0.75, 1.6];

/**
 * One hospital's synthetic patients. Each hospital has its own class mix and
 * its own scanner offset on every feature, so the federation is non-IID the
 * way the real ADNI sites are.
 */
export function hospital(seed: number, nTrain: number, nTest: number): Hospital {
  const r = rng(seed);
  const tilt = [0.42 + 0.2 * (r() - 0.5), 0.31 + 0.2 * (r() - 0.5), 0.27 + 0.2 * (r() - 0.5)];
  const total = tilt.reduce((a, b) => a + b, 0);
  const mix = tilt.map((v) => v / total);
  const shift = FEATURES.map(() => 0.3 * gauss(r));
  const draw = (): Subject => {
    const u = r();
    const y = u < mix[0] ? 0 : u < mix[0] + mix[1] ? 1 : 2;
    const s = SEVERITY[y] + 0.45 * gauss(r);
    return { y, x: FEATURES.map((f, j) => f.load * s + shift[j] + gauss(r)) };
  };
  return {
    mix,
    train: Array.from({ length: nTrain }, draw),
    test: Array.from({ length: nTest }, draw),
  };
}

/* ---- the 84-parameter model ---- */

const wi = (k: number, j: number): number => k * D + j;
const bi = (k: number): number => D * K + k;

export function zeros(): Float64Array {
  return new Float64Array(P);
}

export function probabilities(w: Float64Array, x: number[]): number[] {
  const z = CLASSES.map((_, k) => x.reduce((acc, v, j) => acc + w[wi(k, j)] * v, w[bi(k)]));
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function predict(w: Float64Array, x: number[]): number {
  const p = probabilities(w, x);
  return p.indexOf(Math.max(...p));
}

/** Correct predictions and count on a set of subjects. */
export function score(w: Float64Array, subjects: Subject[]): { correct: number; total: number } {
  const correct = subjects.filter((s) => predict(w, s.x) === s.y).length;
  return { correct, total: subjects.length };
}

/** One subject's local training: `epochs` gradient steps from w, returned as the change in w. */
export function localDelta(w: Float64Array, s: Subject, epochs = 3, lr = 0.5): Float64Array {
  const v = Float64Array.from(w);
  for (let e = 0; e < epochs; e++) {
    const p = probabilities(v, s.x);
    for (let k = 0; k < K; k++) {
      const g = p[k] - (k === s.y ? 1 : 0);
      for (let j = 0; j < D; j++) v[wi(k, j)] -= lr * g * s.x[j];
      v[bi(k)] -= lr * g;
    }
  }
  return v.map((value, i) => value - w[i]);
}

/** Scale a vector down to L2 norm at most C. */
export function clip(delta: Float64Array, C = 1): Float64Array {
  const norm = Math.sqrt(delta.reduce((a, v) => a + v * v, 0));
  const f = norm > C ? C / norm : 1;
  return delta.map((v) => v * f);
}

/** Poisson-sample subjects with rate q and sum their clipped updates. */
export function clippedSum(w: Float64Array, train: Subject[], q: number, r: () => number, C = 1): { sum: Float64Array; count: number } {
  const sum = zeros();
  let count = 0;
  for (const s of train) {
    if (r() >= q) continue;
    count++;
    clip(localDelta(w, s), C).forEach((v, i) => { sum[i] += v; });
  }
  return { sum, count };
}

/* ---- secure aggregation over 32-bit integers ---- */

export const SCALE = 1e5;

export function encode(v: Float64Array): Uint32Array {
  return Uint32Array.from(v, (x) => Math.round(x * SCALE) >>> 0);
}

export function decode(u: Uint32Array): Float64Array {
  return Float64Array.from(u, (x) => (x | 0) / SCALE);
}

/** A mask stream of `length` 32-bit words, deterministic in the seed. */
export function maskStream(seed: number, length: number): Uint32Array {
  const r = rng(seed);
  return Uint32Array.from({ length }, () => Math.floor(r() * 4294967296) >>> 0);
}

/** Add (mine below the peer's id) or subtract each pairwise mask, modulo 2^32. */
export function mask(encoded: Uint32Array, me: number, peers: { id: number; seed: number }[]): Uint32Array {
  const out = Uint32Array.from(encoded);
  for (const peer of peers) {
    const m = maskStream(peer.seed, out.length);
    const sign = me < peer.id ? 1 : -1;
    for (let i = 0; i < out.length; i++) out[i] = (out[i] + sign * m[i]) >>> 0;
  }
  return out;
}

/** Sum masked vectors modulo 2^32: the pairwise masks cancel exactly. */
export function sumMod(vectors: Uint32Array[]): Uint32Array {
  const out = new Uint32Array(vectors[0].length);
  for (const v of vectors) for (let i = 0; i < v.length; i++) out[i] = (out[i] + v[i]) >>> 0;
  return out;
}

/** Seed shared by the pair (a, b) in one round of a solo run. */
export function pairSeed(runSeed: number, a: number, b: number, round: number): number {
  const lo = Math.min(a, b), hi = Math.max(a, b);
  return (Math.imul(runSeed ^ 0x9e3779b9, 31) + Math.imul(lo + 1, 7919) + Math.imul(hi + 1, 104729) + Math.imul(round + 1, 1299709)) >>> 0;
}

/* ---- the server step ---- */

/** w + (sum + N(0, σ²C²I)) / expected count: noise is added once, here. */
export function serverStep(w: Float64Array, sum: Float64Array, expected: number, sigma: number, r: () => number, C = 1): Float64Array {
  return w.map((v, i) => v + (sum[i] + sigma * C * gauss(r)) / expected);
}

/** Feature importance: the sum over classes of each feature's absolute weight. */
export function importance(w: Float64Array): { name: string; value: number }[] {
  return FEATURES.map((f, j) => ({ name: f.name, value: CLASSES.reduce((a, _, k) => a + Math.abs(w[wi(k, j)]), 0) }))
    .sort((a, b) => b.value - a.value);
}
