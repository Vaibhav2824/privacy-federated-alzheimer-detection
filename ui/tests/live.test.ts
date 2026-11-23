import { describe, expect, it } from 'vitest';

import { BUDGETS, budget, DELTA, Q, ROUNDS, spent } from '../src/lib/budgets';
import * as F from '../src/lib/fedcore';
import { newKeyPair, roundSeed, sharedSecret } from '../src/lib/keys';

describe('fedcore', () => {
  it('has the paper model size', () => {
    expect(F.D).toBe(27);
    expect(F.P).toBe(84);
  });

  it('draws reproducible, non-IID synthetic hospitals', () => {
    const a = F.hospital(1, 40, 10), b = F.hospital(1, 40, 10), c = F.hospital(2, 40, 10);
    expect(a.train[0].x).toEqual(b.train[0].x);
    expect(a.mix).not.toEqual(c.mix);
    expect(a.mix.reduce((s, v) => s + v, 0)).toBeCloseTo(1);
    expect(a.train).toHaveLength(40);
    expect(a.test).toHaveLength(10);
    expect(new Set(F.hospital(3, 400, 0).train.map((s) => s.y))).toEqual(new Set([0, 1, 2]));
  });

  it('gives a uniform model at zero and probabilities that sum to one', () => {
    const p = F.probabilities(F.zeros(), F.hospital(4, 1, 0).train[0].x);
    p.forEach((v) => expect(v).toBeCloseTo(1 / 3));
    expect(F.predict(F.zeros(), new Array(F.D).fill(0))).toBe(0);
  });

  it('local training moves towards the true class, and clipping caps the norm', () => {
    const s = F.hospital(5, 1, 0).train[0];
    const d = F.localDelta(F.zeros(), s);
    const after = F.probabilities(d, s.x);
    expect(after[s.y]).toBeGreaterThan(1 / 3);
    const norm = (v: Float64Array): number => Math.sqrt(v.reduce((a, x) => a + x * x, 0));
    expect(norm(F.clip(d, 1))).toBeCloseTo(Math.min(1, norm(d)));
    const small = Float64Array.from({ length: F.P }, () => 0.001);
    expect(F.clip(small, 1)).toEqual(small);
  });

  it('samples about q of the patients', () => {
    const h = F.hospital(6, 200, 0);
    const { count, sum } = F.clippedSum(F.zeros(), h.train, 0.5, F.rng(9));
    expect(count).toBeGreaterThan(70);
    expect(count).toBeLessThan(130);
    expect(sum).toHaveLength(F.P);
  });

  it('round-trips fixed point, and pairwise masks cancel in the sum', () => {
    const vals = [Float64Array.from({ length: F.P }, (_, i) => i / 10 - 3), Float64Array.from({ length: F.P }, () => -1.25), Float64Array.from({ length: F.P }, () => 0.5)];
    expect(Array.from(F.decode(F.encode(vals[0])))).toEqual(Array.from(vals[0]).map((v) => Math.round(v * F.SCALE) / F.SCALE));
    const masked = vals.map((v, i) => F.mask(F.encode(v), i, [0, 1, 2].filter((j) => j !== i).map((j) => ({ id: j, seed: F.pairSeed(7, i, j, 3) }))));
    expect(F.decode(masked[0])).not.toEqual(F.decode(F.encode(vals[0])));
    const total = F.decode(F.sumMod(masked));
    total.forEach((v, i) => expect(v).toBeCloseTo(vals[0][i] - 1.25 + 0.5, 4));
    expect(F.pairSeed(1, 2, 5, 0)).toBe(F.pairSeed(1, 5, 2, 0));
    expect(F.maskStream(4, 3)).toEqual(F.maskStream(4, 3));
  });

  it('a federation learns well above chance, noise-free and private', () => {
    const hs = [1, 2, 3, 4, 5, 6, 7, 8].map((s) => F.hospital(s * 101, 150, 60));
    const test = hs.flatMap((h) => h.test);
    for (const sigma of [0, 10.156]) {
      let w = F.zeros();
      const r = F.rng(11);
      for (let t = 0; t < ROUNDS; t++) {
        const sum = F.zeros();
        hs.forEach((h) => F.clippedSum(w, h.train, Q, r).sum.forEach((v, i) => { sum[i] += v; }));
        w = F.serverStep(w, sum, Q * 1200, sigma, r);
      }
      const s = F.score(w, test);
      expect(s.total).toBe(480);
      expect(s.correct / s.total).toBeGreaterThan(0.5);
      const top = F.importance(w).slice(0, 8).map((f) => f.name).join(' ');
      expect(top).toMatch(/hippocampus|Amyloid|Tau|entorhinal/);
    }
  });

  it('gauss is standard normal on average', () => {
    const r = F.rng(3);
    const xs = Array.from({ length: 4000 }, () => F.gauss(r));
    const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
    expect(Math.abs(mean)).toBeLessThan(0.06);
  });
});

describe('budgets', () => {
  it('match the paper schedule and end at the target epsilon', () => {
    expect([ROUNDS, Q, DELTA]).toEqual([60, 0.5, 0.001]);
    const one = budget('1');
    expect(one.sigma).toBeCloseTo(10.16, 1);
    expect(spent(one, 0)).toBe(0);
    expect(spent(one, ROUNDS)).toBeCloseTo(1, 2);
    expect(spent(one, 999)).toBe(spent(one, ROUNDS));
    expect(spent(budget('10'), ROUNDS)).toBeCloseTo(10, 1);
    expect(spent(budget('none'), 5)).toBeNull();
    expect(budget('nope')).toBe(BUDGETS[0]);
    const traj = one.epsByRound as number[];
    expect(traj.every((v, i) => i === 0 || v >= traj[i - 1])).toBe(true);
  });
});

describe('keys', () => {
  it('two hospitals derive the same per-round seed; rounds differ', async () => {
    const a = await newKeyPair(), b = await newKeyPair();
    const ab = await sharedSecret(a.privateKey, b.publicKey);
    const ba = await sharedSecret(b.privateKey, a.publicKey);
    expect(ab).toEqual(ba);
    expect(await roundSeed(ab, 4)).toBe(await roundSeed(ba, 4));
    expect(await roundSeed(ab, 4)).not.toBe(await roundSeed(ab, 5));
  });
});
