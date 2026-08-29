import { describe, expect, it } from 'vitest';

import {
  CHANCE,
  describeConfidence,
  formatMeanStd,
  formatPercent,
  softmax,
  toPredictions,
} from '../src/lib/predictions';

describe('softmax', () => {
  it('produces a distribution that sums to one', () => {
    const out = softmax([1, 2, 3]);
    expect(out.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
  });

  it('gives equal logits equal probability', () => {
    expect(softmax([2, 2, 2])).toEqual([1 / 3, 1 / 3, 1 / 3]);
  });

  it('ranks the largest logit highest', () => {
    const out = softmax([0.1, 5, 0.2]);
    expect(Math.max(...out)).toBe(out[1]);
  });

  it('stays finite for logits large enough to overflow a naive exp', () => {
    const out = softmax([1000, 999, 998]);
    expect(out.every(Number.isFinite)).toBe(true);
    expect(out.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
  });

  it('accepts a Float32Array straight from the model', () => {
    expect(softmax(new Float32Array([0, 0]))).toEqual([0.5, 0.5]);
  });

  it('rejects an empty logit vector', () => {
    expect(() => softmax([])).toThrow(/at least one logit/);
  });
});

describe('toPredictions', () => {
  it('labels the probabilities in model output order', () => {
    const out = toPredictions([0.1, 0.2, 0.7]);
    expect(out.map((p) => p.label)).toEqual(['AD', 'MCI', 'CN']);
  });

  it('sorts most likely first', () => {
    const out = toPredictions([0.5, 0.2, 0.3]);
    expect(out[0]).toEqual({ label: 'CN', probability: 0.5 });
  });

  it('rejects a vector of the wrong length', () => {
    expect(() => toPredictions([0.5, 0.5])).toThrow(/Expected 3 probabilities/);
  });
});

describe('describeConfidence', () => {
  it('calls a near-uniform result no better than chance', () => {
    expect(describeConfidence(0.34)).toBe('no better than chance');
  });

  it('does not flatter a result just above chance', () => {
    expect(describeConfidence(0.45)).toBe('weak, above chance');
  });

  it('reports a majority result as moderate', () => {
    expect(describeConfidence(0.6)).toBe('moderate');
  });

  it('reserves strong for a clearly dominant class', () => {
    expect(describeConfidence(0.85)).toBe('strong');
  });

  it('measures the chance band against the three-class rate', () => {
    expect(describeConfidence(CHANCE)).toBe('no better than chance');
    expect(describeConfidence(CHANCE + 0.06)).toBe('weak, above chance');
  });
});

describe('formatPercent', () => {
  it('renders a proportion as a percentage', () => {
    expect(formatPercent(0.4567)).toBe('45.7%');
  });

  it('honours a requested precision', () => {
    expect(formatPercent(0.4567, 0)).toBe('46%');
  });
});

describe('formatMeanStd', () => {
  it('renders a mean with its spread', () => {
    expect(formatMeanStd(0.361, 0.072, 100)).toBe('36.1 ± 7.2');
  });

  it('renders a mean alone when there is no spread to report', () => {
    expect(formatMeanStd(0.5, null, 100)).toBe('50.0');
  });

  it('renders a dash for a missing measurement', () => {
    expect(formatMeanStd(null, null)).toBe('-');
    expect(formatMeanStd(undefined, 0.1)).toBe('-');
  });

  it('keeps unscaled values at the requested precision', () => {
    expect(formatMeanStd(0.354, 0.072, 1, 3)).toBe('0.354 ± 0.072');
  });
});
