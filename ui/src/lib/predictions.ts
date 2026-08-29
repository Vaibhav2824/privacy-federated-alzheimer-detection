/**
 * Turning raw model logits into something a reader can act on.
 *
 * The expanded-cohort model performs only modestly above the three-class chance
 * rate, so the presentation layer has to make confidence legible rather than
 * flattering. `describeConfidence` exists for that reason and is deliberately
 * conservative about what counts as a confident prediction.
 */

import { CLASS_NAMES, type ClassName } from './preprocess';

/** Chance accuracy for the balanced three-class task. */
export const CHANCE = 1 / 3;

export interface Prediction {
  label: ClassName;
  probability: number;
}

/**
 * Numerically stable softmax: the max logit is subtracted first so a large
 * logit cannot overflow `Math.exp`.
 */
export function softmax(logits: readonly number[] | Float32Array): number[] {
  const values = Array.from(logits);
  if (values.length === 0) {
    throw new Error('softmax needs at least one logit');
  }
  const max = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - max));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

/** Pair each probability with its class name, most likely first. */
export function toPredictions(probabilities: readonly number[]): Prediction[] {
  if (probabilities.length !== CLASS_NAMES.length) {
    throw new Error(
      `Expected ${CLASS_NAMES.length} probabilities, received ${probabilities.length}`,
    );
  }
  return CLASS_NAMES.map((label, index) => ({ label, probability: probabilities[index] }))
    .sort((a, b) => b.probability - a.probability);
}

/**
 * How much weight a reader should put on the top prediction.
 *
 * The bands are set against chance, not against 100%: a 40% top class in a
 * three-class problem is barely above guessing and must not be described as a
 * confident answer.
 */
export function describeConfidence(topProbability: number): string {
  if (topProbability < CHANCE + 0.05) {
    return 'no better than chance';
  }
  if (topProbability < 0.5) {
    return 'weak, above chance';
  }
  if (topProbability < 0.7) {
    return 'moderate';
  }
  return 'strong';
}

/** Format a probability as a whole percentage. */
export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/** Format a mean with its standard deviation, or a dash when unavailable. */
export function formatMeanStd(
  mean: number | null | undefined,
  std: number | null | undefined,
  scale = 1,
  digits = 1,
): string {
  if (mean === null || mean === undefined) {
    return '-';
  }
  const scaled = (mean * scale).toFixed(digits);
  if (std === null || std === undefined) {
    return scaled;
  }
  return `${scaled} ± ${(std * scale).toFixed(digits)}`;
}
