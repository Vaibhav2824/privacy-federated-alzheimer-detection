/**
 * Privacy budgets for the live federation. The noise multipliers and the
 * epsilon spent after each round come from Opacus's PRV accountant for the
 * paper's schedule: 60 rounds, subject sampling rate q = 0.5, delta = 0.001.
 * (Regenerate with PRVAccountant if ROUNDS, Q or DELTA change.)
 */

export const ROUNDS = 60;
export const Q = 0.5;
export const DELTA = 0.001;

export interface Budget { key: string; label: string; sigma: number; epsByRound: number[] | null }

export const BUDGETS: readonly Budget[] = [
  { key: '1', label: 'ε = 1', sigma: 10.156, epsByRound: [0.097, 0.142, 0.178, 0.210, 0.238, 0.264, 0.288, 0.311, 0.333, 0.353, 0.373, 0.392, 0.411, 0.429, 0.446, 0.463, 0.480, 0.496, 0.511, 0.527, 0.542, 0.557, 0.571, 0.585, 0.599, 0.613, 0.627, 0.640, 0.653, 0.666, 0.679, 0.692, 0.704, 0.716, 0.729, 0.741, 0.752, 0.764, 0.776, 0.787, 0.799, 0.810, 0.821, 0.833, 0.844, 0.854, 0.865, 0.876, 0.886, 0.897, 0.907, 0.918, 0.928, 0.938, 0.948, 0.959, 0.969, 0.978, 0.988, 0.998] },
  { key: '10', label: 'ε = 10', sigma: 1.688, epsByRound: [1.034, 1.448, 1.780, 2.067, 2.328, 2.568, 2.793, 3.006, 3.210, 3.405, 3.593, 3.775, 3.952, 4.124, 4.291, 4.455, 4.615, 4.772, 4.927, 5.078, 5.227, 5.373, 5.518, 5.660, 5.801, 5.939, 6.076, 6.211, 6.345, 6.477, 6.608, 6.738, 6.866, 6.993, 7.119, 7.244, 7.368, 7.490, 7.612, 7.733, 7.853, 7.972, 8.090, 8.208, 8.324, 8.440, 8.555, 8.670, 8.783, 8.896, 9.009, 9.121, 9.232, 9.342, 9.452, 9.562, 9.671, 9.779, 9.887, 9.994] },
  { key: 'none', label: 'No privacy', sigma: 0, epsByRound: null },
];

export function budget(key: string): Budget {
  return BUDGETS.find((b) => b.key === key) ?? BUDGETS[0];
}

/** Epsilon spent after `rounds` completed rounds; null when there is no privacy. */
export function spent(b: Budget, rounds: number): number | null {
  if (!b.epsByRound) return null;
  if (rounds <= 0) return 0;
  return b.epsByRound[Math.min(rounds, b.epsByRound.length) - 1];
}
