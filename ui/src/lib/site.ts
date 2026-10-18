/**
 * Shape of, and queries over, `public/data/site.json`, written by
 * `ppxfl-alzheimer/src/export_site.py`.
 *
 * Every number on every page comes through here, so no figure is typed into a
 * page by hand and the site cannot drift from the recorded runs.
 */

export interface Condition {
  /** `centralised+cov` is the centralised model with age and sex appended. */
  family: 'centralised' | 'centralised+cov' | 'federated' | 'privacy';
  model: string;
  label_set: string;
  split_scheme: string;
  client_scheme: string;
  target_epsilon: number | null;
  n_runs: number;
  n_subjects: number;
  chance_accuracy: number;
  noise_multiplier?: number;
  perturbed_dimension?: number;
  accuracy?: number;
  accuracy_std?: number;
  balanced_accuracy?: number;
  macro_f1?: number;
  macro_auroc?: number;
  explanation_agreement?: number;
  medial_temporal_share?: number;
}

interface MetricMean {
  accuracy: { mean: number; std: number };
}

export interface DimensionRow {
  model: string;
  target_epsilon: number;
  noise_multiplier: number;
  dimension: number;
  worst_case_ratio: number;
}

export interface Client {
  n_subjects: number;
  n_sites: number;
  class_counts: [number, number, number];
  label_entropy: number;
}

export interface SiteData {
  conditions: Condition[];
  attack_auroc: Record<string, number>;
  confounds: {
    n_subjects: number;
    results: Record<string, Record<string, MetricMean>>;
    stratified: Record<string, Record<string, MetricMean & { n: number }>>;
  };
  dimension_law: DimensionRow[];
  dp_config: {
    clip_norm: number;
    rounds: number;
    local_epochs: number;
    subject_sample_rate: number;
    delta: number;
    accountant: string;
    prox_mu: number;
  };
  partition: { n_clients: number; n_sites: number; clients: Client[] };
  history: {
    v1: { accuracy: number };
    v2: { accuracy: number };
    v3: { accuracy: number; n_subjects: number; agreement_eps10: number };
    v4: { accuracy: number; n_subjects: number };
    v5: { accuracy: number; n_subjects: number; agreement_eps10: number };
    v6: { accuracy: number; n_subjects: number };
    v7: { accuracy: number; n_subjects: number };
  };
}

export const TASKS = [
  { key: 'cn-ad', name: 'CN vs AD', classes: ['CN', 'AD'] },
  { key: 'mci-ad', name: 'MCI vs AD', classes: ['MCI', 'AD'] },
  { key: 'cn-mci', name: 'CN vs MCI', classes: ['CN', 'MCI'] },
  { key: 'cn-mci-ad', name: 'CN / MCI / AD', classes: ['CN', 'MCI', 'AD'] },
] as const;

/** The centralised family with age and sex appended to the imaging features. */
export const COV: Condition['family'] = 'centralised+cov';

/** Privacy budgets in the order the explorer walks them: none, then tightening. */
export const BUDGETS: (number | null)[] = [null, 10, 5, 2, 1];

export const MODEL_NAMES: Record<string, string> = {
  ensemble: 'Ensemble',
  ensemble_cv: 'Tuned ensemble',
  logreg_cv: 'Tuned logistic regression',
  logreg: 'Logistic regression',
  lda: 'Shrinkage LDA',
  rf: 'Random forest',
  svm: 'RBF SVM',
  roi_logreg: 'Region logistic regression',
};

/** The one condition matching every given field, if any. */
export function pick(data: SiteData, query: Partial<Condition>): Condition | undefined {
  const entries = Object.entries(query) as [keyof Condition, unknown][];
  return data.conditions.find((c) => entries.every(([key, value]) => c[key] === value));
}

/** The most accurate centralised model for a task and split (imaging only by default). */
export function bestCentralised(
  data: SiteData,
  labelSet: string,
  split: string,
  family: Condition['family'] = 'centralised',
): Condition {
  const candidates = data.conditions.filter(
    (c) => c.family === family && c.label_set === labelSet && c.split_scheme === split,
  );
  if (candidates.length === 0) throw new Error(`no centralised result for ${labelSet}/${split}`);
  return candidates.reduce((a, b) => ((b.accuracy ?? 0) > (a.accuracy ?? 0) ? b : a));
}

/** The subject-level DP condition for one budget (null = non-private). */
export function privacyAt(data: SiteData, labelSet: string, epsilon: number | null): Condition | undefined {
  return pick(data, { family: 'privacy', label_set: labelSet, target_epsilon: epsilon });
}

/** Mean membership-inference AUROC for a budget, when that attack was run. */
export function attackAt(data: SiteData, labelSet: string, epsilon: number | null): number | undefined {
  return data.attack_auroc[`${labelSet}|${epsilon === null ? 'none' : epsilon.toFixed(1)}`];
}

/** 0.789 -> "78.9%"; missing -> an en dash. */
export function pct(value: number | undefined, digits = 1): string {
  return value === undefined ? '–' : `${(value * 100).toFixed(digits)}%`;
}

/** A fixed-point number, or an en dash. */
export function fixed(value: number | undefined, digits = 3): string {
  return value === undefined ? '–' : value.toFixed(digits);
}

/** Signed difference in percentage points, e.g. "+3.3 pp". */
export function points(after: number | undefined, before: number | undefined): string {
  if (after === undefined || before === undefined) return '–';
  const diff = (after - before) * 100;
  return `${diff >= 0 ? '+' : '−'}${Math.abs(diff).toFixed(1)} pp`;
}

export function budgetLabel(epsilon: number | null): string {
  return epsilon === null ? 'Non-private' : `ε = ${epsilon}`;
}
