/**
 * Shape of, and queries over, the results summary produced by `src/aggregate.py`.
 *
 * The dashboard renders whatever this file contains: no number is written into
 * the page by hand, so the site cannot drift from the recorded experiments.
 */

export interface Condition {
  condition: string;
  model: string;
  method: string;
  dp_scope: string | null;
  epsilon: number | null;
  n_runs: number;
  folds: number[];
  seeds: number[];
  tags: string[];
  accuracy_mean: number | null;
  accuracy_std: number | null;
  f1_macro_mean: number | null;
  f1_macro_std: number | null;
  auroc_macro_mean: number | null;
  auroc_macro_std: number | null;
  precision_macro_mean: number | null;
  recall_macro_mean: number | null;
  actual_epsilon_mean: number | null;
  perturbed_params: number | null;
}

export interface ResultsSummary {
  cohort: string;
  chance_accuracy: number;
  n_conditions: number;
  conditions: Condition[];
}

/** Display names for the internal method identifiers. */
export const METHOD_LABELS: Record<string, string> = {
  centralised: 'Centralised training',
  fedavg: 'Federated averaging',
  centralised_dp: 'Centralised, sample-level DP',
  fedavg_dp: 'Federated, sample-level DP',
  dpfedavg_userlevel: 'Federated, subject-level DP',
};

export function methodLabel(method: string): string {
  return METHOD_LABELS[method] ?? method;
}

/** Display names for the model architectures. */
export const MODEL_LABELS: Record<string, string> = {
  resnet50: 'ResNet50',
  vgg19: 'VGG19',
};

export function modelLabel(model: string): string {
  return MODEL_LABELS[model] ?? model;
}

/**
 * A readable name for one condition, used as a table row heading.
 *
 * The model has to appear: two architectures under the same method would
 * otherwise produce two identical row labels with different numbers.
 */
export function conditionLabel(condition: Condition): string {
  const parts = [`${modelLabel(condition.model)}, ${methodLabel(condition.method).toLowerCase()}`];
  if (condition.dp_scope) {
    parts.push(`${condition.dp_scope}-scope`);
  }
  if (condition.epsilon !== null) {
    parts.push(`ε = ${condition.epsilon}`);
  }
  return parts.join(', ');
}

/** Conditions that used no privacy mechanism, strongest macro-F1 first. */
export function nonPrivateConditions(summary: ResultsSummary): Condition[] {
  return summary.conditions
    .filter((condition) => condition.epsilon === null)
    .sort((a, b) => (b.f1_macro_mean ?? 0) - (a.f1_macro_mean ?? 0));
}

/** A condition that ran under a privacy budget, with epsilon known to be set. */
export type PrivateCondition = Condition & { epsilon: number };

function isPrivate(condition: Condition): condition is PrivateCondition {
  return condition.epsilon !== null;
}

/** Private conditions, grouped by mechanism and ordered by increasing budget. */
export function privateConditions(summary: ResultsSummary): PrivateCondition[] {
  return summary.conditions.filter(isPrivate).sort((a, b) => {
    const byMethod = a.method.localeCompare(b.method);
    if (byMethod !== 0) return byMethod;
    const byScope = (a.dp_scope ?? '').localeCompare(b.dp_scope ?? '');
    if (byScope !== 0) return byScope;
    return a.epsilon - b.epsilon;
  });
}

/** The privacy-utility curve for one mechanism and scope. */
export function epsilonSweep(
  summary: ResultsSummary,
  method: string,
  scope: string | null,
): PrivateCondition[] {
  return summary.conditions
    .filter(
      (condition): condition is PrivateCondition =>
        condition.method === method && condition.dp_scope === scope && isPrivate(condition),
    )
    .sort((a, b) => a.epsilon - b.epsilon);
}

/** The best non-private condition, used as the utility reference point. */
export function bestNonPrivate(summary: ResultsSummary): Condition | null {
  return nonPrivateConditions(summary)[0] ?? null;
}

/** How far above the three-class chance rate a condition sits, in points. */
export function pointsAboveChance(
  condition: Condition,
  chance: number,
): number | null {
  if (condition.accuracy_mean === null) {
    return null;
  }
  return (condition.accuracy_mean - chance) * 100;
}

/** Total runs behind the summary, for the "what this is built from" line. */
export function totalRuns(summary: ResultsSummary): number {
  return summary.conditions.reduce((sum, condition) => sum + condition.n_runs, 0);
}
