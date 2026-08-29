import { describe, expect, it } from 'vitest';

import {
  bestNonPrivate,
  conditionLabel,
  epsilonSweep,
  methodLabel,
  modelLabel,
  nonPrivateConditions,
  pointsAboveChance,
  privateConditions,
  totalRuns,
  type Condition,
  type ResultsSummary,
} from '../src/lib/results';

function condition(overrides: Partial<Condition> = {}): Condition {
  return {
    condition: 'resnet50 | centralised',
    model: 'resnet50',
    method: 'centralised',
    dp_scope: null,
    epsilon: null,
    n_runs: 7,
    folds: [0, 1, 2, 3, 4],
    seeds: [42, 123, 2024],
    tags: [],
    accuracy_mean: 0.361,
    accuracy_std: 0.072,
    f1_macro_mean: 0.354,
    f1_macro_std: 0.072,
    auroc_macro_mean: 0.579,
    auroc_macro_std: 0.054,
    precision_macro_mean: 0.36,
    recall_macro_mean: 0.36,
    actual_epsilon_mean: null,
    perturbed_params: null,
    ...overrides,
  };
}

function summary(conditions: Condition[]): ResultsSummary {
  return {
    cohort: 'v2',
    chance_accuracy: 1 / 3,
    n_conditions: conditions.length,
    conditions,
  };
}

describe('modelLabel', () => {
  it('expands a known architecture identifier', () => {
    expect(modelLabel('vgg19')).toBe('VGG19');
  });

  it('passes an unknown architecture through unchanged', () => {
    expect(modelLabel('efficientnet')).toBe('efficientnet');
  });
});

describe('methodLabel', () => {
  it('expands a known method identifier', () => {
    expect(methodLabel('dpfedavg_userlevel')).toBe('Federated, subject-level DP');
  });

  it('passes an unknown identifier through unchanged', () => {
    expect(methodLabel('something_new')).toBe('something_new');
  });
});

describe('conditionLabel', () => {
  it('names a non-private condition by its model and method', () => {
    expect(conditionLabel(condition())).toBe('ResNet50, centralised training');
  });

  it('distinguishes two architectures under the same method', () => {
    const resnet = conditionLabel(condition({ model: 'resnet50' }));
    const vgg = conditionLabel(condition({ model: 'vgg19' }));
    expect(resnet).not.toBe(vgg);
  });

  it('passes an unknown architecture through unchanged', () => {
    expect(conditionLabel(condition({ model: 'efficientnet' }))).toContain('efficientnet');
  });

  it('includes the scope and budget for a private condition', () => {
    const label = conditionLabel(
      condition({ method: 'dpfedavg_userlevel', dp_scope: 'head', epsilon: 2 }),
    );
    expect(label).toBe('ResNet50, federated, subject-level dp, head-scope, ε = 2');
  });
});

describe('nonPrivateConditions', () => {
  it('keeps only conditions with no privacy budget', () => {
    const data = summary([condition(), condition({ epsilon: 5, dp_scope: 'head' })]);
    expect(nonPrivateConditions(data)).toHaveLength(1);
  });

  it('orders them by macro-F1, strongest first', () => {
    const data = summary([
      condition({ method: 'fedavg', f1_macro_mean: 0.3 }),
      condition({ method: 'centralised', f1_macro_mean: 0.35 }),
    ]);
    expect(nonPrivateConditions(data).map((c) => c.method)).toEqual(['centralised', 'fedavg']);
  });

  it('treats a missing F1 as the weakest result, whichever side it is on', () => {
    const missingFirst = summary([
      condition({ method: 'fedavg', f1_macro_mean: null }),
      condition({ method: 'centralised', f1_macro_mean: 0.1 }),
    ]);
    expect(nonPrivateConditions(missingFirst)[0].method).toBe('centralised');

    const missingSecond = summary([
      condition({ method: 'centralised', f1_macro_mean: 0.1 }),
      condition({ method: 'fedavg', f1_macro_mean: null }),
    ]);
    expect(nonPrivateConditions(missingSecond)[0].method).toBe('centralised');
  });
});

describe('privateConditions', () => {
  it('groups by mechanism then scope then increasing budget', () => {
    const data = summary([
      condition({ method: 'dpfedavg_userlevel', dp_scope: 'head', epsilon: 10 }),
      condition({ method: 'dpfedavg_userlevel', dp_scope: 'full', epsilon: 2 }),
      condition({ method: 'dpfedavg_userlevel', dp_scope: 'head', epsilon: 2 }),
      condition({ method: 'centralised_dp', dp_scope: 'head', epsilon: 5 }),
    ]);
    expect(
      privateConditions(data).map((c) => `${c.method}/${c.dp_scope}/${c.epsilon}`),
    ).toEqual([
      'centralised_dp/head/5',
      'dpfedavg_userlevel/full/2',
      'dpfedavg_userlevel/head/2',
      'dpfedavg_userlevel/head/10',
    ]);
  });

  it('excludes non-private conditions', () => {
    expect(privateConditions(summary([condition()]))).toEqual([]);
  });

  it('sorts by budget when method and scope match', () => {
    const data = summary([
      condition({ method: 'centralised_dp', dp_scope: 'head', epsilon: 10 }),
      condition({ method: 'centralised_dp', dp_scope: 'head', epsilon: 2 }),
    ]);
    expect(privateConditions(data).map((c) => c.epsilon)).toEqual([2, 10]);
  });

  it('treats a missing scope as sortable from either side', () => {
    const scopeSecond = summary([
      condition({ method: 'centralised_dp', dp_scope: 'head', epsilon: 5 }),
      condition({ method: 'centralised_dp', dp_scope: null, epsilon: 5 }),
    ]);
    expect(privateConditions(scopeSecond)[0].dp_scope).toBeNull();

    const scopeFirst = summary([
      condition({ method: 'centralised_dp', dp_scope: null, epsilon: 5 }),
      condition({ method: 'centralised_dp', dp_scope: 'head', epsilon: 5 }),
    ]);
    expect(privateConditions(scopeFirst)[0].dp_scope).toBeNull();
  });

  it('orders by budget even when one side records no budget value', () => {
    const data = summary([
      condition({ method: 'centralised_dp', dp_scope: 'head', epsilon: 5 }),
      condition({ method: 'centralised_dp', dp_scope: 'head', epsilon: 0 }),
    ]);
    expect(privateConditions(data).map((c) => c.epsilon)).toEqual([0, 5]);
  });
});

describe('epsilonSweep', () => {
  it('returns one mechanism and scope, ordered by budget', () => {
    const data = summary([
      condition({ method: 'dpfedavg_userlevel', dp_scope: 'head', epsilon: 10 }),
      condition({ method: 'dpfedavg_userlevel', dp_scope: 'head', epsilon: 2 }),
      condition({ method: 'dpfedavg_userlevel', dp_scope: 'full', epsilon: 5 }),
    ]);
    expect(epsilonSweep(data, 'dpfedavg_userlevel', 'head').map((c) => c.epsilon)).toEqual([2, 10]);
  });

  it('returns nothing for a mechanism that was never run', () => {
    expect(epsilonSweep(summary([condition()]), 'fedavg_dp', 'full')).toEqual([]);
  });

  it('sorts a zero budget ahead of a positive one', () => {
    const data = summary([
      condition({ method: 'centralised_dp', dp_scope: 'head', epsilon: 5 }),
      condition({ method: 'centralised_dp', dp_scope: 'head', epsilon: 0 }),
    ]);
    expect(epsilonSweep(data, 'centralised_dp', 'head').map((c) => c.epsilon)).toEqual([0, 5]);
  });
});

describe('bestNonPrivate', () => {
  it('picks the strongest non-private condition', () => {
    const data = summary([
      condition({ method: 'fedavg', f1_macro_mean: 0.304 }),
      condition({ method: 'centralised', f1_macro_mean: 0.354 }),
    ]);
    expect(bestNonPrivate(data)?.method).toBe('centralised');
  });

  it('returns null when every condition is private', () => {
    expect(bestNonPrivate(summary([condition({ epsilon: 2 })]))).toBeNull();
  });
});

describe('pointsAboveChance', () => {
  it('reports the gap over chance in percentage points', () => {
    expect(pointsAboveChance(condition({ accuracy_mean: 0.4 }), 1 / 3)).toBeCloseTo(6.67, 2);
  });

  it('reports a negative gap for a below-chance result', () => {
    expect(pointsAboveChance(condition({ accuracy_mean: 0.28 }), 1 / 3)).toBeLessThan(0);
  });

  it('returns null when accuracy was not recorded', () => {
    expect(pointsAboveChance(condition({ accuracy_mean: null }), 1 / 3)).toBeNull();
  });
});

describe('totalRuns', () => {
  it('sums the runs behind every condition', () => {
    expect(totalRuns(summary([condition({ n_runs: 7 }), condition({ n_runs: 3 })]))).toBe(10);
  });

  it('is zero for an empty summary', () => {
    expect(totalRuns(summary([]))).toBe(0);
  });
});
