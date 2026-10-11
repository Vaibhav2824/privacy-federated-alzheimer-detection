import { describe, expect, it } from 'vitest';

import {
  attackAt,
  bestCentralised,
  budgetLabel,
  fixed,
  pct,
  pick,
  points,
  privacyAt,
  type Condition,
  type SiteData,
} from '../src/lib/site';

function condition(overrides: Partial<Condition>): Condition {
  return {
    family: 'centralised',
    model: 'logreg',
    label_set: 'cn-ad',
    split_scheme: 'subject',
    client_scheme: 'none',
    target_epsilon: null,
    n_runs: 3,
    n_subjects: 500,
    chance_accuracy: 0.52,
    ...overrides,
  };
}

const data = {
  conditions: [
    condition({ model: 'logreg', accuracy: 0.75 }),
    condition({ model: 'ensemble', accuracy: 0.79 }),
    condition({ model: 'rf' }),
    condition({ family: 'privacy', model: 'roi_logreg', target_epsilon: 10, accuracy: 0.78 }),
    condition({ family: 'privacy', model: 'roi_logreg', accuracy: 0.75 }),
  ],
  attack_auroc: { 'cn-mci-ad|none': 0.59, 'cn-mci-ad|10.0': 0.55 },
} as unknown as SiteData;

describe('site queries', () => {
  it('picks the condition matching every field', () => {
    expect(pick(data, { model: 'ensemble' })?.accuracy).toBe(0.79);
    expect(pick(data, { model: 'svm' })).toBeUndefined();
  });

  it('returns the most accurate centralised model, ignoring missing accuracy', () => {
    expect(bestCentralised(data, 'cn-ad', 'subject').model).toBe('ensemble');
    expect(() => bestCentralised(data, 'cn-ad', 'site')).toThrow(/no centralised result/);
    const missingFirst = { conditions: [condition({ model: 'rf' }), condition({ model: 'lda', accuracy: 0.6 })] } as SiteData;
    expect(bestCentralised(missingFirst, 'cn-ad', 'subject').model).toBe('lda');
    const withCovariates = { conditions: [condition({ family: 'centralised+cov', model: 'logreg_cv', accuracy: 0.8 })] } as SiteData;
    expect(bestCentralised(withCovariates, 'cn-ad', 'subject', 'centralised+cov').accuracy).toBe(0.8);
    expect(() => bestCentralised(withCovariates, 'cn-ad', 'subject')).toThrow();
  });

  it('separates the private and non-private privacy arms', () => {
    expect(privacyAt(data, 'cn-ad', 10)?.accuracy).toBe(0.78);
    expect(privacyAt(data, 'cn-ad', null)?.accuracy).toBe(0.75);
  });

  it('looks up attack AUROC by budget key', () => {
    expect(attackAt(data, 'cn-mci-ad', null)).toBe(0.59);
    expect(attackAt(data, 'cn-mci-ad', 10)).toBe(0.55);
    expect(attackAt(data, 'cn-ad', 1)).toBeUndefined();
  });
});

describe('formatting', () => {
  it('formats percentages, fixed numbers and dashes', () => {
    expect(pct(0.789)).toBe('78.9%');
    expect(pct(undefined)).toBe('–');
    expect(fixed(0.43712)).toBe('0.437');
    expect(fixed(undefined)).toBe('–');
  });

  it('signs point differences', () => {
    expect(points(0.784, 0.751)).toBe('+3.3 pp');
    expect(points(0.716, 0.751)).toBe('−3.5 pp');
    expect(points(undefined, 0.5)).toBe('–');
  });

  it('labels budgets', () => {
    expect(budgetLabel(null)).toBe('Non-private');
    expect(budgetLabel(2)).toBe('ε = 2');
  });
});
