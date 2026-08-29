/**
 * The results surface: tables and one chart, rendered from results_summary.json.
 *
 * Chart drawing is inline SVG rather than a charting library, so the page ships
 * no extra dependency and the chart inherits the page's theme tokens directly.
 */

import { formatMeanStd, formatPercent } from '../lib/predictions';
import {
  bestNonPrivate,
  conditionLabel,
  epsilonSweep,
  nonPrivateConditions,
  pointsAboveChance,
  privateConditions,
  totalRuns,
  type Condition,
  type ResultsSummary,
} from '../lib/results';

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function headlineFigure(term: string, value: string, note: string): HTMLDivElement {
  const block = element('div', 'headline');
  block.append(element('dt', undefined, term));
  const definition = element('dd', undefined, value);
  definition.append(element('span', 'headline-note', note));
  block.append(definition);
  return block;
}

function metricCells(condition: Condition): HTMLTableCellElement[] {
  const accuracy = element('td', 'numeric');
  accuracy.textContent = formatMeanStd(condition.accuracy_mean, condition.accuracy_std, 100);

  const f1 = element('td', 'numeric');
  f1.textContent = formatMeanStd(condition.f1_macro_mean, condition.f1_macro_std, 1, 3);

  const auroc = element('td', 'numeric');
  auroc.textContent = formatMeanStd(condition.auroc_macro_mean, condition.auroc_macro_std, 1, 3);

  const runs = element('td', 'numeric', String(condition.n_runs));

  return [accuracy, f1, auroc, runs];
}

function resultsTable(
  heading: string,
  caption: string,
  rows: Condition[],
  emphasise?: Condition | null,
  extraColumn?: { header: string; value: (condition: Condition) => string },
): HTMLElement {
  const block = element('section', 'table-block');
  block.append(element('h3', undefined, heading));
  block.append(element('p', 'table-caption', caption));

  if (rows.length === 0) {
    block.append(element('p', 'empty', 'No runs recorded for this group yet.'));
    return block;
  }

  const scroll = element('div', 'table-scroll');
  const table = element('table');

  const headRow = element('tr');
  headRow.append(element('th', undefined, 'Condition'));
  for (const label of ['Accuracy (%)', 'Macro F1', 'Macro AUROC', 'Runs']) {
    headRow.append(element('th', 'numeric', label));
  }
  if (extraColumn) {
    headRow.append(element('th', 'numeric', extraColumn.header));
  }
  const head = element('thead');
  head.append(headRow);
  table.append(head);

  const body = element('tbody');
  for (const condition of rows) {
    const row = element('tr');
    if (emphasise && condition.condition === emphasise.condition) {
      row.dataset.emphasis = 'true';
    }
    row.append(element('th', undefined, conditionLabel(condition)));
    row.append(...metricCells(condition));
    if (extraColumn) {
      row.append(element('td', 'numeric', extraColumn.value(condition)));
    }
    body.append(row);
  }
  table.append(body);

  scroll.append(table);
  block.append(scroll);
  return block;
}

const SVG_NS = 'http://www.w3.org/2000/svg';

function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attributes: Record<string, string | number>,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) {
    node.setAttribute(key, String(value));
  }
  return node;
}

/**
 * Accuracy against privacy budget for each mechanism, with the chance rate
 * drawn in. The chance line is the point of the chart: without it a reader
 * cannot tell which of these bars represent a model that learned anything.
 */
function privacyUtilityChart(summary: ResultsSummary): HTMLElement {
  const block = element('section', 'table-block');
  block.append(element('h3', undefined, 'Accuracy against privacy budget'));
  block.append(
    element(
      'p',
      'table-caption',
      'Each line is one mechanism. The dashed rule is the three-class chance rate; a point on or below it means the model learned nothing usable at that budget.',
    ),
  );

  const series = [
    { label: 'Centralised, sample-level DP (head)', points: epsilonSweep(summary, 'centralised_dp', 'head') },
    { label: 'Federated, subject-level DP (full)', points: epsilonSweep(summary, 'dpfedavg_userlevel', 'full') },
    { label: 'Federated, subject-level DP (head)', points: epsilonSweep(summary, 'dpfedavg_userlevel', 'head') },
  ].filter((entry) => entry.points.length > 0);

  if (series.length === 0) {
    block.append(element('p', 'empty', 'No privacy sweep recorded yet.'));
    return block;
  }

  const width = 720;
  const height = 320;
  const pad = { top: 24, right: 200, bottom: 44, left: 52 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;

  const budgets = [...new Set(series.flatMap((s) => s.points.map((p) => p.epsilon)))].sort(
    (a, b) => a - b,
  );
  const maxAccuracy = Math.max(
    0.6,
    ...series.flatMap((s) => s.points.map((p) => p.accuracy_mean ?? 0)),
  );

  const x = (epsilon: number): number =>
    pad.left + (budgets.indexOf(epsilon) / Math.max(1, budgets.length - 1)) * plotWidth;
  const y = (accuracy: number): number => pad.top + (1 - accuracy / maxAccuracy) * plotHeight;

  const chart = svg('svg', {
    viewBox: `0 0 ${width} ${height}`,
    class: 'chart',
    role: 'img',
    'aria-label':
      'Line chart of mean accuracy against privacy budget for each differential privacy mechanism, with the three-class chance rate marked.',
  });

  for (const fraction of [0, 0.25, 0.5, 0.75, 1]) {
    const value = maxAccuracy * fraction;
    const gridY = y(value);
    chart.append(
      svg('line', {
        x1: pad.left,
        x2: pad.left + plotWidth,
        y1: gridY,
        y2: gridY,
        stroke: 'currentColor',
        'stroke-opacity': 0.14,
      }),
    );
    const tick = svg('text', {
      x: pad.left - 10,
      y: gridY + 4,
      'text-anchor': 'end',
      'font-size': 11,
      fill: 'currentColor',
      'fill-opacity': 0.65,
    });
    tick.textContent = formatPercent(value, 0);
    chart.append(tick);
  }

  const chanceY = y(summary.chance_accuracy);
  chart.append(
    svg('line', {
      x1: pad.left,
      x2: pad.left + plotWidth,
      y1: chanceY,
      y2: chanceY,
      stroke: 'currentColor',
      'stroke-dasharray': '5 4',
      'stroke-opacity': 0.6,
    }),
  );
  const chanceLabel = svg('text', {
    x: pad.left + plotWidth + 8,
    y: chanceY + 4,
    'font-size': 11,
    fill: 'currentColor',
    'fill-opacity': 0.7,
  });
  chanceLabel.textContent = 'chance';
  chart.append(chanceLabel);

  for (const epsilon of budgets) {
    const label = svg('text', {
      x: x(epsilon),
      y: height - 16,
      'text-anchor': 'middle',
      'font-size': 11,
      fill: 'currentColor',
      'fill-opacity': 0.65,
    });
    label.textContent = `ε = ${epsilon}`;
    chart.append(label);
  }

  const strokes = ['currentColor', 'currentColor', 'currentColor'];
  const dashes = ['0', '6 3', '2 3'];

  series.forEach((entry, index) => {
    const usable = entry.points.filter((point) => point.accuracy_mean !== null);
    const path = usable
      .map((point, i) => `${i === 0 ? 'M' : 'L'} ${x(point.epsilon)} ${y(point.accuracy_mean ?? 0)}`)
      .join(' ');
    chart.append(
      svg('path', {
        d: path,
        fill: 'none',
        stroke: strokes[index],
        'stroke-width': 2,
        'stroke-dasharray': dashes[index],
        'stroke-opacity': 0.85,
      }),
    );
    for (const point of usable) {
      chart.append(
        svg('circle', {
          cx: x(point.epsilon),
          cy: y(point.accuracy_mean ?? 0),
          r: 3.5,
          fill: 'currentColor',
        }),
      );
    }

    const legendY = pad.top + 6 + index * 20;
    chart.append(
      svg('line', {
        x1: pad.left + plotWidth + 42,
        x2: pad.left + plotWidth + 68,
        y1: legendY,
        y2: legendY,
        stroke: strokes[index],
        'stroke-width': 2,
        'stroke-dasharray': dashes[index],
      }),
    );
    const legendText = svg('text', {
      x: pad.left + plotWidth + 74,
      y: legendY + 4,
      'font-size': 11,
      fill: 'currentColor',
      'fill-opacity': 0.8,
    });
    legendText.textContent = entry.label;
    chart.append(legendText);
  });

  block.append(chart);
  return block;
}

export function mountDashboard(root: HTMLElement, summary: ResultsSummary): void {
  const best = bestNonPrivate(summary);
  const gap = best ? pointsAboveChance(best, summary.chance_accuracy) : null;

  const headlines = element('dl', 'headline-grid');
  headlines.append(
    headlineFigure(
      'Best non-private accuracy',
      best ? formatMeanStd(best.accuracy_mean, best.accuracy_std, 100) + '%' : '-',
      best ? conditionLabel(best) : 'no runs recorded',
    ),
    headlineFigure(
      'Above chance',
      gap === null ? '-' : `${gap > 0 ? '+' : ''}${gap.toFixed(1)} pts`,
      `chance is ${formatPercent(summary.chance_accuracy)} for three classes`,
    ),
    headlineFigure(
      'Runs behind these numbers',
      String(totalRuns(summary)),
      `${summary.n_conditions} conditions, cohort ${summary.cohort}`,
    ),
  );
  root.append(headlines);

  root.append(
    resultsTable(
      'Without a privacy budget',
      'Centralised training and federated averaging, averaged over folds and seeds. These are the utility ceiling everything below is measured against.',
      nonPrivateConditions(summary),
      best,
    ),
  );

  root.append(
    resultsTable(
      'Under differential privacy',
      'Sample-level DP bounds the influence of a single slice; subject-level DP bounds the influence of an entire patient, which is the guarantee a hospital would need.',
      privateConditions(summary),
      null,
      {
        header: 'Perturbed params',
        value: (condition) =>
          condition.perturbed_params === null
            ? '-'
            : condition.perturbed_params.toLocaleString('en-GB'),
      },
    ),
  );

  root.append(privacyUtilityChart(summary));
}
