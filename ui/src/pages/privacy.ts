import { attackAt, BUDGETS, budgetLabel, fixed, pct, points, privacyAt, type SiteData } from '../lib/site';
import { icon, kpi, mountPage, pageHead, section } from '../ui/layout';
import { animate } from '../ui/motion';

const sigma = (data: SiteData, eps: number | null): string =>
  (privacyAt(data, 'cn-ad', eps)?.noise_multiplier ?? 0).toFixed(2);

function steps(data: SiteData, active: number | null): string {
  return BUDGETS.map((eps, i) => `<button type="button" data-eps="${eps ?? 'none'}"
      class="rounded-xl border p-3 text-left transition-colors ${eps === active ? 'border-teal bg-teal/[0.06]' : 'border-hairline bg-surface hover:border-teal/50'}"
      aria-pressed="${eps === active}">
      <span class="font-mono text-xs text-slate">Step 0${i + 1}</span>
      <span class="block font-display text-lg">${budgetLabel(eps)}</span>
      <span class="num text-xs text-slate">σ = ${sigma(data, eps)}</span></button>`).join('');
}

function cards(data: SiteData, eps: number | null): string {
  const cnad = privacyAt(data, 'cn-ad', eps);
  const base = privacyAt(data, 'cn-ad', null);
  const three = privacyAt(data, 'cn-mci-ad', eps);
  const attack = attackAt(data, 'cn-mci-ad', eps);
  const vs = (now: string, ref: string): string => (eps === null ? 'Reference model' : `${now} vs non-private (${ref})`);
  return [
    kpi('CN vs AD accuracy', pct(cnad?.accuracy), vs(points(cnad?.accuracy, base?.accuracy), pct(base?.accuracy)), 'text-teal'),
    kpi('3-class accuracy', pct(three?.accuracy), `Chance ${pct(three?.chance_accuracy)}`),
    kpi('Attack AUROC (3-class)', fixed(attack), `${vs(fixed(attack), fixed(attackAt(data, 'cn-mci-ad', null)))}; chance 0.500`, 'text-ad'),
    kpi('Attribution agreement', fixed(cnad?.explanation_agreement), vs(fixed(cnad?.explanation_agreement), fixed(base?.explanation_agreement))),
  ].join('');
}

/** Accuracy and attribution agreement across budgets, directly labelled. */
function chart(data: SiteData): string {
  const W = 640, H = 300, L = 48, R = 40, T = 24, B = 48;
  const x = (i: number): number => L + (i * (W - L - R)) / (BUDGETS.length - 1);
  const yAcc = (v: number): number => T + (1 - (v - 0.5) / 0.5) * (H - T - B);
  const yAgr = (v: number): number => T + (1 - v) * (H - T - B);
  const rows = BUDGETS.map((eps) => privacyAt(data, 'cn-ad', eps));
  const line = (pts: [number, number][]): string => pts.map(([px, py], i) => `${i ? 'L' : 'M'}${px.toFixed(1)},${py.toFixed(1)}`).join(' ');
  const acc = rows.map((r, i): [number, number] => [x(i), yAcc(r?.accuracy ?? 0.5)]);
  const agr = rows.map((r, i): [number, number] => [x(i), yAgr(Math.max(0, r?.explanation_agreement ?? 0))]);
  const grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1].map((v) => `<line x1="${L}" x2="${W - R}" y1="${yAcc(v)}" y2="${yAcc(v)}" stroke="#e2e8f0" stroke-dasharray="3 3"/>
      <text x="${L - 6}" y="${yAcc(v) + 4}" text-anchor="end" class="fill-slate font-mono text-[10px]">${v * 100}%</text>`).join('');
  const labels = BUDGETS.map((eps, i) => `<text x="${x(i)}" y="${H - B + 20}" text-anchor="middle" class="fill-ink font-mono text-[11px]">${budgetLabel(eps)}</text>`).join('');
  const dots = (pts: [number, number][], vals: string[], color: string, dy: number): string =>
    pts.map(([px, py], i) => `<circle cx="${px}" cy="${py}" r="4" fill="${color}"/><text x="${px}" y="${py + dy}" text-anchor="middle" class="font-mono text-[10px]" fill="${color}">${vals[i]}</text>`).join('');
  return `<svg viewBox="0 0 ${W} ${H}" class="w-full" role="img" aria-label="CN vs AD accuracy and attribution agreement across privacy budgets">
    ${grid}
    <line x1="${L}" x2="${W - R}" y1="${yAcc(0.5)}" y2="${yAcc(0.5)}" stroke="#D55E00" stroke-dasharray="4 3"/>
    <path d="${line(acc)}" fill="none" stroke="#0F766E" stroke-width="2"/>
    <path d="${line(agr)}" fill="none" stroke="#0f172a" stroke-width="1.5" stroke-dasharray="5 4"/>
    ${dots(acc, rows.map((r) => pct(r?.accuracy)), '#0F766E', -10)}
    ${dots(agr, rows.map((r) => `ρ ${fixed(r?.explanation_agreement, 2)}`), '#0f172a', 16)}
    <text x="${W - R + 4}" y="${acc[acc.length - 1][1] + 4}" class="fill-teal font-mono text-[10px]">acc</text>
    <text x="${W - R + 4}" y="${agr[agr.length - 1][1] + 4}" class="fill-ink font-mono text-[10px]">ρ</text>
    ${labels}
  </svg>`;
}

function matrix(data: SiteData): string {
  const rows = BUDGETS.map((eps) => {
    const cnad = privacyAt(data, 'cn-ad', eps);
    const three = privacyAt(data, 'cn-mci-ad', eps);
    return `<tr class="border-t border-hairline">
      <td class="p-3 font-medium">${budgetLabel(eps)}</td>
      <td class="num p-3 text-right">${sigma(data, eps)}</td>
      <td class="num p-3 text-right">${pct(cnad?.accuracy)}</td>
      <td class="num p-3 text-right">${pct(three?.accuracy)}</td>
      <td class="num p-3 text-right">${fixed(three?.macro_f1)}</td>
      <td class="num p-3 text-right">${fixed(attackAt(data, 'cn-mci-ad', eps))}</td>
      <td class="num p-3 text-right">${fixed(cnad?.explanation_agreement)}</td>
      <td class="num p-3 text-right">${pct(cnad?.medial_temporal_share)}</td></tr>`;
  }).join('');
  return `<div class="overflow-x-auto rounded-lg border border-hairline bg-surface"><table class="w-full text-sm">
    <thead class="bg-muted text-left font-mono text-xs uppercase tracking-wider text-slate"><tr>
      <th class="p-3">Budget</th><th class="p-3 text-right">Noise σ</th><th class="p-3 text-right">CN vs AD acc</th>
      <th class="p-3 text-right">3-class acc</th><th class="p-3 text-right">3-class macro F1</th><th class="p-3 text-right">Attack AUROC</th>
      <th class="p-3 text-right">Attribution ρ</th><th class="p-3 text-right">Medial-temporal share</th></tr></thead>
    <tbody>${rows}</tbody></table></div>
    <p class="mt-2 text-xs text-slate">Means over three seeds, subject-disjoint folds, federated over real sites. Attack: subject-level loss-based
      membership inference on the three-class model. Attribution ρ: Spearman agreement of CN vs AD region attributions with the non-private reference fit.</p>`;
}

function render(data: SiteData): string {
  const { dp_config: dp } = data;
  const p0 = privacyAt(data, 'cn-ad', null);
  const p10 = privacyAt(data, 'cn-ad', 10);
  return `
  ${pageHead('Empirical privacy evaluation · Rényi differential privacy', 'Privacy-budget explorer',
    'How calibrated Gaussian noise at each privacy budget shapes diagnostic accuracy, membership-inference risk and how far the model\'s explanations can be trusted.',
    `<span class="badge shrink-0">δ = ${dp.delta} · subject-level · clip C = ${dp.clip_norm}</span>`)}
  <section class="shell mt-10">
    <div class="card">
      <p class="font-mono text-xs uppercase tracking-wider text-slate">Interactive parameter</p>
      <h2 class="mt-1 font-sans text-lg font-semibold">Select a privacy budget (ε)</h2>
      <div id="steps" class="mt-4 grid grid-cols-2 gap-3 md:grid-cols-5">${steps(data, 10)}</div>
      <p class="mt-4 rounded-xl bg-muted p-3 text-sm text-slate" id="explain"></p>
    </div>
    <div id="cards" class="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">${cards(data, 10)}</div>
  </section>
  ${section('Accuracy vs explanation', 'More accurate, less explicable', `<div class="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
    <div class="card">${chart(data)}
      <p class="mt-2 text-xs text-slate">Solid: CN vs AD accuracy (left axis). Dashed: attribution agreement ρ (0 to 1). Orange dashed line: 50% chance.</p></div>
    <div class="card flex flex-col gap-4">
      <span class="badge badge-amber w-fit">${icon('lightbulb', '!text-sm')} Key finding</span>
      <p class="font-display text-xl leading-8">At ε = 10 the private model scores ${pct(p10?.accuracy)} against
        ${pct(p0?.accuracy)} for its non-private twin, while the agreement of its region attributions falls from
        ${fixed(p0?.explanation_agreement)} to ${fixed(p10?.explanation_agreement)}.</p>
      <p class="text-sm text-slate">At this cohort size the noise acts partly as regularisation, which is why accuracy can rise.
        An accuracy-only privacy-utility curve cannot show that the explanation moved.</p>
      <div class="rounded-xl bg-muted p-4 text-sm"><p class="font-mono text-xs uppercase tracking-wider text-slate">Calibration</p>
        <p class="mt-2">σ is the smallest noise multiplier for which the Rényi DP accountant certifies (ε, δ = ${dp.delta}) after
          ${dp.rounds} rounds with subject sampling rate ${dp.subject_sample_rate} and per-subject clipping at C = ${dp.clip_norm}.</p></div>
    </div></div>`)}
  ${section('Full trade-off matrix', 'Every budget, every metric', matrix(data))}`;
}

void mountPage('privacy', render).then((data) => {
  if (!data) return;
  const explain = (eps: number | null): void => {
    const node = document.getElementById('explain');
    if (node) node.textContent = eps === null
      ? 'No noise: the reference fit that private models and their explanations are compared against.'
      : `ε = ${eps}: Gaussian noise multiplier σ = ${sigma(data, eps)} on clipped per-subject updates (C = ${data.dp_config.clip_norm}).`;
  };
  explain(10);
  document.getElementById('steps')?.addEventListener('click', (event) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>('button[data-eps]');
    if (!button) return;
    const eps = button.dataset.eps === 'none' ? null : Number(button.dataset.eps);
    const stepsNode = document.getElementById('steps');
    const cardsNode = document.getElementById('cards');
    if (stepsNode) stepsNode.innerHTML = steps(data, eps);
    if (cardsNode) {
      cardsNode.innerHTML = cards(data, eps);
      animate(cardsNode, true);
    }
    explain(eps);
  });
});
