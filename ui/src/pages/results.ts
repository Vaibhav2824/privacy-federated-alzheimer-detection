import { bestCentralised, fixed, MODEL_NAMES, pct, points, TASKS, type SiteData } from '../lib/site';
import { bar, icon, kpi, mountPage, pageHead, REPO, section } from '../ui/layout';

const SPLITS = [
  { key: 'subject', name: 'Subject-disjoint', color: 'bg-teal' },
  { key: 'site', name: 'Site-disjoint', color: 'bg-slate' },
];

const CLASS_TAG: Record<string, string> = {
  CN: 'border-cn/30 bg-cn/10 text-cn',
  MCI: 'border-mci/40 bg-mci/10 text-[#9a6a00]',
  AD: 'border-ad/30 bg-ad/10 text-ad',
};

function table(data: SiteData): string {
  const rows = TASKS.map((task) => {
    const cells = SPLITS.map((split, i) => {
      const c = bestCentralised(data, task.key, split.key);
      const acc = c.accuracy ?? 0;
      return `<tr class="${i === 0 ? 'border-t border-hairline' : ''}">
        ${i === 0 ? `<td rowspan="2" class="p-4 align-top">
          <div class="flex gap-1">${task.classes.map((k) => `<span class="rounded border px-1.5 font-mono text-[11px] ${CLASS_TAG[k]}">${k}</span>`).join('')}</div>
          <p class="mt-2 font-serif text-xl">${task.name}</p>
          <p class="num text-xs text-slate">n = ${c.n_subjects}</p></td>` : ''}
        <td class="p-3"><span class="badge ${split.key === 'site' ? '!border-slate/20 !bg-slate/10 !text-slate' : ''}">${split.name}</span></td>
        <td class="min-w-48 p-3">
          <div class="flex justify-between"><span class="num font-semibold">${pct(acc)}</span><span class="num text-xs text-slate">± ${pct(c.accuracy_std)}</span></div>
          <div class="track mt-1"><span class="fill ${split.color}" style="width:${acc * 100}%"></span>
            <span class="absolute inset-y-0 border-l-2 border-dashed border-ink/60" style="left:${c.chance_accuracy * 100}%"></span></div></td>
        <td class="num p-3 text-right">${fixed(c.macro_f1)}</td>
        <td class="num p-3 text-right">${fixed(c.macro_auroc)}</td>
        <td class="num p-3 text-right text-slate">${pct(c.chance_accuracy)}</td>
        <td class="num p-3 text-right">${pct(bestCentralised(data, task.key, split.key, 'centralised+cov').accuracy)}</td>
        <td class="p-3 text-sm text-slate">${MODEL_NAMES[c.model] ?? c.model}</td>
        ${i === 0 ? `<td rowspan="2" class="num p-4 text-right align-middle">${points(bestCentralised(data, task.key, 'site').accuracy, c.accuracy)}</td>` : ''}
      </tr>`;
    }).join('');
    return cells;
  }).join('');
  return `<div class="overflow-x-auto rounded-lg border border-hairline bg-surface">
    <table class="w-full text-sm">
      <thead class="bg-muted text-left font-mono text-xs uppercase tracking-wider text-slate"><tr>
        <th class="p-3">Task</th><th class="p-3">Evaluation</th><th class="p-3">Accuracy (± sd, dashed = chance)</th>
        <th class="p-3 text-right">Macro F1</th><th class="p-3 text-right">Macro AUROC</th><th class="p-3 text-right">Chance</th>
        <th class="p-3 text-right">+ age &amp; sex</th><th class="p-3">Best model</th><th class="p-3 text-right">Site − subject</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    <p class="mt-2 text-xs text-slate">Mean ± standard deviation over three seeds of five-fold out-of-fold evaluation. Chance is the majority-class rate.
      Imaging features only; each row shows the best of an ensemble, a tuned ensemble and a tuned logistic regression, with hyperparameters
      chosen by nested cross-validation. The age-and-sex column appends both to the same features.</p>`;
}

function demographics(data: SiteData): string {
  const c = data.confounds;
  const NAMES: [string, string][] = [
    ['sex', 'Sex only'], ['age', 'Age only'], ['sex+age', 'Sex + age'],
    ['imaging', 'Imaging (region model)'], ['imaging+demographics', 'Imaging + demographics'],
  ];
  const panel = (ls: string, title: string): string => {
    const r = c.results[ls];
    const s = c.stratified[ls];
    return `<div class="card">
      <p class="font-mono text-xs uppercase tracking-wider">${title}</p>
      <div class="mt-4 flex flex-col gap-3">
        ${NAMES.map(([k, n]) => bar(n, r[k]?.accuracy.mean, pct(r[k]?.accuracy.mean), k.startsWith('imaging') ? 'bg-teal' : 'bg-slate/50')).join('')}
      </div>
      <div class="mt-4 grid grid-cols-2 gap-3 border-t border-hairline pt-4 text-sm">
        <p>Imaging, female only <span class="num block text-lg">${pct(s.F?.accuracy.mean)}</span><span class="num text-xs text-slate">n = ${s.F?.n}</span></p>
        <p>Imaging, male only <span class="num block text-lg">${pct(s.M?.accuracy.mean)}</span><span class="num text-xs text-slate">n = ${s.M?.n}</span></p>
      </div></div>`;
  };
  const sexAge = c.results['cn-mci-ad']['sex+age'].accuracy.mean;
  const imaging = c.results['cn-mci-ad'].imaging.accuracy.mean;
  return `<div class="grid gap-8 lg:grid-cols-[1fr_1.2fr]">
    <div class="font-serif text-lg leading-8">
      <p>An imaging accuracy is evidence about imaging only if the cohort's demographics could not have produced it
        unaided. On the <span class="num">${c.n_subjects}</span> subjects for whom both sex and age are known, the same
        folds and seeds give a sex-and-age model <span class="num">${pct(sexAge)}</span> on three classes, essentially chance.</p>
      <p class="mt-4">The region model reaches <span class="num">${pct(imaging)}</span> on the same subjects,
        <span class="num">${points(imaging, sexAge)}</span> above the demographic baseline, and keeps working within each
        sex, where sex is constant and can contribute nothing.</p>
    </div>
    <div class="grid gap-4 md:grid-cols-2">${panel('cn-mci-ad', 'CN / MCI / AD accuracy')}${panel('cn-ad', 'CN vs AD accuracy')}</div>
  </div>`;
}

function render(data: SiteData): string {
  const cnad = bestCentralised(data, 'cn-ad', 'subject');
  const threeSite = bestCentralised(data, 'cn-mci-ad', 'site');
  const gaps = TASKS.map((t) => (bestCentralised(data, t.key, 'site').accuracy ?? 0) - (bestCentralised(data, t.key, 'subject').accuracy ?? 0));
  const meanGap = gaps.reduce((a, b) => a + b, 0) / gaps.length;
  const demo = data.confounds.results['cn-mci-ad'];
  return `
  ${pageHead('Benchmarks · subject-disjoint vs site-disjoint evaluation', 'Diagnostic results',
    'Four diagnostic tasks, each evaluated twice: with no subject in more than one split, and with no acquisition site in more than one split, the setting a federated model is actually deployed into.')}
  <section class="shell mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
    ${kpi('Top binary AUROC', fixed(cnad.macro_auroc), 'CN vs AD, subject-disjoint', 'text-teal')}
    ${kpi('3-class, site-disjoint', pct(threeSite.accuracy), `${points(threeSite.accuracy, threeSite.chance_accuracy)} above chance (${pct(threeSite.chance_accuracy)})`)}
    ${kpi('Mean site − subject gap', `${meanGap >= 0 ? '+' : '−'}${Math.abs(meanGap * 100).toFixed(1)} pp`, 'Holding out whole sites, averaged over four tasks', 'text-ad')}
    ${kpi('Demographic baseline', pct(demo['sex+age'].accuracy.mean), `Sex + age, 3-class; imaging ${points(demo.imaging.accuracy.mean, demo['sex+age'].accuracy.mean)}`)}
  </section>
  ${section('Audited task performance', 'Performance across evaluation regimes', table(data))}
  ${section('Ablation · confounder control', 'Could demographics explain this?', demographics(data))}
  <section class="shell mt-14">
    <div class="card flex flex-col items-start justify-between gap-4 bg-muted md:flex-row md:items-center">
      <div class="flex gap-4"><span class="text-teal">${icon('verified_user')}</span>
        <div><p class="font-serif text-xl">Reproduce these numbers</p>
          <p class="text-sm text-slate">Every per-run result, the fold definitions and the scripts that produced them are in the repository.
            Scans are not redistributed: they require an ADNI data use agreement.</p></div></div>
      <a class="btn btn-primary" href="${REPO}">${icon('code')} View the code and results</a>
    </div>
  </section>`;
}

void mountPage('results', render);
