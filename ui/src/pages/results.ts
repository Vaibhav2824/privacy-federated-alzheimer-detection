import { attackAt, bestCentralised, COV, fixed, MODEL_NAMES, pct, points, privacyAt, TASKS, type SiteData } from '../lib/site';
import { bar, icon, mountPage, REPO, section } from '../ui/layout';
import { ring } from '../ui/motion';

const SPLITS = [
  { key: 'subject', name: 'Subject-disjoint', color: 'bg-linear-to-r from-teal to-teal-bright' },
  { key: 'site', name: 'Site-disjoint', color: 'bg-linear-to-r from-sky/70 to-sky' },
];

const CLASS_TAG: Record<string, string> = {
  CN: 'border-cn/30 bg-cn/10 text-cn',
  MCI: 'border-mci/40 bg-mci/10 text-[#9a6a00]',
  AD: 'border-ad/30 bg-ad/10 text-ad',
};

const GRADIENT = 'bg-linear-to-r from-teal to-teal-bright';

const tags = (classes: readonly string[]): string =>
  classes.map((k) => `<span class="rounded-md border px-1.5 font-mono text-[11px] ${CLASS_TAG[k]}">${k}</span>`).join('');

/** The one number the page leads with, at the size it deserves. */
function hero(data: SiteData): string {
  const c = bestCentralised(data, 'cn-ad', 'subject');
  const acc = c.accuracy ?? 0;
  const chip = (i: string, text: string): string =>
    `<span class="inline-flex items-center gap-1.5 rounded-full border border-hairline bg-surface/80 px-3 py-1 text-sm shadow-sm backdrop-blur">${icon(i, '!text-base text-teal')}${text}</span>`;
  return `
  <section class="relative overflow-hidden border-b border-hairline bg-surface">
    <div class="bg-grid absolute inset-0"></div>
    <div class="aurora -top-40 right-[5%] h-[28rem] w-[28rem] bg-teal-bright/25"></div>
    <div class="aurora top-1/3 -left-24 h-80 w-80 bg-sky/20 [animation-delay:-5s]"></div>
    <div class="shell relative grid items-center gap-12 py-16 md:py-24 lg:grid-cols-[1.25fr_1fr]">
      <div data-stagger>
        <p class="eyebrow"><span class="live-dot"></span>Headline result · CN vs AD · held-out subjects</p>
        <p class="mt-6 font-display text-[88px] font-extrabold leading-none tracking-tighter sm:text-[128px] lg:text-[148px]">
          <span class="text-gradient" data-count>${pct(acc)}</span></p>
        <h1 class="mt-4 max-w-xl text-2xl leading-snug md:text-3xl">accuracy telling Alzheimer's disease from healthy ageing, on brains the model never saw.</h1>
        <p class="mt-4 max-w-xl text-slate">${c.n_subjects} ADNI subjects, imaging features only, mean of three seeds of five-fold
          subject-disjoint cross-validation (± ${pct(c.accuracy_std)}).</p>
        <div class="mt-6 flex flex-wrap gap-2">
          ${chip('trending_up', `<b class="num">${points(acc, c.chance_accuracy)}</b>&nbsp;over chance`)}
          ${chip('area_chart', `AUROC <b class="num">${fixed(c.macro_auroc)}</b>`)}
          ${chip('shield_lock', 'Trained without pooling scans')}
        </div>
      </div>
      <div class="relative mx-auto aspect-square w-full max-w-sm" data-reveal>
        <div class="absolute inset-6 rounded-full bg-linear-to-br from-mint/70 to-sky/10 blur-2xl"></div>
        <div class="relative h-full w-full">${ring(acc, 260, 18, 'url(#ring-grad)', c.chance_accuracy)}</div>
        <div class="absolute inset-0 grid place-items-center text-center">
          <div><p class="font-mono text-xs uppercase tracking-wider text-slate">Macro AUROC</p>
            <p class="num text-5xl font-semibold tracking-tight" data-count="0.3">${fixed(c.macro_auroc)}</p>
            <p class="mt-1 font-mono text-xs text-slate">dashed tick = chance ${pct(c.chance_accuracy)}</p></div>
        </div>
      </div>
    </div>
  </section>`;
}

/** Three reasons the headline holds up, each the strongest objection answered with a number. */
function proofs(data: SiteData): string {
  const subj = bestCentralised(data, 'cn-ad', 'subject');
  const site = bestCentralised(data, 'cn-ad', 'site');
  const p0 = privacyAt(data, 'cn-ad', null);
  const p1 = privacyAt(data, 'cn-ad', 1);
  const a0 = attackAt(data, 'cn-mci-ad', null);
  const a1 = attackAt(data, 'cn-mci-ad', 1);
  const demo = data.confounds.results['cn-ad'];
  const card = (n: string, i: string, q: string, big: string, caption: string, body: string, extra = ''): string => `
    <article class="card spotlight ${extra} flex flex-col !p-7">
      <div class="flex items-center justify-between">
        <span class="grid h-11 w-11 place-items-center rounded-xl bg-mint text-teal">${icon(i, '!text-[22px]')}</span>
        <span class="font-mono text-xs text-slate">${n}</span></div>
      <p class="mt-5 text-sm font-medium text-slate">${q}</p>
      <p class="num mt-1 text-5xl font-semibold tracking-tight text-ink" data-count>${big}</p>
      <p class="mt-1 text-sm font-medium text-teal">${caption}</p>
      <div class="mt-6 flex flex-1 flex-col justify-end gap-3">${body}</div>
    </article>`;
  return `<div class="grid gap-6 lg:grid-cols-3" data-stagger>
    ${card('01', 'local_hospital', 'Does it survive hospitals it never trained on?', pct(site.accuracy),
      `${points(site.accuracy, subj.accuracy)} when whole sites are held out`,
      bar('Unseen subjects', subj.accuracy, pct(subj.accuracy), GRADIENT, subj.chance_accuracy)
      + bar('Unseen acquisition sites', site.accuracy, pct(site.accuracy), 'bg-linear-to-r from-sky/70 to-sky', site.chance_accuracy), 'beam')}
    ${card('02', 'lock', 'What does a formal privacy guarantee cost?', points(p1?.accuracy, p0?.accuracy),
      `Subject-level DP at ε = 1, δ = ${data.dp_config.delta}`,
      bar('Federated, non-private', p0?.accuracy, pct(p0?.accuracy), 'bg-slate/40', 0.5)
      + bar('Federated, ε = 1', p1?.accuracy, pct(p1?.accuracy), GRADIENT, 0.5)
      + `<p class="rounded-xl bg-muted px-3 py-2 text-xs text-slate">Membership-inference attack AUROC (3-class)
          <b class="num text-ink">${fixed(a0)} → ${fixed(a1)}</b>, toward the 0.500 of a coin flip.</p>`)}
    ${card('03', 'diversity_3', 'Is it just reading age and sex?', points(demo.imaging.accuracy.mean, demo['sex+age'].accuracy.mean),
      'imaging over a demographics-only model',
      bar('Sex + age only', demo['sex+age'].accuracy.mean, pct(demo['sex+age'].accuracy.mean), 'bg-slate/40', subj.chance_accuracy)
      + bar('Brain imaging', demo.imaging.accuracy.mean, pct(demo.imaging.accuracy.mean), GRADIENT, subj.chance_accuracy))}
  </div>`;
}

/** Every task as a race against its own chance rate. */
function lift(data: SiteData): string {
  const rows = TASKS.map((task, i) => {
    const s = bestCentralised(data, task.key, 'subject');
    const site = bestCentralised(data, task.key, 'site');
    const acc = s.accuracy ?? 0;
    return `<div class="grid items-center gap-3 md:grid-cols-[11rem_1fr_8rem]">
      <div><div class="flex gap-1">${tags(task.classes)}</div><p class="mt-1 font-display text-lg font-semibold">${task.name}</p></div>
      <div class="relative h-12 overflow-hidden rounded-xl bg-muted">
        <span data-bar="${0.15 + i * 0.12}" class="absolute inset-y-0 left-0 rounded-xl bg-linear-to-r from-teal via-teal-bright to-sky" style="width:${acc * 100}%">
          <span class="num absolute inset-y-0 right-3 flex items-center text-sm font-semibold text-white">${pct(acc)}</span></span>
        <span class="absolute inset-y-0 left-0 rounded-l-xl bg-surface/55 [background-image:repeating-linear-gradient(135deg,transparent_0_6px,rgb(15_23_42/0.1)_6px_8px)]" style="width:${s.chance_accuracy * 100}%"></span>
        <span class="absolute inset-y-0 border-l-2 border-dashed border-ink/60" style="left:${s.chance_accuracy * 100}%"></span>
        <span class="absolute inset-y-0 left-3 hidden items-center font-mono text-xs font-medium text-ink sm:flex">chance ${pct(s.chance_accuracy)}</span>
      </div>
      <div class="md:text-right"><p class="num text-2xl font-semibold text-teal" data-count="${0.2 + i * 0.12}">${points(acc, s.chance_accuracy)}</p>
        <p class="num text-xs text-slate">site-disjoint ${pct(site.accuracy)}</p></div>
    </div>`;
  }).join('');
  return `<div class="card !p-6 md:!p-8"><div class="flex flex-col gap-6">${rows}</div>
    <p class="mt-6 border-t border-hairline pt-4 text-xs text-slate">Hatched: the majority-class rate. Bar: best imaging-only model, subject-disjoint.
      Right: lift in percentage points, and the same task when whole acquisition sites are held out.</p></div>`;
}

function table(data: SiteData): string {
  const rows = TASKS.map((task) => SPLITS.map((split, i) => {
    const c = bestCentralised(data, task.key, split.key);
    const acc = c.accuracy ?? 0;
    return `<tr class="${i === 0 ? 'border-t border-hairline' : ''} transition-colors hover:bg-paper">
        ${i === 0 ? `<td rowspan="2" class="p-4 align-top">
          <div class="flex gap-1">${tags(task.classes)}</div>
          <p class="mt-2 font-display text-lg font-semibold">${task.name}</p>
          <p class="num text-xs text-slate">n = ${c.n_subjects}</p></td>` : ''}
        <td class="p-3"><span class="badge ${split.key === 'site' ? '!border-sky/20 !bg-sky/10 !text-[#0369a1]' : ''}">${split.name}</span></td>
        <td class="min-w-48 p-3">
          <div class="flex justify-between"><span class="num font-semibold">${pct(acc)}</span><span class="num text-xs text-slate">± ${pct(c.accuracy_std)}</span></div>
          <div class="track mt-1"><span data-bar class="fill ${split.color}" style="width:${acc * 100}%"></span>
            <span class="absolute inset-y-0 border-l-2 border-dashed border-ink/60" style="left:${c.chance_accuracy * 100}%"></span></div></td>
        <td class="num p-3 text-right">${fixed(c.macro_f1)}</td>
        <td class="num p-3 text-right">${fixed(c.macro_auroc)}</td>
        <td class="num p-3 text-right text-slate">${pct(c.chance_accuracy)}</td>
        <td class="num p-3 text-right">${pct(bestCentralised(data, task.key, split.key, COV).accuracy)}</td>
        <td class="p-3 text-sm text-slate">${MODEL_NAMES[c.model] ?? c.model}</td>
        ${i === 0 ? `<td rowspan="2" class="num p-4 text-right align-middle">${points(bestCentralised(data, task.key, 'site').accuracy, c.accuracy)}</td>` : ''}
      </tr>`;
  }).join('')).join('');
  return `<div class="overflow-x-auto rounded-2xl border border-hairline bg-surface shadow-(--shadow-card)">
    <table class="w-full text-sm">
      <thead class="bg-muted/70 text-left font-mono text-xs uppercase tracking-wider text-slate"><tr>
        <th class="p-3">Task</th><th class="p-3">Evaluation</th><th class="p-3">Accuracy (± sd, dashed = chance)</th>
        <th class="p-3 text-right">Macro F1</th><th class="p-3 text-right">Macro AUROC</th><th class="p-3 text-right">Chance</th>
        <th class="p-3 text-right">+ age &amp; sex</th><th class="p-3">Best model</th><th class="p-3 text-right">Site − subject</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    <p class="mt-3 text-xs text-slate">Mean ± standard deviation over three seeds of five-fold out-of-fold evaluation. Chance is the majority-class rate.
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
    return `<div class="card spotlight">
      <p class="font-mono text-xs uppercase tracking-wider">${title}</p>
      <div class="mt-4 flex flex-col gap-3">
        ${NAMES.map(([k, n]) => bar(n, r[k]?.accuracy.mean, pct(r[k]?.accuracy.mean), k.startsWith('imaging') ? GRADIENT : 'bg-slate/40')).join('')}
      </div>
      <div class="mt-4 grid grid-cols-2 gap-3 border-t border-hairline pt-4 text-sm">
        <p>Imaging, female only <span class="num block text-lg font-semibold">${pct(s.F?.accuracy.mean)}</span><span class="num text-xs text-slate">n = ${s.F?.n}</span></p>
        <p>Imaging, male only <span class="num block text-lg font-semibold">${pct(s.M?.accuracy.mean)}</span><span class="num text-xs text-slate">n = ${s.M?.n}</span></p>
      </div></div>`;
  };
  const sexAge = c.results['cn-mci-ad']['sex+age'].accuracy.mean;
  const imaging = c.results['cn-mci-ad'].imaging.accuracy.mean;
  return `<div class="grid gap-8 lg:grid-cols-[1fr_1.2fr]">
    <div class="text-lg leading-8 text-ink/80">
      <p>An imaging accuracy is evidence about imaging only if the cohort's demographics could not have produced it
        unaided. On the <span class="num">${c.n_subjects}</span> subjects for whom both sex and age are known, the same
        folds and seeds give a sex-and-age model <span class="num">${pct(sexAge)}</span> on three classes, essentially chance.</p>
      <p class="mt-4">The region model reaches <span class="num font-semibold text-teal">${pct(imaging)}</span> on the same subjects,
        <span class="num font-semibold text-teal">${points(imaging, sexAge)}</span> above the demographic baseline, and keeps working within each
        sex, where sex is constant and can contribute nothing.</p>
    </div>
    <div class="grid gap-4 md:grid-cols-2">${panel('cn-mci-ad', 'CN / MCI / AD accuracy')}${panel('cn-ad', 'CN vs AD accuracy')}</div>
  </div>`;
}

function render(data: SiteData): string {
  return `
  ${hero(data)}
  ${section('Why it holds up', 'Three objections, answered with numbers', proofs(data),
    'The usual reasons a medical-imaging accuracy turns out to be hollow, each tested on the same subjects and folds.')}
  ${section('Lift over chance', 'Every task beats its own chance rate', lift(data),
    'Binary tasks have a chance rate near 50% and the three-class task near 34%; what matters is the distance above it.')}
  ${section('Audited task performance', 'Performance across evaluation regimes', table(data))}
  ${section('Ablation · confounder control', 'Could demographics explain this?', demographics(data))}
  <section class="shell mt-20" data-reveal>
    <div class="card relative flex flex-col items-start justify-between gap-4 overflow-hidden bg-linear-to-br from-mint/60 via-surface to-sky/10 !p-8 md:flex-row md:items-center">
      <div class="flex gap-4"><span class="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-surface text-teal shadow-sm">${icon('verified_user', '!text-[22px]')}</span>
        <div><p class="font-display text-xl font-semibold">Reproduce these numbers</p>
          <p class="text-sm text-slate">Every per-run result, the fold definitions and the scripts that produced them are in the repository.
            Scans are not redistributed: they require an ADNI data use agreement.</p></div></div>
      <a class="btn btn-primary shrink-0" href="${REPO}">${icon('code')} View the code and results</a>
    </div>
  </section>`;
}

void mountPage('results', render);
