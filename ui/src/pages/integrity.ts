import { bestCentralised, fixed, pct, TASKS, type SiteData } from '../lib/site';
import { icon, mountPage, pageHead, REPO, section } from '../ui/layout';

type Status = 'Retracted' | 'Superseded' | 'Current';

const STATUS_STYLE: Record<Status, string> = {
  Retracted: 'badge badge-amber',
  Superseded: 'badge !border-slate/20 !bg-slate/10 !text-slate',
  Current: 'badge',
};

function milestone(status: Status, title: string, metric: string, metricNote: string, finding: string, body: string): string {
  return `<li class="relative pl-10">
    <span class="absolute left-0 top-1 flex h-7 w-7 items-center justify-center rounded-full border ${status === 'Current' ? 'border-teal bg-teal text-white' : 'border-hairline bg-surface text-slate'}">
      ${icon(status === 'Retracted' ? 'close' : status === 'Current' ? 'check' : 'history', '!text-sm')}</span>
    <div class="card">
      <span class="${STATUS_STYLE[status]}">${status}</span>
      <h3 class="mt-2 text-xl">${title}</h3>
      <div class="mt-4 grid gap-4 rounded-xl bg-muted p-4 md:grid-cols-2">
        <div><p class="font-mono text-xs uppercase tracking-wider text-slate">Three-class accuracy</p>
          <p class="num mt-1 text-2xl font-semibold ${status === 'Retracted' ? 'text-amber' : 'text-ink'}">${metric}</p>
          <p class="text-xs text-slate">${metricNote}</p></div>
        <div><p class="font-mono text-xs uppercase tracking-wider text-slate">Finding</p><p class="mt-1 font-medium">${finding}</p></div>
      </div>
      <p class="mt-4 text-slate">${body}</p>
    </div></li>`;
}

function timeline(data: SiteData): string {
  const h = data.history;
  return `<ol class="flex flex-col gap-6 border-l border-hairline pl-0 [&>li]:-ml-3.5">
    ${milestone('Retracted', 'v1 · Preliminary benchmark, 32 subjects', pct(h.v1.accuracy), 'Federated ResNet50, slice-level split',
      'Slice-level leakage',
      'Individual MRI slices, not subjects, were split, so slices of the same brain sat in both training and test data. The network could recognise the subject rather than the disease.')}
    ${milestone('Retracted', 'v2 · Subject-disjoint rebuild', pct(h.v2.accuracy), 'ResNet50, subject-disjoint folds; chance 33.3%',
      'Slice-axis defect',
      'With the leak removed, accuracy fell to chance. The cause was the slice rule: it picked a different anatomical plane per subject, so no architecture or optimiser could have helped.')}
    ${milestone('Superseded', `v3 · MNI registration and region features, ${h.v3.n_subjects} subjects`, pct(h.v3.accuracy), 'Best model, subject-disjoint',
      'Orientation measured, not assumed',
      'Every scan oriented by a 48-candidate registration search and registered to MNI152; anatomy summarised as grey-matter and CSF fractions over Harvard-Oxford regions. The signal rose clearly above chance.')}
    ${milestone('Superseded', `v4 · Cohort expanded to ${h.v4.n_subjects} subjects`, pct(h.v4.accuracy), 'Best model, subject-disjoint',
      'Smaller cohort was not systematically biased',
      'Rerun unchanged on 2.6 times the subjects. The headline barely moved; two secondary findings did not survive (below).')}
    ${milestone('Superseded', `v5 · ${h.v5.n_subjects} subjects, real-site federation, subject-level DP`, pct(h.v5.accuracy), 'Best model, subject-disjoint',
      'Privacy, federation and explanations measured together',
      'Federated over real acquisition sites with subject-level differential privacy, membership-inference auditing and region-level explanations, under both subject- and site-disjoint evaluation.')}
    ${milestone('Current', `v6 · Proper tissue segmentation, same ${h.v6.n_subjects} subjects`, pct(h.v6.accuracy), 'Best model, subject-disjoint, imaging only',
      'The features were the bottleneck',
      'The intensity-only tissue mixture on affine-registered volumes was replaced by network segmentation, nonlinear warping and Neuromorphometrics region volumes over intracranial volume. Same subjects, folds and seeds. These are the numbers on this site.')}
  </ol>`;
}

const DEFECT = `<div class="overflow-hidden rounded-lg border border-hairline bg-[#0f172a] text-[#e2e8f0]">
  <div class="flex justify-between border-b border-white/10 px-4 py-2 font-mono text-xs text-white/60"><span>preprocessing · slice selection</span><span>v2 → v3</span></div>
  <pre class="overflow-x-auto p-4 font-mono text-xs leading-6"><span class="text-[#fca5a5]"># the defect (v2)</span>
<span class="bg-[#ba1a1a]/30">axial_axis = np.argmax(data.shape)   # longest array axis, not an anatomical axis</span>

<span class="text-[#80d5cb]"># the fix (v3 onward)</span>
<span class="text-white/70"># orientation fitted per (series, shape) group by registering all 48 axis</span>
<span class="text-white/70"># permutations and reflections to a whole-head ICBM152 target, then every</span>
<span class="text-white/70"># scan registered to MNI152 so a voxel index means the same anatomy.</span></pre></div>`;

const ORIENTATION = [
  ['(256, 240, 160)', 'RAS', '0', 'Sagittal'],
  ['(256, 256, 166)', 'IPL', '0', 'Axial'],
  ['(166, 256, 256)', 'RAS', '1', 'Coronal'],
];

function downgraded(data: SiteData): string {
  const siteWins = TASKS.filter((t) => (bestCentralised(data, t.key, 'site').accuracy ?? 0) > (bestCentralised(data, t.key, 'subject').accuracy ?? 0)).length;
  const claim = (n: string, title: string, original: string, audited: string, resolution: string): string => `<div class="card flex flex-col">
    <span class="badge badge-amber w-fit">Downgraded claim ${n}</span>
    <h3 class="mt-3 text-xl">${title}</h3>
    <p class="mt-4 font-mono text-xs uppercase tracking-wider text-slate">Original claim</p>
    <p class="mt-1 border-l-2 border-hairline pl-3 font-display italic">${original}</p>
    <p class="mt-4 font-mono text-xs uppercase tracking-wider text-slate">Audited finding</p>
    <p class="mt-1 flex-1">${audited}</p>
    <p class="mt-4 border-t border-hairline pt-3 text-sm text-slate">${resolution}</p></div>`;
  return `<div class="grid gap-6 md:grid-cols-2">
    ${claim('01', 'Explanation collapse was overstated',
      `On ${data.history.v3.n_subjects} subjects, attribution agreement at ε = 10 measured ${fixed(data.history.v3.agreement_eps10)}: a near-total collapse.`,
      `On ${data.history.v5.n_subjects} subjects with the same pipeline the measurement gives ${fixed(data.history.v5.agreement_eps10)}. Agreement still falls by more than half at a budget where accuracy rises, but the collapse was substantially a small-sample artefact.`,
      'Reported at its revised strength.')}
    ${claim('02', 'Site-disjoint evaluation is not free',
      'On a demographic subset, site-disjoint accuracy came out above subject-disjoint.',
      `On the full cohort site-disjoint beats subject-disjoint on ${siteWins} of ${TASKS.length} tasks. A reversal that appears on one subset and not on the whole is not evidence that holding out sites costs nothing.`,
      'The full-cohort ordering is the one reported.')}
  </div>`;
}

function render(data: SiteData): string {
  return `
  ${pageHead('Methodological integrity · audit trail', 'The path to honest measurement',
    `A record of the errors found, the claims retracted, and how the headline went from ${pct(data.history.v1.accuracy)} to a defensible ${pct(data.history.v6.accuracy)} on three classes.`)}
  ${section('01 · Chronological evolution', 'Milestones and corrections', timeline(data))}
  ${section('02 · Root cause', 'The slice-axis defect', `${DEFECT}
    <div class="mt-6 overflow-x-auto rounded-lg border border-hairline bg-surface"><table class="w-full text-sm">
      <thead class="bg-muted text-left font-mono text-xs uppercase tracking-wider text-slate"><tr><th class="p-3">Stored shape</th>
        <th class="p-3">Header orientation</th><th class="p-3">np.argmax(shape)</th><th class="p-3">Plane actually extracted</th></tr></thead>
      <tbody>${ORIENTATION.map((r) => `<tr class="border-t border-hairline"><td class="num p-3">${r[0]}</td><td class="p-3">${r[1]}</td>
        <td class="num p-3 text-amber">axis ${r[2]}</td><td class="p-3 font-medium text-danger">${r[3]}</td></tr>`).join('')}</tbody></table></div>
    <p class="mt-2 text-sm text-slate">The rule raised no error. Chance-level accuracy was the correct result for that input.</p>`,
    'The v2 preprocessing chose the slice axis as the longest array axis. Over this cohort that extracts a different anatomical plane per subject.')}
  ${section('03 · Negative results', 'Findings we downgraded', downgraded(data),
    'Two findings from the first cohort did not survive the larger one. Both are reported at their revised strength.')}
  <section class="shell mt-14"><div class="card flex flex-col items-start justify-between gap-4 bg-muted md:flex-row md:items-center">
    <div class="flex gap-4"><span class="text-teal">${icon('folder_open')}</span><div><p class="font-display text-xl">Audit artefacts</p>
      <p class="text-sm text-slate">Results from every cohort, including the retracted ones, stay in the repository for the audit trail.</p></div></div>
    <a class="btn btn-primary" href="${REPO}">${icon('code')} Browse the results</a></div></section>`;
}

void mountPage('integrity', render);
