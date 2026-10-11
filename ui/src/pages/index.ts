import { bestCentralised, fixed, pct, pick, points, privacyAt, type Condition, type SiteData } from '../lib/site';
import { BIBTEX, href, icon, kpi, mountPage, PAPER, section } from '../ui/layout';


const DIM_NAMES: Record<string, string> = {
  roi_logreg: 'Region model',
  resnet50_head: 'ResNet50 head only',
  resnet50_full: 'ResNet50, full network',
};

/** Log-scale bar width for a noise-to-signal ratio between 0.1 and ~150. */
const logWidth = (ratio: number): number => Math.min(1, Math.max(0.02, (Math.log10(ratio) + 1) / 3.2));

function dimensionLaw(data: SiteData): string {
  const rows = data.dimension_law.filter((r) => r.target_epsilon === 1).sort((a, b) => a.dimension - b.dimension);
  const threshold = logWidth(1) * 100;
  const bars = rows
    .map((r) => {
      const ok = r.worst_case_ratio < 1;
      return `<div class="flex flex-col gap-1">
        <div class="flex justify-between text-sm"><span class="font-medium">${DIM_NAMES[r.model] ?? r.model}
          <span class="num text-slate">(d = ${r.dimension.toLocaleString('en')})</span></span>
          <span class="num ${ok ? 'text-teal' : 'text-danger'}">${r.worst_case_ratio.toFixed(2)} · ${ok ? 'signal survives' : 'noise dominates'}</span></div>
        <div class="relative h-6 rounded bg-muted">
          <span class="absolute inset-y-0 left-0 rounded ${ok ? 'bg-teal' : r.worst_case_ratio < 10 ? 'bg-slate' : 'bg-danger'}" style="width:${logWidth(r.worst_case_ratio) * 100}%"></span>
          <span class="absolute -inset-y-1 border-l-2 border-dashed border-ad" style="left:${threshold}%"></span>
        </div></div>`;
    })
    .join('');
  return `<div class="grid gap-8 lg:grid-cols-2">
    <div>
      <p class="font-serif text-lg leading-8">The Gaussian noise added to a summed update is isotropic, so its expected
        norm grows as <span class="num">σC√d</span>, while the update itself is bounded by <span class="num">NC</span>
        whatever the dimension. Whether a private model is usable is therefore set by the number of
        parameters it perturbs, not by the cohort or the mechanism.</p>
      <p class="mt-4 text-slate">At ε = 1 the region model stays at
        <span class="num text-ink">${pct(privacyAt(data, 'cn-ad', 1)?.accuracy)}</span> CN vs AD accuracy.</p>
    </div>
    <div class="card">
      <p class="font-serif text-xl">Worst-case noise-to-signal ratio at ε = 1</p>
      <p class="mt-1 text-sm text-slate">Log scale. Dashed line: ratio 1.0, above which the noise outweighs the update.</p>
      <div class="mt-6 flex flex-col gap-5">${bars}</div>
    </div>
  </div>`;
}

function costs(data: SiteData): string {
  const central = (ls: string): Condition | undefined => pick(data, { family: 'centralised', model: 'logreg_cv', label_set: ls, split_scheme: 'subject' });
  const fed = (ls: string): Condition | undefined => pick(data, { family: 'federated', label_set: ls, split_scheme: 'subject', client_scheme: 'natural' });
  const p1 = privacyAt(data, 'cn-ad', 1);
  const p0 = privacyAt(data, 'cn-ad', null);
  const p10 = privacyAt(data, 'cn-ad', 10);
  const card = (n: string, i: string, title: string, value: string, body: string): string => `<div class="card">
    <div class="flex items-center justify-between"><span class="text-teal">${icon(i)}</span><span class="font-mono text-xs text-slate">Cost ${n}</span></div>
    <h3 class="mt-4 font-sans text-base font-medium">${title}</h3>
    <p class="num mt-1 text-2xl font-semibold text-ad">${value}</p>
    <p class="mt-3 text-sm text-slate">${body}</p></div>`;
  return `<div class="grid gap-6 md:grid-cols-3">
    ${card('01', 'hub', 'Federating over real sites', `${points(fed('cn-mci-ad')?.accuracy, central('cn-mci-ad')?.accuracy)} / ${points(fed('cn-ad')?.accuracy, central('cn-ad')?.accuracy)}`,
      `Three-class / CN vs AD accuracy of federated training over ${data.partition.n_clients} real-site clients, against a tuned logistic regression trained centrally on the same subjects.`)}
    ${card('02', 'lock', 'Subject-level privacy at ε = 1', points(p1?.accuracy, p0?.accuracy),
      `CN vs AD accuracy of the private model against its non-private twin (${pct(p1?.accuracy)} vs ${pct(p0?.accuracy)}).`)}
    ${card('03', 'psychology', 'Explanations under privacy', `${fixed(p0?.explanation_agreement)} → ${fixed(p10?.explanation_agreement)}`,
      'Spearman agreement of region attributions with the reference model, non-private versus ε = 10, while accuracy holds.')}
  </div>`;
}

function render(data: SiteData): string {
  const three = bestCentralised(data, 'cn-mci-ad', 'subject');
  const cnad = bestCentralised(data, 'cn-ad', 'subject');
  const { v1, v2 } = data.history;
  const triad = (c: string, label: string): string =>
    `<span class="flex items-center gap-1.5"><span class="h-2 w-2 rounded-full ${c}"></span><span class="font-mono text-xs">${label}</span></span>`;
  return `
  <section class="border-b border-hairline bg-[#fbf9f5] py-16">
    <div class="shell flex flex-col gap-8">
      <div class="flex flex-wrap items-center justify-between gap-3">
        <p class="eyebrow">Research paper <span class="text-slate">/ Federated healthcare machine learning</span></p>
        <span class="badge">Open source · MIT</span>
      </div>
      <div class="max-w-4xl">
        <h1 class="text-4xl leading-tight md:text-[54px] md:leading-[62px]">Federated brain MRI, measured honestly.</h1>
        <p class="mt-4 font-serif text-xl leading-8 text-slate">CN / MCI / AD classification from ADNI structural MRI,
          trained without pooling scans across sites, evaluated so that no subject, and under the stricter protocol no
          site, appears in more than one split.</p>
      </div>
      <div class="flex flex-col justify-between gap-6 border-t border-hairline pt-6 md:flex-row md:items-center">
        <div><p class="font-medium">Adit Srivastava, Aditya Raj, Vaibhav Gupta, Hardik Gupta, Agha Alfi Mirza</p>
          <p class="text-sm text-slate">PES University, Bangalore</p></div>
        <div class="flex flex-wrap gap-2">
          <a class="btn btn-primary" href="${href('demo')}">Try the in-browser demo ${icon('arrow_forward')}</a>
          <a class="btn" href="${PAPER}">${icon('description')} Read the paper (PDF)</a>
          <a class="btn" href="${href('integrity')}">Splits and integrity</a>
        </div>
      </div>
      <div class="card flex flex-wrap items-center gap-5 !py-3">
        <span class="font-mono text-xs text-slate">Colour-blind-safe class palette</span>
        ${triad('bg-cn', 'CN · cognitively normal')}${triad('bg-mci', 'MCI · mild cognitive impairment')}${triad('bg-ad', "AD · Alzheimer's disease")}
      </div>
    </div>
  </section>

  <section class="shell mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
    ${kpi('Cohort', String(three.n_subjects), 'ADNI subjects with T1 MRI, one scan each')}
    ${kpi('Sites → clients', `${data.partition.n_sites} → ${data.partition.n_clients}`, 'Real acquisition sites, small ones pooled into federated clients')}
    ${kpi('CN vs AD', pct(cnad.accuracy), `Imaging only, subject-disjoint (AUROC ${fixed(cnad.macro_auroc)}); ${pct(bestCentralised(data, 'cn-ad', 'subject', 'centralised+cov').accuracy)} with age and sex`, 'text-teal')}
    ${kpi('Private CN vs AD', pct(privacyAt(data, 'cn-ad', 1)?.accuracy), `Subject-level DP at ε = 1, δ = ${data.dp_config.delta}`)}
  </section>

  <section class="shell mt-10">
    <div class="notice flex gap-4">
      <span class="mt-0.5">${icon('history_edu')}</span>
      <div><p class="font-mono text-xs uppercase tracking-wider">Integrity note · we corrected ourselves</p>
        <p class="mt-2 text-ink">An earlier version reported <strong class="num text-amber">${pct(v1.accuracy)}</strong> federated accuracy
          by splitting MRI slices rather than subjects, so one patient's scans sat on both sides of the split. Rebuilt on
          subject-disjoint folds it fell to <strong class="num">${pct(v2.accuracy)}</strong>, against a 33.3% chance rate, and a
          second defect in the slice rule explained why. Both are documented.</p>
        <a class="mt-3 inline-block text-sm text-teal hover:underline" href="${href('integrity')}">Read the full audit trail →</a></div>
    </div>
  </section>

  ${section('Dimension law', 'The cost of subject-level privacy is set by the perturbed dimension', dimensionLaw(data))}
  ${section('Evaluation dimensions', 'What we measured across the pipeline', costs(data),
    'Three trade-offs measured on the same subjects and folds: what federating costs, what privacy costs, and what privacy does to explanations.')}

  <section class="shell mt-14">
    <div class="card flex flex-col items-start justify-between gap-4 md:flex-row md:items-center">
      <div><p class="eyebrow">In-browser demo</p>
        <h2 class="mt-1 text-2xl">Run a classifier without uploading anything</h2>
        <p class="mt-1 text-slate">The demo model runs locally through WebAssembly. It is the earlier slice model and is labelled as such.</p></div>
      <div class="flex gap-2"><a class="btn btn-primary" href="${href('demo')}">${icon('play_arrow')} Open the demo</a>
        <a class="btn" href="${href('results')}">Results tables</a></div>
    </div>
  </section>

  <section class="shell mt-14 grid gap-6 lg:grid-cols-2">
    <div class="card bg-muted text-sm text-slate">
      <p class="font-mono text-xs uppercase tracking-wider text-ink">ADNI acknowledgement</p>
      <p class="mt-2">Data used in the preparation of this work were obtained from the Alzheimer's Disease Neuroimaging
        Initiative (ADNI) database. ADNI investigators contributed to the design and implementation of ADNI and provided
        data, but did not participate in the analysis or writing of this report.</p>
    </div>
    <div class="card">
      <div class="flex items-center justify-between"><p class="font-mono text-xs uppercase tracking-wider">BibTeX</p>
        <button class="btn !px-2 !py-1 text-xs" id="copy-bib" type="button">${icon('content_copy')} Copy</button></div>
      <pre class="mt-3 overflow-x-auto rounded bg-paper p-4 font-mono text-xs leading-5" id="bib">${BIBTEX}</pre>
    </div>
  </section>`;
}

void mountPage('index', render).then(() => {
  const button = document.getElementById('copy-bib');
  button?.addEventListener('click', () => {
    void navigator.clipboard.writeText(BIBTEX).then(() => button.replaceChildren('Copied'));
  });
});
