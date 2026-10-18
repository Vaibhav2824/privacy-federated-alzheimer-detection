import { privacyAt, type SiteData } from '../lib/site';
import { BIBTEX, icon, mountPage, pageHead, REPO, section } from '../ui/layout';

function stages(features: number): string {
  const STAGES: [string, string, string, string][] = [
    ['ADNI T1 MRI', 'T1-weighted', 'One scan per subject, chosen by ADNI preprocessing level and then earliest acquisition; derived mask series excluded.', 'NIfTI'],
    ['Orientation search', '48 candidates', 'Every axis permutation and reflection registered to a whole-head ICBM152 target, fitted per series and array shape.', 'Measured, not read from headers'],
    ['Skull stripping', 'deepbet', 'Run after orientation, because the stripper leaves face and neck behind when a scan lies on its side.', 'Brain mask'],
    ['MNI152 registration and QC', 'Affine', 'Mutual-information affine registration to MNI152; a per-subject correlation score excludes failures by a stated rule.', 'QC-gated cohort'],
    ['Region volumes (deepmriprep)', `${features} regions`, 'Network tissue segmentation and nonlinear warping; the Neuromorphometrics atlas is warped back and each region’s grey-matter (or ventricular CSF) volume is divided by intracranial volume.', 'Named anatomy'],
    ['DP-FedAvg and audit', 'Subject-level', 'Federated over real sites with per-subject clipping and Gaussian noise, PRV privacy accounting, membership inference and region attribution.', 'ε, δ certified'],
  ];
  return `<div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-6">${STAGES.map(([title, tag, body, out], i) => `
    <div class="card flex flex-col !p-4 ${i === STAGES.length - 1 ? 'border-teal bg-teal/[0.04]' : ''}">
      <div class="flex items-center justify-between gap-2"><span class="num rounded-xl bg-muted px-1.5 text-xs">0${i + 1}</span>
        <span class="badge !text-[10px]">${tag}</span></div>
      <h3 class="mt-3 font-display text-lg leading-6">${title}</h3>
      <p class="mt-2 flex-1 text-xs leading-5 text-slate">${body}</p>
      <p class="mt-3 border-t border-hairline pt-2 font-mono text-[11px] text-slate">${out}</p></div>`).join('')}</div>`;
}

function guarantees(data: SiteData): string {
  const items: [string, string, string][] = [
    ['Subject-disjoint folds', 'No subject leakage', 'Each subject contributes one scan and sits in exactly one of five folds, so no brain is ever on both sides of a split. Three seeds per configuration.'],
    ['Site-disjoint folds', 'Unseen scanners', 'Every task is also evaluated with whole acquisition sites held out, the setting a federated model is deployed into.'],
    ['Bounded per-subject sensitivity', `Clip C = ${data.dp_config.clip_norm}`, `Each subject's update is clipped before aggregation and Gaussian noise is added once at the server; a PRV accountant certifies (ε, δ = ${data.dp_config.delta}) over all rounds.`],
  ];
  return items.map(([title, tag, body], i) => `<div class="flex gap-4 ${i ? 'border-t border-hairline pt-5' : ''}">
    <span class="num flex h-7 w-7 shrink-0 items-center justify-center rounded border border-hairline text-xs">1.${i + 1}</span>
    <div><p class="font-medium">${title} <span class="badge ml-1 !text-[10px]">${tag}</span></p><p class="mt-1 text-sm text-slate">${body}</p></div></div>`).join('');
}

function hyperparameters(data: SiteData, dimension: number | undefined): string {
  const d = data.dp_config;
  const rows: [string, string][] = [
    ['Communication rounds', String(d.rounds)],
    ['Local epochs per round', String(d.local_epochs)],
    ['Subject sampling rate (q)', String(d.subject_sample_rate)],
    ['Per-subject clip norm (C)', String(d.clip_norm)],
    ['Target δ', String(d.delta)],
    ['Perturbed dimension, 3-class', dimension === undefined ? '–' : dimension.toLocaleString('en')],
    ['Evaluation', '5 folds × 3 seeds'],
  ];
  return rows.map(([k, v]) => `<div class="flex justify-between border-b border-hairline py-2.5 text-sm last:border-0">
    <span class="text-slate">${k}</span><span class="num font-medium">${v}</span></div>`).join('');
}

const REPRODUCE = `# 1. install the pipeline
pip install -e "ppxfl-alzheimer[dev]"

# 2. fit each series group's orientation against ICBM152
cd ppxfl-alzheimer && python -m src.fit_orientation --data-root ..

# 3. run the pipeline (stages are cached and skippable)
python run_v3.py --stages features centralised federated privacy aggregate figures

# 4. tests, and a check that the paper's numbers match the results
pytest --cov --cov-config=.coveragerc
python tools/texcheck.py && python src/check_paper_numbers.py`;

function render(data: SiteData): string {
  const dimension = privacyAt(data, 'cn-mci-ad', 1)?.perturbed_dimension;
  // Three-class softmax regression perturbs (features + 1) x 3 parameters.
  const features = dimension === undefined ? 0 : dimension / 3 - 1;
  return `
  ${pageHead('Experimental specification · pipeline architecture', 'End-to-end methodology and reproduction',
    'From raw ADNI scans to a federated, privacy-preserving region model: every stage, the guarantees each one enforces, and the commands that rerun it.',
    `<div class="flex flex-wrap gap-2 md:max-w-xs md:justify-end"><span class="badge">${icon('code', '!text-sm')} Open source</span>
      <span class="badge">${icon('verified_user', '!text-sm')} Subject-level DP</span></div>`)}
  ${section('Stages 01 → 06', 'From scanner volume to private federated model', stages(features))}
  <section class="shell mt-14 grid gap-6 lg:grid-cols-[1.5fr_1fr]">
    <div class="card"><h2 class="font-sans text-lg font-semibold">${icon('lock', 'text-teal')} Guarantees enforced</h2>
      <div class="mt-5 flex flex-col gap-5">${guarantees(data)}</div>
      <p class="mt-5 rounded bg-amber-tint p-3 text-xs text-amber">Federation is simulated in one process. Clients exchange only
        clipped, noised updates, but they are not yet separate machines; that is planned work.</p></div>
    <div class="card"><h2 class="font-sans text-lg font-semibold">Hyperparameters</h2>
      <div class="mt-3">${hyperparameters(data, dimension)}</div></div>
  </section>
  ${section('Reproduction', 'Rerun the pipeline', `<div class="overflow-hidden rounded-lg bg-[#0f172a]">
      <div class="flex items-center justify-between border-b border-white/10 px-4 py-2 font-mono text-xs text-white/60">
        <span>bash</span><a class="hover:text-white" href="${REPO}">${REPO.replace('https://', '')}</a></div>
      <pre class="overflow-x-auto p-4 font-mono text-xs leading-6 text-[#e2e8f0]">${REPRODUCE}</pre></div>`,
    'ADNI scans are not redistributed: obtaining them requires a data use agreement with ADNI directly. No scan, derived array, weight or subject-level table is committed.')}
  ${section('Citation', 'Cite this work', `<div class="card"><pre class="overflow-x-auto font-mono text-xs leading-5">${BIBTEX}</pre></div>`)}`;
}

void mountPage('method', render);
