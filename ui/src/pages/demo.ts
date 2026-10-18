import { API_URL, parseTiv, predictRegions, wake, type ClassName, type RegionPrediction } from '../lib/api';
import { loadClassifier } from '../lib/inference';
import { pct, type SiteData } from '../lib/site';
import { mountChat } from '../ui/chat';
import { href, icon, mountPage, pageHead } from '../ui/layout';
import { animate, ring } from '../ui/motion';

const EXAMPLE = `${import.meta.env.BASE_URL}examples/`;
const CLASSES: ClassName[] = ['CN', 'MCI', 'AD'];
const CLASS_BAR: Record<ClassName, string> = { CN: 'bg-cn', MCI: 'bg-mci', AD: 'bg-ad' };
const CLASS_HEX: Record<ClassName, string> = { CN: '#0072b2', MCI: '#e69f00', AD: '#d55e00' };
const CLASS_TEXT: Record<ClassName, string> = { CN: 'text-cn', MCI: 'text-[#9a6a00]', AD: 'text-ad' };
const CLASS_NAME: Record<ClassName, string> = {
  CN: 'Cognitively normal', MCI: 'Mild cognitive impairment', AD: "Alzheimer's disease",
};

const DEEPMRIPREP = `pip install deepmriprep
python -c "from deepmriprep import run_preprocess; run_preprocess(input_paths=['t1.nii.gz'], output_dir='out', outputs=['neuromorphometrics_volumes', 'tiv'], dir_format='flat')"`;

/** Escape text from the API before it goes into markup. */
const esc = (text: string): string =>
  text.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

const FILE_INPUT = 'w-full rounded-xl border border-hairline bg-paper p-1.5 text-sm text-slate file:mr-3 file:rounded-lg file:border-0 file:bg-ink file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-white hover:file:bg-ink/85';

function steps(): string {
  const step = (n: number, i: string, title: string, body: string): string => `<li class="card spotlight flex items-start gap-4 !p-5">
      <span class="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-mint text-teal">${icon(i, '!text-[20px]')}</span>
      <div><p class="font-mono text-xs text-slate">Step ${n}</p><p class="font-display font-semibold">${title}</p>
        <p class="mt-0.5 text-sm text-slate">${body}</p></div></li>`;
  return `<ol class="shell mt-10 grid gap-4 md:grid-cols-3" data-stagger>
    ${step(1, 'neurology', 'Measure the brain locally', '134 region volumes from a T1 scan, on your own machine.')}
    ${step(2, 'shield_lock', 'Send numbers, not scans', 'Only the volumes and intracranial volume reach the private model.')}
    ${step(3, 'insights', 'See the answer and the why', 'Class probabilities plus the regions that drove them.')}
  </ol>`;
}

function emptyState(data: SiteData): string {
  const acc = data.conditions.find((c) => c.family === 'privacy' && c.label_set === 'cn-mci-ad' && c.target_epsilon === 10)?.accuracy;
  return `<div class="flex h-full flex-col items-center justify-center gap-5 py-10 text-center">
      <div class="relative grid h-40 w-40 place-items-center">
        <div class="absolute inset-0 rounded-full border-2 border-dashed border-hairline"></div>
        <div class="absolute inset-5 rounded-full bg-linear-to-br from-mint/80 to-sky/10 blur-xl"></div>
        <span class="relative text-teal">${icon('neurology', '!text-[56px]')}</span>
      </div>
      <div><p class="font-display text-2xl font-semibold">Ready when you are</p>
        <p class="mx-auto mt-2 max-w-sm text-sm text-slate">Press <b class="text-ink">Run on a sample brain</b> for a one-click result, or upload your own
          deepmriprep output. This model scores <span class="num text-ink">${pct(acc)}</span> three-class accuracy under
          subject-disjoint cross-validation.</p></div>
    </div>`;
}

function loadingState(): string {
  return `<div class="flex h-full flex-col items-center justify-center gap-6 py-10 text-center">
      <div class="relative grid h-40 w-40 place-items-center">
        <span class="absolute inset-0 animate-ping rounded-full bg-teal-bright/20 [animation-duration:2s]"></span>
        <span class="absolute inset-4 animate-spin rounded-full border-4 border-mint border-t-teal-bright [animation-duration:1.2s]"></span>
        <span class="relative text-teal">${icon('neurology', '!text-[48px]')}</span>
      </div>
      <div><p class="font-display text-xl font-semibold">Running the private model…</p>
        <p class="mt-1 text-sm text-slate">The free server can take up to a minute to wake on the first request.</p></div>
      <div class="w-full max-w-sm space-y-2"><div class="skeleton"></div><div class="skeleton w-4/5"></div><div class="skeleton w-3/5"></div></div>
    </div>`;
}

type Mode = 'three' | 'two';

/** Which question the visitor asked: chosen before classifying, switchable after. */
let mode: Mode = 'three';
let last: RegionPrediction | null = null;

const MODES: Record<Mode, { title: string; sub: string; icon: string }> = {
  three: { title: 'Three classes', sub: 'CN / MCI / AD', icon: 'stacked_bar_chart' },
  two: { title: 'Two classes', sub: 'CN vs AD', icon: 'compare_arrows' },
};

function modePicker(): string {
  const option = (m: Mode): string => `<button type="button" data-mode="${m}" aria-pressed="${m === mode}"
      class="flex items-center gap-3 rounded-xl border-2 p-3 text-left transition-colors ${m === mode ? 'border-teal-bright bg-mint/40' : 'border-hairline bg-paper hover:border-teal-bright/50'}">
      <span class="grid h-9 w-9 shrink-0 place-items-center rounded-lg ${m === mode ? 'bg-teal text-white' : 'bg-muted text-slate'}">${icon(MODES[m].icon, '!text-[20px]')}</span>
      <span><span class="block text-sm font-semibold">${MODES[m].title}</span><span class="block font-mono text-xs text-slate">${MODES[m].sub}</span></span></button>`;
  return `<p class="text-sm font-medium">1. Choose what to classify</p>
    <div class="mt-2 grid grid-cols-2 gap-2">${option('three')}${option('two')}</div>`;
}

/** The headline block both modes share: confidence ring, verdict and caption. */
function verdict(cls: ClassName, conf: number, caption: string): string {
  return `<div class="flex flex-col items-center gap-6 sm:flex-row" data-stagger>
      <div class="relative h-40 w-40 shrink-0">${ring(conf, 200, 16, CLASS_HEX[cls])}
        <div class="absolute inset-0 grid place-items-center text-center"><div>
          <p class="num text-3xl font-semibold" data-count>${pct(conf)}</p><p class="font-mono text-[10px] uppercase tracking-wider text-slate">confidence</p></div></div>
      </div>
      <div class="text-center sm:text-left">
        <p class="eyebrow"><span class="live-dot"></span>Model output</p>
        <p class="mt-2 font-display text-4xl font-extrabold tracking-tight md:text-5xl ${CLASS_TEXT[cls]}">${cls}</p>
        <p class="mt-1 text-lg font-medium">${CLASS_NAME[cls]}</p>
        <p class="mt-1 text-xs text-slate">${caption} · research output, not a diagnosis</p>
      </div>
    </div>`;
}

function threeClass(out: RegionPrediction): string {
  const top = out.predicted;
  const segments = CLASSES.map((c) => `<span data-bar="0.3" class="${CLASS_BAR[c]} h-full" style="width:${out.probabilities[c] * 100}%" title="${c} ${pct(out.probabilities[c])}"></span>`).join('');
  const legend = CLASSES.map((c) => `<div class="rounded-xl border p-3 ${c === top ? 'border-transparent bg-paper ring-2' : 'border-hairline'}" style="--tw-ring-color:${CLASS_HEX[c]}">
      <p class="flex items-center gap-1.5 text-xs text-slate"><span class="h-2 w-2 rounded-full ${CLASS_BAR[c]}"></span>${c}</p>
      <p class="num mt-1 text-xl font-semibold ${c === top ? CLASS_TEXT[c] : ''}" data-count="0.3">${pct(out.probabilities[c])}</p>
      <p class="truncate text-[11px] text-slate">${CLASS_NAME[c]}</p></div>`).join('');
  return `${verdict(top, out.probabilities[top], 'Most probable of three classes')}
    <div><div class="flex h-3 overflow-hidden rounded-full bg-muted">${segments}</div>
      <div class="mt-3 grid grid-cols-3 gap-2">${legend}</div></div>`;
}

/**
 * CN vs AD, read off the three-class model: P(AD | CN or AD) = AD / (CN + AD).
 * Only meaningful for a scan already known to be CN or AD; it is not the
 * separately trained binary model the Results page reports.
 */
function twoClass(out: RegionPrediction): string {
  const { CN: cn, AD: ad } = out.probabilities;
  const pAd = ad / (cn + ad);
  const top: ClassName = pAd >= 0.5 ? 'AD' : 'CN';
  const other = top === 'AD' ? 'cognitively normal' : "Alzheimer's disease";
  const seg = (c: ClassName, v: number, delay: number): string =>
    `<span data-bar="${delay}" class="${CLASS_BAR[c]} flex h-full items-center ${c === 'CN' ? 'justify-start pl-3' : 'justify-end pr-3'} overflow-hidden whitespace-nowrap font-mono text-xs font-semibold text-white" style="width:${v * 100}%">${v >= 0.25 ? `${c} ${pct(v)}` : ''}</span>`;
  return `${verdict(top, top === 'AD' ? pAd : 1 - pAd, `Rather than ${other}`)}
    <div><div class="flex h-9 overflow-hidden rounded-full bg-muted">${seg('CN', 1 - pAd, 0.3)}${seg('AD', pAd, 0.4)}</div>
      <div class="mt-2 flex justify-between font-mono text-xs"><span class="text-cn">CN · normal ${pct(1 - pAd)}</span><span class="text-ad">AD · Alzheimer's ${pct(pAd)}</span></div>
      <p class="mt-3 rounded-xl bg-paper p-3 text-xs text-slate">For a scan known to be either cognitively normal or Alzheimer's; MCI is left out.
        Computed from the three-class model as AD ÷ (CN + AD), so it is not the separately trained CN vs AD model on the Results page.</p></div>`;
}

function result(out: RegionPrediction): string {
  const m = out.model;
  const maxAbs = Math.max(...out.top_regions.map((r) => Math.abs(r.contribution)), 1e-9);
  const regions = out.top_regions.map((r, i) => {
    const w = (Math.abs(r.contribution) / maxAbs) * 50;
    const pos = r.contribution >= 0;
    return `<div class="grid grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)_3.5rem] items-center gap-3 text-sm">
        <p class="min-w-0"><span class="block truncate" title="${esc(r.region)}">${esc(r.region)}</span>
          <span class="font-mono text-[11px] text-slate">${esc(r.tissue)} · z ${r.z_score.toFixed(2)}</span></p>
        <div class="relative h-6 rounded-md bg-muted">
          <span class="absolute inset-y-0 left-1/2 w-px bg-ink/30"></span>
          <span data-bar="${0.4 + i * 0.05}" class="absolute inset-y-1 rounded ${pos ? 'left-1/2 bg-linear-to-r from-teal to-teal-bright' : 'right-1/2 bg-linear-to-l from-ad/70 to-ad'}" style="width:${w}%"></span>
        </div>
        <span class="num text-right text-xs font-semibold ${pos ? 'text-teal' : 'text-ad'}">${pos ? '+' : '−'}${Math.abs(r.contribution).toFixed(3)}</span></div>`;
  }).join('');
  const fact = (label: string, value: string): string => `<div class="rounded-xl bg-paper p-3"><p class="font-mono text-[10px] uppercase tracking-wider text-slate">${label}</p>
      <p class="num mt-0.5 text-sm font-semibold">${value}</p></div>`;
  return `<div class="flex flex-col gap-6">
    <div class="flex flex-wrap items-baseline justify-between gap-2">
      <p class="font-display text-lg font-semibold">${MODES[mode].title} <span class="text-slate">${MODES[mode].sub}</span></p>
      ${mode === 'three'
        ? `<span class="badge !bg-muted !text-slate !border-hairline">CV accuracy ${pct(m.evaluation.accuracy)} · chance ${pct(m.evaluation.chance_accuracy)}</span>`
        : '<span class="badge">MCI left out</span>'}</div>
    ${mode === 'three' ? threeClass(out) : twoClass(out)}
    <div class="border-t border-hairline pt-5">
      <div class="flex items-baseline justify-between gap-3"><p class="font-display text-lg font-semibold">Why: regions that drove it</p>
        <p class="hidden font-mono text-[11px] text-slate sm:block"><span class="text-ad">◀ against</span> · <span class="text-teal">towards ▶</span></p></div>
      <p class="mt-1 text-xs text-slate">z: how far a region's volume sits from the training cohort's mean. Bar: its push towards the
        three-class prediction (${out.predicted}), relative to an average cohort member.</p>
      <div class="mt-4 flex flex-col gap-2">${regions}</div>
    </div>
    <div class="grid grid-cols-2 gap-2 sm:grid-cols-4">
      ${fact('Privacy', `ε = ${m.privacy.accounted_epsilon.toFixed(2)}`)}${fact('δ', String(m.privacy.delta))}
      ${fact('Trained on', `${m.training.n_subjects} · ${m.training.n_clients} sites`)}${fact('CV accuracy', `${pct(m.evaluation.accuracy)} / ${pct(m.evaluation.chance_accuracy)}`)}
    </div>
    <p class="text-xs text-slate">${esc(m.name)}. ${esc(m.disclaimer)}</p>
  </div>`;
}

function regionSection(data: SiteData): string {
  return `
  <section class="shell mt-8 grid gap-6 lg:grid-cols-[1fr_1.35fr]">
    <div class="flex flex-col gap-6" data-reveal>
      <div class="card">
        <p class="eyebrow">The paper's model · server API</p>
        <h2 class="mt-2 text-2xl">Classify region volumes</h2>
        <p class="mt-2 text-sm text-slate">The v7 region model trained with subject-level DP-FedAvg at ε = 10. It reads the
          region volumes deepmriprep measures from a T1 scan, so the scan itself never leaves your machine.</p>
        <div id="mode-slot" class="mt-5">${modePicker()}</div>
        <p class="mt-5 text-sm font-medium">2. Run it</p>
        <button id="run-example" type="button" class="btn btn-primary mt-2 h-12 w-full justify-center text-base">${icon('bolt')} Run on a sample brain (MNI152)</button>
        <div class="my-5 flex items-center gap-3 font-mono text-[11px] uppercase tracking-wider text-slate"><span class="h-px flex-1 bg-hairline"></span>or use your own<span class="h-px flex-1 bg-hairline"></span></div>
        <div class="flex flex-col gap-4 text-sm">
          <label class="flex flex-col gap-1.5 font-medium">Region volumes CSV <span class="font-mono text-xs font-normal text-slate">neuromorphometrics_volumes….csv</span>
            <input id="vol-file" type="file" accept=".csv,text/csv" class="${FILE_INPUT}"></label>
          <label class="flex flex-col gap-1.5 font-medium">Intracranial volume <span class="font-mono text-xs font-normal text-slate">tiv….csv, or type cm³</span>
            <span class="flex gap-2"><input id="tiv-file" type="file" accept=".csv,text/csv" class="${FILE_INPUT} min-w-0 flex-1">
            <input id="tiv-value" type="number" min="700" max="2600" step="any" placeholder="cm³" class="w-28 rounded-xl border border-hairline bg-paper px-3 text-sm"></span></label>
        </div>
        <button id="run-api" type="button" class="btn mt-5 w-full justify-center">${icon('play_arrow')} Classify my files</button>
        <p id="api-status" class="mt-3 min-h-6 font-mono text-xs text-slate" role="status"></p>
      </div>
      <details class="card group !p-5 text-sm">
        <summary class="flex cursor-pointer list-none items-center justify-between font-medium">Getting the inputs from your own T1 scan
          <span class="transition-transform group-open:rotate-180">${icon('expand_more')}</span></summary>
        <pre class="mt-3 overflow-x-auto rounded-xl bg-ink p-4 font-mono text-[11px] leading-5 text-mint">${DEEPMRIPREP}</pre>
        <p class="mt-3 text-slate">About a minute per scan on a GPU. The sample is deepmriprep's output for the public
          MNI152 template, an average healthy adult brain, so it contains no ADNI data.</p>
      </details>
    </div>
    <div class="card beam min-h-[34rem] !p-7" id="api-result" aria-live="polite" data-reveal>${emptyState(data)}</div>
  </section>`;
}

function legacySection(data: SiteData): string {
  return `
  <section class="shell mt-20" data-reveal>
    <p class="eyebrow">In-browser inference · ONNX Runtime Web</p>
    <h2 class="mt-2 text-2xl md:text-4xl">The legacy slice classifier</h2>
    <div class="notice mt-4 flex gap-4">
      <span class="mt-0.5">${icon('warning')}</span>
      <p class="text-ink">This is the earlier v2 ResNet50 slice model, not the paper's model. On held-out subjects it scored
        <strong class="num">${pct(data.history.v2.accuracy)}</strong>, close to the 33.3% chance rate, for the reason set out on the
        <a class="text-teal underline" href="${href('integrity')}">Integrity</a> page. It is kept to show fully private inference:
        the image is classified inside this tab and never sent anywhere.</p>
    </div>
    <div class="card mt-6"><p class="font-display text-xl font-semibold">Classify a slice</p>
      <p class="mt-1 text-sm text-slate">PNG or JPEG of a single axial slice.</p>
      <div id="chat-root" class="mt-4"></div></div>
  </section>`;
}

function render(data: SiteData): string {
  return `
  ${pageHead('Live demo · region model on Render, slice model in your browser', 'Try the model, live',
    'Run the paper\'s privacy-preserving region model on deepmriprep region volumes, or the legacy slice model entirely inside this tab.',
    `<span class="badge shrink-0 !px-3 !py-1">${icon('shield', '!text-sm')} No scan is ever uploaded</span>`)}
  ${steps()}
  ${regionSection(data)}
  ${legacySection(data)}
  <section class="shell mt-8"><div class="card bg-muted text-sm">
    <p class="font-mono text-xs uppercase tracking-wider text-amber">Research demonstration · not a medical device</p>
    <p class="mt-2 text-slate">Nothing shown here is a diagnosis or a prognosis, and it must not be used to inform anyone's care.</p>
  </div></section>`;
}

async function readText(input: HTMLInputElement): Promise<string | null> {
  const file = input.files?.[0];
  return file ? file.text() : null;
}

function wireRegionModel(data: SiteData): void {
  const status = document.getElementById('api-status');
  const panel = document.getElementById('api-result');
  const buttons = [...document.querySelectorAll<HTMLButtonElement>('#run-api, #run-example')];
  const set = (text: string, error = false): void => {
    if (!status) return;
    status.textContent = text;
    status.classList.toggle('text-danger', error);
  };

  async function run(load: () => Promise<[string, number]>): Promise<void> {
    buttons.forEach((b) => (b.disabled = true));
    set(`Contacting ${API_URL.replace('https://', '')}…`);
    try {
      const [csv, tiv] = await load();
      if (panel) panel.innerHTML = loadingState();
      const out = await predictRegions(csv, tiv);
      if (panel) {
        last = out;
        panel.innerHTML = result(out);
        animate(panel, true);
        if (window.innerWidth < 1024) panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
      set('Done.');
    } catch (error) {
      if (panel) panel.innerHTML = emptyState(data);
      set(error instanceof Error ? error.message : String(error), true);
    } finally {
      buttons.forEach((b) => (b.disabled = false));
    }
  }

  document.getElementById('mode-slot')?.addEventListener('click', (event) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>('button[data-mode]');
    if (!button) return;
    mode = button.dataset.mode as Mode;
    const slot = document.getElementById('mode-slot');
    if (slot) slot.innerHTML = modePicker();
    if (panel && last) {
      panel.innerHTML = result(last);
      animate(panel, true);
    }
  });

  document.getElementById('run-example')?.addEventListener('click', () => void run(async () => {
    const [csv, tiv] = await Promise.all(
      ['mni152_neuromorphometrics_volumes.csv', 'mni152_tiv.csv'].map((f) => fetch(`${EXAMPLE}${f}`).then((r) => r.text())),
    );
    return [csv, parseTiv(tiv)];
  }));

  document.getElementById('run-api')?.addEventListener('click', () => void run(async () => {
    const csv = await readText(document.getElementById('vol-file') as HTMLInputElement);
    if (!csv) throw new Error('Choose the region volumes CSV first.');
    const typed = (document.getElementById('tiv-value') as HTMLInputElement).value;
    const tivText = await readText(document.getElementById('tiv-file') as HTMLInputElement);
    if (!tivText && !typed) throw new Error('Add the TIV file or type the intracranial volume in cm³.');
    return [csv, parseTiv(tivText ?? typed)];
  }));
}

void mountPage('demo', render).then((data) => {
  if (!data) return;
  wake();
  wireRegionModel(data);
  const root = document.getElementById('chat-root');
  // The reference quoted back is the slice model's own held-out accuracy.
  if (root) mountChat(root, { classifier: loadClassifier(), referenceAccuracy: data.history.v2.accuracy });
});
