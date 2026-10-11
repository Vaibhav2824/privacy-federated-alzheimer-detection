import { API_URL, parseTiv, predictRegions, wake, type RegionPrediction } from '../lib/api';
import { loadClassifier } from '../lib/inference';
import { pct, type SiteData } from '../lib/site';
import { mountChat } from '../ui/chat';
import { href, icon, mountPage, pageHead } from '../ui/layout';

const EXAMPLE = `${import.meta.env.BASE_URL}examples/`;
const CLASS_BAR: Record<string, string> = { CN: 'bg-cn', MCI: 'bg-mci', AD: 'bg-ad' };
const CLASS_NAME: Record<string, string> = {
  CN: 'Cognitively normal', MCI: 'Mild cognitive impairment', AD: "Alzheimer's disease",
};

const DEEPMRIPREP = `pip install deepmriprep
python -c "from deepmriprep import run_preprocess; run_preprocess(input_paths=['t1.nii.gz'], output_dir='out', outputs=['neuromorphometrics_volumes', 'tiv'], dir_format='flat')"`;

/** Escape text from the API before it goes into markup. */
const esc = (text: string): string =>
  text.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

function regionSection(data: SiteData): string {
  return `
  <section class="shell mt-8 grid gap-6 lg:grid-cols-[1fr_1.3fr]">
    <div class="flex flex-col gap-6">
      <div class="card">
        <p class="eyebrow">The paper's model · server API</p>
        <h2 class="mt-2 font-serif text-2xl">Classify region volumes</h2>
        <p class="mt-2 text-sm text-slate">The v6 region model trained with subject-level DP-FedAvg at ε = 10. It reads the
          region volumes deepmriprep measures from a T1 scan, so the scan itself never leaves your machine; only
          134 numbers and the intracranial volume are sent.</p>
        <div class="mt-5 flex flex-col gap-3 text-sm">
          <label class="flex flex-col gap-1">Region volumes CSV <span class="font-mono text-xs text-slate">neuromorphometrics_volumes….csv</span>
            <input id="vol-file" type="file" accept=".csv,text/csv" class="rounded border border-hairline bg-surface p-2"></label>
          <label class="flex flex-col gap-1">Intracranial volume <span class="font-mono text-xs text-slate">tiv….csv, or type cm³</span>
            <span class="flex gap-2"><input id="tiv-file" type="file" accept=".csv,text/csv" class="min-w-0 flex-1 rounded border border-hairline bg-surface p-2">
            <input id="tiv-value" type="number" min="700" max="2600" step="any" placeholder="cm³" class="w-28 rounded border border-hairline bg-surface p-2"></span></label>
        </div>
        <div class="mt-5 flex flex-wrap gap-2">
          <button id="run-api" type="button" class="btn btn-primary">${icon('play_arrow')} Classify</button>
          <button id="run-example" type="button" class="btn">${icon('neurology')} Try the MNI152 template</button>
        </div>
        <p id="api-status" class="mt-3 min-h-6 font-mono text-xs text-slate" role="status"></p>
      </div>
      <div class="card bg-muted text-sm">
        <p class="font-medium">Getting the inputs from your own T1 scan</p>
        <pre class="mt-3 overflow-x-auto rounded bg-paper p-3 font-mono text-[11px] leading-5">${DEEPMRIPREP}</pre>
        <p class="mt-3 text-slate">About a minute per scan on a GPU. The example is deepmriprep's output for the public
          MNI152 template, an average healthy adult brain, so it contains no ADNI data.</p>
      </div>
    </div>
    <div class="card" id="api-result" aria-live="polite">
      <p class="font-serif text-2xl">Result</p>
      <p class="mt-2 text-sm text-slate">Choose a file or try the template. The model is evaluated at
        ${pct(data.conditions.find((c) => c.family === 'privacy' && c.label_set === 'cn-mci-ad' && c.target_epsilon === 10)?.accuracy)}
        three-class accuracy under subject-disjoint cross-validation.</p>
    </div>
  </section>`;
}

function result(out: RegionPrediction): string {
  const bars = (['CN', 'MCI', 'AD'] as const).map((c) => `<div class="flex flex-col gap-1">
      <div class="flex justify-between text-sm"><span class="flex items-center gap-2"><span class="h-2 w-2 rounded-full ${CLASS_BAR[c]}"></span>${CLASS_NAME[c]} (${c})</span>
        <span class="num font-semibold">${pct(out.probabilities[c])}</span></div>
      <div class="track"><span class="fill ${CLASS_BAR[c]}" style="width:${out.probabilities[c] * 100}%"></span></div></div>`).join('');
  const rows = out.top_regions.map((r) => `<tr class="border-t border-hairline">
      <td class="p-2">${esc(r.region)} <span class="font-mono text-[11px] text-slate">${esc(r.tissue)}</span></td>
      <td class="num p-2 text-right">${r.z_score.toFixed(2)}</td>
      <td class="num p-2 text-right ${r.contribution >= 0 ? 'text-teal' : 'text-ad'}">${r.contribution >= 0 ? '+' : '−'}${Math.abs(r.contribution).toFixed(3)}</td></tr>`).join('');
  const m = out.model;
  return `<div class="flex items-center justify-between"><p class="font-serif text-2xl">Predicted class probabilities</p>
      <span class="badge">${out.predicted}</span></div>
    <div class="mt-5 flex flex-col gap-4">${bars}</div>
    <p class="mt-8 font-serif text-xl">Regions that drove the prediction</p>
    <p class="mt-1 text-sm text-slate">z-score: how far each region's volume sits from the training cohort's mean.
      Contribution: its push towards the predicted class, relative to an average cohort member.</p>
    <div class="mt-3 overflow-x-auto"><table class="w-full text-sm"><thead class="text-left font-mono text-xs uppercase tracking-wider text-slate">
      <tr><th class="p-2">Region</th><th class="p-2 text-right">z-score</th><th class="p-2 text-right">Contribution</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="mt-6 rounded bg-muted p-3 text-xs text-slate">${esc(m.name)}: accounted ε = ${m.privacy.accounted_epsilon.toFixed(2)},
      δ = ${m.privacy.delta}; trained on ${m.training.n_subjects} ADNI subjects across ${m.training.n_clients} site clients;
      cross-validated accuracy ${pct(m.evaluation.accuracy)} (chance ${pct(m.evaluation.chance_accuracy)}). ${esc(m.disclaimer)}</p>`;
}

function legacySection(data: SiteData): string {
  return `
  <section class="shell mt-16">
    <p class="eyebrow">In-browser inference · ONNX Runtime Web</p>
    <h2 class="mt-2 text-2xl md:text-3xl">The legacy slice classifier</h2>
    <div class="notice mt-4 flex gap-4">
      <span class="mt-0.5">${icon('warning')}</span>
      <p class="text-ink">This is the earlier v2 ResNet50 slice model, not the paper's model. On held-out subjects it scored
        <strong class="num">${pct(data.history.v2.accuracy)}</strong>, close to the 33.3% chance rate, for the reason set out on the
        <a class="text-teal underline" href="${href('integrity')}">Integrity</a> page. It is kept to show fully private inference:
        the image is classified inside this tab and never sent anywhere.</p>
    </div>
    <div class="card mt-6"><p class="font-serif text-2xl">Classify a slice</p>
      <p class="mt-1 text-sm text-slate">PNG or JPEG of a single axial slice.</p>
      <div id="chat-root" class="mt-4"></div></div>
  </section>`;
}

function render(data: SiteData): string {
  return `
  ${pageHead('Live demo · region model on Render, slice model in your browser', 'Try the models',
    'Run the paper\'s privacy-preserving region model on deepmriprep region volumes, or the legacy slice model entirely inside this tab.',
    `<span class="badge shrink-0">${icon('shield', '!text-sm')} No scan is ever uploaded</span>`)}
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

function wireRegionModel(): void {
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
    set(`Contacting ${API_URL.replace('https://', '')}… the free server can take up to a minute to wake.`);
    try {
      const [csv, tiv] = await load();
      const out = await predictRegions(csv, tiv);
      if (panel) panel.innerHTML = result(out);
      set('Done.');
    } catch (error) {
      set(error instanceof Error ? error.message : String(error), true);
    } finally {
      buttons.forEach((b) => (b.disabled = false));
    }
  }

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
  wireRegionModel();
  const root = document.getElementById('chat-root');
  // The reference quoted back is the slice model's own held-out accuracy.
  if (root) mountChat(root, { classifier: loadClassifier(), referenceAccuracy: data.history.v2.accuracy });
});
