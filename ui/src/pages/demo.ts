import { loadClassifier } from '../lib/inference';
import { pct, type SiteData } from '../lib/site';
import { mountChat } from '../ui/chat';
import { href, icon, mountPage, pageHead } from '../ui/layout';

function render(data: SiteData): string {
  return `
  ${pageHead('In-browser inference · ONNX Runtime Web (WebAssembly)', 'Try the legacy slice classifier',
    'Classify a single MRI slice entirely inside this tab. The image is read into memory, resized and classified locally, and is never sent anywhere.',
    `<span class="badge shrink-0">${icon('shield', '!text-sm')} Runs locally · no upload</span>`)}
  <section class="shell mt-8">
    <div class="notice flex gap-4">
      <span class="mt-0.5">${icon('warning')}</span>
      <div class="text-ink"><p class="font-mono text-xs uppercase tracking-wider text-amber">Read this first · this is not the paper's model</p>
        <p class="mt-2">This is the earlier v2 ResNet50 slice model. On held-out subjects it scored
          <strong class="num">${pct(data.history.v2.accuracy)}</strong>, close to the 33.3% chance rate, for the reason set out on the
          <a class="text-teal underline" href="${href('integrity')}">Integrity</a> page. It is kept to demonstrate private, in-browser inference,
          not as a working classifier. The region model behind the
          <a class="text-teal underline" href="${href('results')}">results</a> needs registration and segmentation that cannot yet run in a browser.</p></div>
    </div>
  </section>
  <section class="shell mt-8 grid gap-6 lg:grid-cols-[1fr_1.4fr]">
    <div class="flex flex-col gap-6">
      <div class="card">
        <div class="flex items-center gap-3"><span class="flex h-9 w-9 items-center justify-center rounded bg-teal text-white">${icon('lock')}</span>
          <div><p class="font-medium">Stays on this device</p><p class="font-mono text-xs text-teal">No upload · no telemetry</p></div></div>
        <p class="mt-4 text-sm text-slate">The model is downloaded once and runs through WebAssembly. Your image never leaves the browser tab.</p>
      </div>
      <div class="card">
        <p class="font-medium">What happens to the image</p>
        <ol class="mt-3 flex list-decimal flex-col gap-2 pl-5 text-sm text-slate">
          <li>Converted to greyscale and resized to 224 × 224.</li>
          <li>Intensities normalised the way the model was trained.</li>
          <li>Classified as CN, MCI or AD; probabilities shown with a plain statement of how much to trust them.</li>
        </ol>
      </div>
      <div class="card bg-muted text-sm">
        <p class="font-mono text-xs uppercase tracking-wider text-amber">Research demonstration · not a medical device</p>
        <p class="mt-2 text-slate">Nothing shown here is a diagnosis or a prognosis, and it must not be used to inform anyone's care.</p>
      </div>
    </div>
    <div class="card"><p class="font-serif text-2xl">Classify a slice</p>
      <p class="mt-1 text-sm text-slate">PNG or JPEG of a single axial slice.</p>
      <div id="chat-root" class="mt-4"></div></div>
  </section>`;
}

void mountPage('demo', render).then((data) => {
  const root = document.getElementById('chat-root');
  if (!data || !root) return;
  // The reference quoted back is the slice model's own held-out accuracy,
  // never the region model's, which this page does not run.
  mountChat(root, { classifier: loadClassifier(), referenceAccuracy: data.history.v2.accuracy });
});
