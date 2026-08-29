/**
 * The chat surface: DOM glue around the tested logic in src/lib.
 *
 * This module deliberately holds no decision-making of its own. Preprocessing,
 * probability handling and reply wording all live in tested modules; what is
 * here is element construction and event wiring.
 */

import { buildReply, GREETING, unsupportedFileMessage } from '../lib/conversation';
import type { Classifier } from '../lib/inference';
import { INPUT_SIZE, preprocessImageData } from '../lib/preprocess';
import { formatPercent, type Prediction } from '../lib/predictions';

export interface ChatOptions {
  /** Resolves once the model is ready; awaited on the first classification. */
  classifier: Promise<Classifier>;
  /** Held-out accuracy quoted back to the reader, when known. */
  referenceAccuracy: number | null;
}

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

function messageShell(role: 'You' | 'PPXFL', variant: 'user' | 'assistant'): {
  wrapper: HTMLDivElement;
  body: HTMLDivElement;
} {
  const wrapper = element('div', `message message-${variant}`);
  wrapper.append(element('span', 'message-role', role));
  const body = element('div', 'message-body');
  wrapper.append(body);
  return { wrapper, body };
}

function scoreRow(prediction: Prediction): HTMLDivElement {
  const row = element('div', 'score');
  row.append(element('span', undefined, prediction.label));

  const track = element('span', 'score-track');
  const fill = element('span', 'score-fill');
  fill.style.width = `${Math.max(1, prediction.probability * 100).toFixed(1)}%`;
  track.append(fill);
  row.append(track);

  row.append(element('span', 'score-value', formatPercent(prediction.probability)));
  return row;
}

/**
 * Draw the image onto a 224x224 canvas and read back its pixels.
 *
 * The canvas is the only reason this cannot be unit-tested alongside the rest
 * of the pipeline; the arithmetic it feeds is covered in preprocess.test.ts.
 */
function readImagePixels(image: HTMLImageElement): ImageData {
  const canvas = document.createElement('canvas');
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('This browser did not provide a 2D canvas context.');
  }
  context.drawImage(image, 0, 0, INPUT_SIZE, INPUT_SIZE);
  return context.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
}

function loadImage(file: File): Promise<{ image: HTMLImageElement; url: string }> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => resolve({ image, url });
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error(unsupportedFileMessage(file.name)));
    };
    image.src = url;
  });
}

export function mountChat(root: HTMLElement, options: ChatOptions): void {
  const transcript = element('div', 'transcript');
  transcript.setAttribute('role', 'log');
  transcript.setAttribute('aria-live', 'polite');

  const greeting = messageShell('PPXFL', 'assistant');
  greeting.body.append(element('p', undefined, GREETING));
  transcript.append(greeting.wrapper);

  const dropzone = element('div', 'dropzone');
  const input = element('input');
  input.type = 'file';
  input.accept = 'image/png,image/jpeg';
  input.id = 'slice-input';
  input.hidden = true;

  const chooseButton = element('button', 'button button-primary', 'Choose an MRI slice');
  chooseButton.type = 'button';
  chooseButton.addEventListener('click', () => input.click());

  dropzone.append(
    chooseButton,
    element('p', undefined, 'or drop a PNG or JPEG here'),
    input,
  );

  const status = element('p', 'status');
  status.setAttribute('role', 'status');

  root.append(transcript, dropzone, status);

  let busy = false;

  function setStatus(text: string, state: 'idle' | 'busy' | 'error' = 'idle'): void {
    status.textContent = text;
    status.dataset.state = state;
  }

  async function handleFile(file: File): Promise<void> {
    if (busy) return;
    busy = true;
    chooseButton.disabled = true;

    let objectUrl: string | null = null;
    try {
      setStatus('Reading the image.', 'busy');
      const { image, url } = await loadImage(file);
      objectUrl = url;

      const userMessage = messageShell('You', 'user');
      const thumb = element('img', 'thumb');
      thumb.src = url;
      thumb.alt = `Uploaded MRI slice: ${file.name}`;
      userMessage.body.append(thumb);
      transcript.append(userMessage.wrapper);

      const pending = messageShell('PPXFL', 'assistant');
      pending.body.append(element('div', 'skeleton'), element('div', 'skeleton'));
      transcript.append(pending.wrapper);
      transcript.scrollTop = transcript.scrollHeight;

      setStatus('Loading the model and classifying. This runs in your browser.', 'busy');
      const classifier = await options.classifier;
      const pixels = readImagePixels(image);
      const tensor = preprocessImageData(pixels.data, INPUT_SIZE, INPUT_SIZE);
      const predictions = await classifier.classify(tensor);

      pending.body.replaceChildren();
      for (const line of buildReply(predictions, options.referenceAccuracy)) {
        pending.body.append(
          element('p', line.kind === 'caveat' ? 'caveat' : undefined, line.text),
        );
      }
      const scores = element('div', 'scores');
      for (const prediction of predictions) {
        scores.append(scoreRow(prediction));
      }
      pending.body.append(scores);

      setStatus('Done. Drop another slice to try again.');
      transcript.scrollTop = transcript.scrollHeight;
    } catch (error) {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      setStatus(error instanceof Error ? error.message : String(error), 'error');
    } finally {
      busy = false;
      chooseButton.disabled = false;
      input.value = '';
    }
  }

  input.addEventListener('change', () => {
    const file = input.files?.[0];
    if (file) void handleFile(file);
  });

  dropzone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropzone.classList.add('is-active');
  });

  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('is-active'));

  dropzone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropzone.classList.remove('is-active');
    const file = event.dataTransfer?.files?.[0];
    if (file) void handleFile(file);
  });
}
