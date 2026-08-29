/**
 * The words the demo says back.
 *
 * Kept out of the DOM layer so the phrasing is testable: this is the part of
 * the demo most likely to overclaim, and the honesty of the reply matters more
 * than the layout around it.
 */

import { CLASS_LABELS } from './preprocess';
import {
  CHANCE,
  describeConfidence,
  formatPercent,
  type Prediction,
} from './predictions';

export interface ReplyLine {
  kind: 'text' | 'caveat';
  text: string;
}

/**
 * Compose the assistant's reply for one classified slice.
 *
 * The caveat is not optional decoration. A top class near the chance rate is
 * reported as such, and the reply always states that this is a research model.
 */
export function buildReply(
  predictions: readonly Prediction[],
  referenceAccuracy: number | null,
): ReplyLine[] {
  const [top] = predictions;
  const confidence = describeConfidence(top.probability);
  const lines: ReplyLine[] = [
    {
      kind: 'text',
      text: `Most likely ${CLASS_LABELS[top.label]} (${top.label}) at ${formatPercent(
        top.probability,
      )}. Confidence is ${confidence} for a three-class task where chance is ${formatPercent(
        CHANCE,
      )}.`,
    },
  ];

  if (top.probability < CHANCE + 0.05) {
    lines.push({
      kind: 'caveat',
      text: 'The three classes came out near-uniform, so this slice carries no usable signal for the model. Treat it as no prediction.',
    });
  }

  if (referenceAccuracy !== null) {
    lines.push({
      kind: 'text',
      text: `For context, this model averages ${formatPercent(
        referenceAccuracy,
        1,
      )} accuracy on held-out subjects it never trained on.`,
    });
  }

  lines.push({
    kind: 'caveat',
    text: 'Research demonstration only. This is not a medical device and must not be used for diagnosis.',
  });

  return lines;
}

/** Message shown when a dropped file is not an image the browser can decode. */
export function unsupportedFileMessage(fileName: string): string {
  return `I could not read ${fileName}. Drop a PNG or JPEG of a single axial MRI slice.`;
}

/** Opening message, shown before the visitor does anything. */
export const GREETING =
  'Drop an axial MRI slice below and I will classify it as CN, MCI or AD. Everything runs in this browser tab; the image is never uploaded.';
