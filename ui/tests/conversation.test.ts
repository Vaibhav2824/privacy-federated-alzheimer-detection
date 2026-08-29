import { describe, expect, it } from 'vitest';

import {
  GREETING,
  buildReply,
  unsupportedFileMessage,
} from '../src/lib/conversation';
import type { Prediction } from '../src/lib/predictions';

function predictions(cn: number, mci: number, ad: number): Prediction[] {
  return (
    [
      { label: 'CN', probability: cn },
      { label: 'MCI', probability: mci },
      { label: 'AD', probability: ad },
    ] as Prediction[]
  ).sort((a, b) => b.probability - a.probability);
}

describe('buildReply', () => {
  it('names the most likely class and its probability', () => {
    const [first] = buildReply(predictions(0.1, 0.15, 0.75), null);
    expect(first.text).toContain('(AD)');
    expect(first.text).toContain('75.0%');
  });

  it('states the chance rate alongside the confidence', () => {
    const [first] = buildReply(predictions(0.8, 0.1, 0.1), null);
    expect(first.text).toContain('33.3%');
    expect(first.text).toContain('strong');
  });

  it('always closes with the not-a-medical-device caveat', () => {
    const lines = buildReply(predictions(0.8, 0.1, 0.1), null);
    expect(lines[lines.length - 1]).toEqual({
      kind: 'caveat',
      text: expect.stringContaining('not a medical device'),
    });
  });

  it('refuses to present a near-uniform result as a prediction', () => {
    const lines = buildReply(predictions(0.34, 0.33, 0.33), null);
    expect(lines.some((l) => l.kind === 'caveat' && l.text.includes('no prediction'))).toBe(true);
  });

  it('does not add the no-signal caveat to a confident result', () => {
    const lines = buildReply(predictions(0.8, 0.1, 0.1), null);
    expect(lines.some((l) => l.text.includes('no prediction'))).toBe(false);
  });

  it('reports the held-out accuracy when one is available', () => {
    const lines = buildReply(predictions(0.8, 0.1, 0.1), 0.361);
    expect(lines.some((l) => l.text.includes('36.1%'))).toBe(true);
  });

  it('omits the accuracy line when no reference is available', () => {
    const lines = buildReply(predictions(0.8, 0.1, 0.1), null);
    expect(lines.some((l) => l.text.includes('held-out subjects'))).toBe(false);
  });

  it('never phrases the prediction as a diagnosis', () => {
    const spoken = buildReply(predictions(0.95, 0.03, 0.02), 0.361)
      .filter((l) => l.kind === 'text')
      .map((l) => l.text)
      .join(' ')
      .toLowerCase();
    expect(spoken).not.toContain('diagnos');
    expect(spoken).not.toContain('you have');
    expect(spoken).not.toContain('the patient');
  });

  it('explicitly rules out diagnostic use in the caveat', () => {
    const caveats = buildReply(predictions(0.95, 0.03, 0.02), 0.361)
      .filter((l) => l.kind === 'caveat')
      .map((l) => l.text)
      .join(' ');
    expect(caveats).toContain('diagnosis');
  });
});

describe('unsupportedFileMessage', () => {
  it('names the file it could not read and says what is accepted', () => {
    const message = unsupportedFileMessage('scan.dcm');
    expect(message).toContain('scan.dcm');
    expect(message).toContain('PNG or JPEG');
  });
});

describe('GREETING', () => {
  it('promises that the image stays in the browser', () => {
    expect(GREETING).toContain('never uploaded');
  });
});
