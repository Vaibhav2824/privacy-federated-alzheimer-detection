import { describe, expect, it } from 'vitest';

import {
  CLASS_LABELS,
  CLASS_NAMES,
  INPUT_SIZE,
  normaliseIntensity,
  preprocessImageData,
  resizeNearest,
  rgbaToGreyscale,
} from '../src/lib/preprocess';

function solidRgba(pixelCount: number, r: number, g: number, b: number): Uint8ClampedArray {
  const buffer = new Uint8ClampedArray(pixelCount * 4);
  for (let i = 0; i < pixelCount; i += 1) {
    buffer.set([r, g, b, 255], i * 4);
  }
  return buffer;
}

describe('rgbaToGreyscale', () => {
  it('maps white to 1 and black to 0', () => {
    expect(Array.from(rgbaToGreyscale(solidRgba(1, 255, 255, 255)))).toEqual([1]);
    expect(Array.from(rgbaToGreyscale(solidRgba(1, 0, 0, 0)))).toEqual([0]);
  });

  it('weights the colour channels by luma rather than averaging them', () => {
    const [green] = rgbaToGreyscale(solidRgba(1, 0, 255, 0));
    expect(green).toBeCloseTo(0.587, 3);
  });

  it('ignores the alpha channel', () => {
    const opaque = new Uint8ClampedArray([128, 128, 128, 255]);
    const transparent = new Uint8ClampedArray([128, 128, 128, 0]);
    expect(rgbaToGreyscale(opaque)[0]).toBe(rgbaToGreyscale(transparent)[0]);
  });

  it('produces one value per pixel', () => {
    expect(rgbaToGreyscale(solidRgba(9, 10, 10, 10))).toHaveLength(9);
  });

  it('rejects a buffer that is not whole RGBA pixels', () => {
    expect(() => rgbaToGreyscale(new Uint8ClampedArray(7))).toThrow(/multiple of 4/);
  });
});

describe('normaliseIntensity', () => {
  it('stretches the range to span 0 and 1', () => {
    const out = normaliseIntensity(new Float32Array([0.2, 0.4, 0.6]));
    expect(out[0]).toBe(0);
    expect(out[1]).toBeCloseTo(0.5, 6);
    expect(out[2]).toBe(1);
  });

  it('preserves the relative ordering of pixels', () => {
    const out = normaliseIntensity(new Float32Array([0.9, 0.1, 0.5]));
    expect(out[0]).toBeGreaterThan(out[2]);
    expect(out[2]).toBeGreaterThan(out[1]);
  });

  it('returns zeros for a flat image instead of dividing by zero', () => {
    const out = normaliseIntensity(new Float32Array([0.5, 0.5, 0.5]));
    expect(Array.from(out)).toEqual([0, 0, 0]);
  });

  it('passes an empty image straight through', () => {
    expect(normaliseIntensity(new Float32Array(0))).toHaveLength(0);
  });
});

describe('resizeNearest', () => {
  it('resizes to the model input size by default', () => {
    const out = resizeNearest(new Float32Array(16), 4, 4);
    expect(out).toHaveLength(INPUT_SIZE * INPUT_SIZE);
  });

  it('leaves an already-correct image untouched', () => {
    const source = new Float32Array([0, 0.25, 0.5, 0.75]);
    expect(Array.from(resizeNearest(source, 2, 2, 2))).toEqual([0, 0.25, 0.5, 0.75]);
  });

  it('upsamples by repeating source pixels', () => {
    const out = resizeNearest(new Float32Array([0, 1, 2, 3]), 2, 2, 4);
    expect(Array.from(out.slice(0, 4))).toEqual([0, 0, 1, 1]);
    expect(Array.from(out.slice(12, 16))).toEqual([2, 2, 3, 3]);
  });

  it('downsamples without reading past the source edge', () => {
    const source = Float32Array.from({ length: 9 }, (_, i) => i);
    const out = resizeNearest(source, 3, 3, 2);
    expect(out).toHaveLength(4);
    expect(Math.max(...out)).toBeLessThanOrEqual(8);
  });

  it('handles a non-square source', () => {
    const out = resizeNearest(new Float32Array(8), 4, 2, 2);
    expect(out).toHaveLength(4);
  });

  it('rejects dimensions that do not match the buffer', () => {
    expect(() => resizeNearest(new Float32Array(5), 2, 2, 2)).toThrow(/does not match/);
  });

  it('rejects a zero-sized source', () => {
    expect(() => resizeNearest(new Float32Array(0), 0, 4, 2)).toThrow(/Invalid source/);
  });
});

describe('preprocessImageData', () => {
  it('produces exactly the tensor the model expects', () => {
    const out = preprocessImageData(solidRgba(64, 10, 10, 10), 8, 8);
    expect(out).toHaveLength(INPUT_SIZE * INPUT_SIZE);
  });

  it('normalises so the output spans the unit interval', () => {
    const rgba = new Uint8ClampedArray(4 * 4 * 4);
    for (let i = 0; i < 16; i += 1) {
      const value = i * 16;
      rgba.set([value, value, value, 255], i * 4);
    }
    const out = preprocessImageData(rgba, 4, 4);
    expect(Math.min(...out)).toBe(0);
    expect(Math.max(...out)).toBe(1);
  });
});

describe('class metadata', () => {
  it('lists the three diagnostic classes in the model output order', () => {
    expect(CLASS_NAMES).toEqual(['CN', 'MCI', 'AD']);
  });

  it('gives every class a readable expansion', () => {
    for (const name of CLASS_NAMES) {
      expect(CLASS_LABELS[name].length).toBeGreaterThan(0);
    }
  });
});
