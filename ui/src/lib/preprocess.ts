/**
 * Image preprocessing, mirroring what `src/preprocess.py` does at training time.
 *
 * The model was trained on 224x224 single-channel slices scaled to [0, 1], so a
 * dropped browser image has to arrive in exactly that shape or the predictions
 * are meaningless. Everything here is pure so it can be tested without a DOM.
 */

/** Side length the network expects, in pixels. */
export const INPUT_SIZE = 224;

/** Class order of the model's output logits. */
export const CLASS_NAMES = ['CN', 'MCI', 'AD'] as const;

export type ClassName = (typeof CLASS_NAMES)[number];

/** Human-readable expansion of each label, for the reply text. */
export const CLASS_LABELS: Record<ClassName, string> = {
  CN: 'Cognitively normal',
  MCI: 'Mild cognitive impairment',
  AD: "Alzheimer's disease",
};

/**
 * Collapse RGBA bytes to a single greyscale channel in [0, 1].
 *
 * Uses Rec. 601 luma weights rather than a flat mean: MRI slices saved as PNG
 * are already grey, so the two agree, but a colourised screenshot of a slice
 * stays perceptually correct this way.
 */
export function rgbaToGreyscale(rgba: Uint8ClampedArray | Uint8Array): Float32Array {
  if (rgba.length % 4 !== 0) {
    throw new Error(`RGBA buffer length ${rgba.length} is not a multiple of 4`);
  }
  const pixels = new Float32Array(rgba.length / 4);
  for (let i = 0; i < pixels.length; i += 1) {
    const r = rgba[i * 4];
    const g = rgba[i * 4 + 1];
    const b = rgba[i * 4 + 2];
    pixels[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  }
  return pixels;
}

/**
 * Rescale intensities so the darkest pixel maps to 0 and the brightest to 1.
 *
 * MRI slices carry no fixed intensity scale, so training normalised each slice
 * against its own range. A flat image is returned as all zeros instead of
 * dividing by a zero range.
 */
export function normaliseIntensity(pixels: Float32Array): Float32Array {
  if (pixels.length === 0) {
    return pixels;
  }
  let min = pixels[0];
  let max = pixels[0];
  for (const value of pixels) {
    if (value < min) min = value;
    if (value > max) max = value;
  }
  const range = max - min;
  const out = new Float32Array(pixels.length);
  if (range === 0) {
    return out;
  }
  for (let i = 0; i < pixels.length; i += 1) {
    out[i] = (pixels[i] - min) / range;
  }
  return out;
}

/**
 * Nearest-neighbour resize of a single-channel image.
 *
 * Nearest neighbour rather than bilinear so that a slice already at 224x224
 * passes through bit-identical, and so this stays dependency-free.
 */
export function resizeNearest(
  pixels: Float32Array,
  sourceWidth: number,
  sourceHeight: number,
  targetSize = INPUT_SIZE,
): Float32Array {
  if (sourceWidth <= 0 || sourceHeight <= 0) {
    throw new Error(`Invalid source dimensions ${sourceWidth}x${sourceHeight}`);
  }
  if (pixels.length !== sourceWidth * sourceHeight) {
    throw new Error(
      `Pixel buffer of ${pixels.length} does not match ${sourceWidth}x${sourceHeight}`,
    );
  }
  const out = new Float32Array(targetSize * targetSize);
  for (let y = 0; y < targetSize; y += 1) {
    const sourceY = Math.min(sourceHeight - 1, Math.floor((y * sourceHeight) / targetSize));
    for (let x = 0; x < targetSize; x += 1) {
      const sourceX = Math.min(sourceWidth - 1, Math.floor((x * sourceWidth) / targetSize));
      out[y * targetSize + x] = pixels[sourceY * sourceWidth + sourceX];
    }
  }
  return out;
}

/**
 * Full pipeline from raw RGBA image data to the model's NCHW input tensor.
 */
export function preprocessImageData(
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
): Float32Array {
  const grey = rgbaToGreyscale(rgba);
  const resized = resizeNearest(grey, width, height);
  return normaliseIntensity(resized);
}
