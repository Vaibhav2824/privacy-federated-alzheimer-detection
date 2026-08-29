/**
 * onnxruntime-web session wrapper.
 *
 * Inference runs entirely in the visitor's browser: the image they drop never
 * leaves the machine and no request carries it anywhere. That is the point of
 * shipping the model rather than an API.
 */

import * as ort from 'onnxruntime-web';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import wasmLoaderUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';

import { INPUT_SIZE } from './preprocess';
import { softmax, toPredictions, type Prediction } from './predictions';

/** Path the quantised model is served from, relative to the site base. */
export const MODEL_URL = `${import.meta.env.BASE_URL}model/ppxfl_resnet50.onnx`;

// onnxruntime resolves its wasm binaries at runtime by bare filename, which
// resolves against the page URL and 404s both under the dev server and under a
// GitHub Pages project path. Handing it the URLs Vite emitted removes the guess.
ort.env.wasm.wasmPaths = {
  wasm: wasmUrl,
  mjs: wasmLoaderUrl,
};
// Single-threaded: cross-origin isolation headers are not available on Pages,
// so SharedArrayBuffer (and therefore multi-threaded wasm) is not an option.
ort.env.wasm.numThreads = 1;

export interface Classifier {
  classify(tensorData: Float32Array): Promise<Prediction[]>;
}

/**
 * Wrap an ONNX session so callers deal in probabilities, not tensors.
 *
 * Exported separately from `loadClassifier` so tests can drive it with a fake
 * session instead of a 24MB download.
 */
export function createClassifier(
  session: Pick<ort.InferenceSession, 'run' | 'inputNames' | 'outputNames'>,
): Classifier {
  return {
    async classify(tensorData: Float32Array): Promise<Prediction[]> {
      const expected = INPUT_SIZE * INPUT_SIZE;
      if (tensorData.length !== expected) {
        throw new Error(`Expected ${expected} input values, received ${tensorData.length}`);
      }
      const tensor = new ort.Tensor('float32', tensorData, [1, 1, INPUT_SIZE, INPUT_SIZE]);
      const outputs = await session.run({ [session.inputNames[0]]: tensor });
      const logits = outputs[session.outputNames[0]].data as Float32Array;
      return toPredictions(softmax(logits));
    },
  };
}

/** Download and initialise the shipped model. */
export async function loadClassifier(modelUrl: string = MODEL_URL): Promise<Classifier> {
  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  return createClassifier(session);
}
