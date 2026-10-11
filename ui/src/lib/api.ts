/**
 * Client for the region-model API on Render (`backend/app.py`).
 *
 * The API serves the paper's v6 model: subject-level DP-FedAvg at ε = 10 over
 * deepmriprep region volumes. Errors are turned into sentences a visitor can
 * act on; the free Render instance sleeps when idle, so a failed first request
 * usually means "wait for it to wake", and the message says so.
 */

export const API_URL: string =
  (import.meta.env.VITE_API_URL as string | undefined) ?? 'https://ppxfl-api.onrender.com';

export type ClassName = 'CN' | 'MCI' | 'AD';

export interface RegionContribution {
  region: string;
  tissue: string;
  z_score: number;
  contribution: number;
}

export interface ModelCard {
  name: string;
  privacy: { target_epsilon: number; accounted_epsilon: number; delta: number };
  training: { n_subjects: number; n_clients: number };
  evaluation: { accuracy: number; macro_auroc: number; chance_accuracy: number };
  disclaimer: string;
}

export interface RegionPrediction {
  probabilities: Record<ClassName, number>;
  predicted: ClassName;
  top_regions: RegionContribution[];
  model: ModelCard;
}

export const WAKING =
  'Could not reach the API. The free server sleeps when idle and can take up to a minute to wake; try again shortly.';

/** TIV in cm³ from deepmriprep's tiv CSV ("tiv_cm3\n1868.93") or a bare number. */
export function parseTiv(text: string): number {
  const value = Number(text.trim().split(/\s+/).pop());
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error('The TIV file should hold one number in cm³, as deepmriprep writes it.');
  }
  return value;
}

/** The API's error body as one sentence: a string detail, or the first validation message. */
export function errorMessage(status: number, body: unknown): string {
  if (status === 429) return 'Too many requests; try again in a minute.';
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail) && typeof detail[0]?.msg === 'string') return detail[0].msg;
  return `The API returned an error (HTTP ${status}).`;
}

export async function predictRegions(
  volumesCsv: string,
  tivCm3: number,
  fetchFn: typeof fetch = fetch,
  base: string = API_URL,
): Promise<RegionPrediction> {
  let response: Response;
  try {
    response = await fetchFn(`${base}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ volumes_csv: volumesCsv, tiv_cm3: tivCm3 }),
    });
  } catch {
    throw new Error(WAKING);
  }
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) throw new Error(errorMessage(response.status, body));
  return body as RegionPrediction;
}

/** Ping the API so a sleeping instance starts waking while the visitor reads. */
export function wake(fetchFn: typeof fetch = fetch, base: string = API_URL): void {
  void fetchFn(`${base}/health`).catch(() => undefined);
}
