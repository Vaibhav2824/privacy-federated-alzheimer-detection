/**
 * Client for the region-model API on Render (`backend/app.py`).
 *
 * The API serves the v8 private model: 27 a-priori features (20 deepmriprep
 * region volumes plus optional clinical values), 84 parameters, subject-level
 * DP-FedAvg at ε = 1. Errors are turned into sentences a visitor can
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
  privacy: { target_epsilon: number; accounted_epsilon: number; delta: number; parameters: number };
  training: { n_subjects: number; n_clients: number };
  evaluation: { accuracy: number; macro_auroc: number; chance_accuracy: number; cn_ad_accuracy: number };
  disclaimer: string;
}

export interface RegionPrediction {
  probabilities: Record<ClassName, number>;
  predicted: ClassName;
  top_regions: RegionContribution[];
  not_given: string[];
  model: ModelCard;
}

/** Optional clinical values the model also reads; anything left out is filled with the cohort median. */
export const CLINICAL_FIELDS = [
  { key: 'age', label: 'Age', unit: 'years', min: 40, max: 100 },
  { key: 'sex', label: 'Sex', unit: '1 = male, 2 = female', min: 1, max: 2 },
  { key: 'education', label: 'Education', unit: 'years', min: 0, max: 30 },
  { key: 'apoe4', label: 'APOE ε4 alleles', unit: '0, 1 or 2', min: 0, max: 2 },
  { key: 'amyloid_cl', label: 'Amyloid PET', unit: 'centiloids', min: -100, max: 250 },
  { key: 'csf_ptau_ab42', label: 'CSF p-tau181 / Aβ42', unit: 'ratio', min: 0, max: 1 },
  { key: 'plasma_ptau217', label: 'Plasma p-tau217', unit: 'pg/mL', min: 0.001, max: 50 },
  { key: 'plasma_nfl', label: 'Plasma NfL', unit: 'pg/mL', min: 0.1, max: 1000 },
  { key: 'plasma_gfap', label: 'Plasma GFAP', unit: 'pg/mL', min: 1, max: 5000 },
] as const;

export type Clinical = Partial<Record<(typeof CLINICAL_FIELDS)[number]['key'], number>>;

/** Typed form values to numbers: blanks are left out, anything else must be a number in range. */
export function parseClinical(values: Record<string, string>): Clinical {
  const out: Clinical = {};
  for (const field of CLINICAL_FIELDS) {
    const text = (values[field.key] ?? '').trim();
    if (!text) continue;
    const value = Number(text);
    if (!Number.isFinite(value) || value < field.min || value > field.max) {
      throw new Error(`${field.label} should be a number from ${field.min} to ${field.max} (${field.unit}).`);
    }
    out[field.key] = value;
  }
  return out;
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
  clinical: Clinical,
  fetchFn: typeof fetch = fetch,
  base: string = API_URL,
): Promise<RegionPrediction> {
  let response: Response;
  try {
    response = await fetchFn(`${base}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ volumes_csv: volumesCsv, tiv_cm3: tivCm3, clinical }),
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
