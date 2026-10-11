import { describe, expect, it, vi } from 'vitest';

import { API_URL, errorMessage, parseTiv, predictRegions, WAKING, wake } from '../src/lib/api';

const ok = (body: unknown): Response => new Response(JSON.stringify(body), { status: 200 });

describe('parseTiv', () => {
  it('reads deepmriprep tiv files and bare numbers', () => {
    expect(parseTiv('tiv_cm3\n1868.93\n')).toBeCloseTo(1868.93);
    expect(parseTiv(' 1500 ')).toBe(1500);
  });

  it('rejects anything that is not one positive number', () => {
    expect(() => parseTiv('tiv_cm3\n')).toThrow(/one number/);
    expect(() => parseTiv('-3')).toThrow(/one number/);
  });
});

describe('errorMessage', () => {
  it('turns API error bodies into a sentence', () => {
    expect(errorMessage(429, null)).toMatch(/Too many requests/);
    expect(errorMessage(422, { detail: '3 model regions missing' })).toBe('3 model regions missing');
    expect(errorMessage(422, { detail: [{ msg: 'Input should be less than 2600' }] })).toBe(
      'Input should be less than 2600',
    );
    expect(errorMessage(500, { detail: [{}] })).toBe('The API returned an error (HTTP 500).');
    expect(errorMessage(502, null)).toBe('The API returned an error (HTTP 502).');
  });
});

describe('predictRegions', () => {
  it('posts the CSV and TIV and returns the prediction', async () => {
    const fetchFn = vi.fn().mockResolvedValue(ok({ predicted: 'CN' }));
    const result = await predictRegions('ROIid\n4', 1500, fetchFn, 'https://api.test');
    expect(result.predicted).toBe('CN');
    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe('https://api.test/predict');
    expect(JSON.parse(init.body)).toEqual({ volumes_csv: 'ROIid\n4', tiv_cm3: 1500 });
  });

  it('reports validation errors from the API', async () => {
    const fetchFn = vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: 'bad csv' }), { status: 422 }));
    await expect(predictRegions('x', 1500, fetchFn)).rejects.toThrow('bad csv');
  });

  it('copes with a non-JSON error body', async () => {
    const fetchFn = vi.fn().mockResolvedValue(new Response('<html>', { status: 502 }));
    await expect(predictRegions('x', 1500, fetchFn)).rejects.toThrow('HTTP 502');
  });

  it('explains a sleeping server when the request cannot be made', async () => {
    const fetchFn = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));
    await expect(predictRegions('x', 1500, fetchFn)).rejects.toThrow(WAKING);
  });
});

describe('wake', () => {
  it('pings /health and swallows failures', async () => {
    const fetchFn = vi.fn().mockRejectedValue(new Error('offline'));
    wake(fetchFn);
    await Promise.resolve();
    expect(fetchFn).toHaveBeenCalledWith(`${API_URL}/health`);
  });
});
