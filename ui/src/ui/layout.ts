/**
 * Shared chrome for every page (header, nav, research notice, footer), the
 * site.json loader, and the few markup helpers the pages repeat.
 */

import '../styles.css';

import type { SiteData } from '../lib/site';

export const REPO = 'https://github.com/Vaibhav2824/privacy-federated-alzheimer-detection';
export const PAPER = `${REPO}/blob/main/paper.pdf`;

export const BIBTEX = `@misc{srivastava2026ppxfl,
  title  = {Privacy-Preserving and Explainable Federated AI for
            Alzheimer's and Dementia Detection},
  author = {Srivastava, Adit and Raj, Aditya and Gupta, Vaibhav and
            Gupta, Hardik and Mirza, Agha Alfi},
  year   = {2026},
  note   = {PES University, Bangalore},
  url    = {https://github.com/Vaibhav2824/privacy-federated-alzheimer-detection}
}`;

const BASE = import.meta.env.BASE_URL;

export const PAGES = [
  { id: 'index', label: 'Overview', href: BASE },
  { id: 'demo', label: 'Demo', href: `${BASE}demo.html` },
  { id: 'privacy', label: 'Privacy', href: `${BASE}privacy.html` },
  { id: 'federation', label: 'Federation', href: `${BASE}federation.html` },
  { id: 'results', label: 'Results', href: `${BASE}results.html` },
  { id: 'integrity', label: 'Integrity', href: `${BASE}integrity.html` },
  { id: 'method', label: 'Method', href: `${BASE}method.html` },
] as const;

export type PageId = (typeof PAGES)[number]['id'];

export function href(id: PageId): string {
  return PAGES.find((p) => p.id === id)?.href ?? BASE;
}

export const icon = (name: string, cls = ''): string =>
  `<span class="material-symbols-outlined ${cls}" aria-hidden="true">${name}</span>`;

function header(active: PageId): string {
  const links = PAGES.map(
    (p) =>
      `<a href="${p.href}" class="py-1 text-sm ${
        p.id === active ? 'border-b-2 border-teal font-semibold text-teal' : 'text-slate hover:text-ink'
      }"${p.id === active ? ' aria-current="page"' : ''}>${p.label}</a>`,
  ).join('');
  return `
  <a href="#main" class="sr-only focus:not-sr-only">Skip to content</a>
  <header class="sticky top-0 z-50 border-b border-hairline bg-surface/95 backdrop-blur">
    <div class="shell flex h-14 items-center justify-between gap-4">
      <a href="${BASE}" class="flex items-center gap-2">
        <span class="font-serif text-2xl tracking-tight">PPXFL</span>
        <span class="badge hidden sm:inline-flex">PES University · ADNI cohort</span>
      </a>
      <nav aria-label="Pages" class="hidden items-center gap-5 lg:flex">${links}</nav>
      <div class="flex items-center gap-2">
        <a class="btn !px-3 !py-1.5" href="${PAPER}">${icon('description')}<span class="hidden sm:inline">Paper (PDF)</span></a>
        <a class="btn !px-3 !py-1.5" href="${REPO}">${icon('code')}<span class="hidden sm:inline">GitHub</span></a>
      </div>
    </div>
    <nav aria-label="Pages" class="shell flex gap-4 overflow-x-auto pb-2 lg:hidden">${links}</nav>
    <p class="flex items-center justify-center gap-1.5 border-t border-hairline bg-amber-tint py-1 font-mono text-xs text-amber" role="note">
      ${icon('info', '!text-sm')} Research demonstration. Not a medical device. Nothing here is a diagnosis.
    </p>
  </header>`;
}

const FOOTER = `
  <footer class="mt-16 border-t border-hairline py-8">
    <div class="shell flex flex-col justify-between gap-3 text-sm text-slate md:flex-row">
      <p>PES University, Bangalore. Data from the Alzheimer's Disease Neuroimaging Initiative (ADNI);
        scans are not redistributed and require an ADNI data use agreement. MIT licence.</p>
      <a class="text-teal hover:underline" href="${REPO}">Source, results and paper</a>
    </div>
  </footer>`;

async function loadSite(): Promise<SiteData> {
  const response = await fetch(`${BASE}data/site.json`);
  if (!response.ok) throw new Error(`site.json: HTTP ${response.status}`);
  return (await response.json()) as SiteData;
}

/**
 * Render a page: chrome around whatever `render` returns for the loaded data.
 * Returns the data so a page can wire interaction after the markup exists.
 */
export async function mountPage(
  active: PageId,
  render: (data: SiteData) => string,
): Promise<SiteData | null> {
  const app = document.getElementById('app');
  if (!app) return null;
  try {
    const data = await loadSite();
    app.innerHTML = `${header(active)}<main id="main">${render(data)}</main>${FOOTER}`;
    return data;
  } catch (error) {
    app.innerHTML = `${header(active)}<main id="main" class="shell py-16"><p class="notice">
      The results file could not be loaded (${error instanceof Error ? error.message : String(error)}).
      Rebuild it with <code>python -m src.export_site</code>.</p></main>${FOOTER}`;
    return null;
  }
}

/* ---- markup helpers ---- */

export function pageHead(eyebrow: string, title: string, lede: string, aside = ''): string {
  return `
  <section class="border-b border-hairline bg-[#fbf9f5] py-12">
    <div class="shell flex flex-col gap-4">
      <p class="eyebrow">${eyebrow}</p>
      <div class="flex flex-col justify-between gap-6 md:flex-row md:items-end">
        <div class="max-w-3xl">
          <h1 class="text-3xl leading-tight md:text-5xl">${title}</h1>
          <p class="mt-4 font-serif text-lg leading-8 text-slate">${lede}</p>
        </div>
        ${aside}
      </div>
    </div>
  </section>`;
}

export function kpi(label: string, value: string, sub: string, tone = 'text-ink'): string {
  return `<div class="card !p-5">
    <p class="font-mono text-xs uppercase tracking-wider text-slate">${label}</p>
    <p class="num mt-2 text-2xl font-semibold ${tone}">${value}</p>
    <p class="mt-1 text-sm text-slate">${sub}</p>
  </div>`;
}

/** A labelled horizontal bar; `value` in 0..1, optional dashed chance tick. */
export function bar(label: string, value: number | undefined, text: string, color = 'bg-teal', chance?: number): string {
  const tick =
    chance === undefined
      ? ''
      : `<span class="absolute inset-y-0 border-l-2 border-dashed border-slate" style="left:${chance * 100}%" title="chance"></span>`;
  return `<div class="flex flex-col gap-1">
    <div class="flex justify-between text-sm"><span>${label}</span><span class="num">${text}</span></div>
    <div class="track"><span class="fill ${color}" style="width:${Math.max(0, Math.min(1, value ?? 0)) * 100}%"></span>${tick}</div>
  </div>`;
}

export function section(eyebrow: string, title: string, body: string, intro = ''): string {
  return `<section class="shell mt-14">
    <p class="eyebrow">${eyebrow}</p>
    <h2 class="mt-2 text-2xl md:text-3xl">${title}</h2>
    ${intro ? `<p class="mt-2 max-w-3xl text-slate">${intro}</p>` : ''}
    <div class="mt-6">${body}</div>
  </section>`;
}
