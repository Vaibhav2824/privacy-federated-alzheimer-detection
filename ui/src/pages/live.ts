import QRCode from 'qrcode';

import { API_URL } from '../lib/api';
import { BUDGETS, budget, type Budget, Q, ROUNDS, spent } from '../lib/budgets';
import * as F from '../lib/fedcore';
import { newKeyPair, roundSeed, sharedSecret } from '../lib/keys';
import { icon, mountPage, pageHead, section } from '../ui/layout';

// The room server: a Cloudflare Worker when VITE_LIVE_WS_URL is set, else the API on Render.
const WS_URL = (import.meta.env.VITE_LIVE_WS_URL as string | undefined) ?? `${API_URL.replace(/^http/, 'ws')}/live/ws`;
const ROOM = new URLSearchParams(location.search).get('room');
const PER_HOSPITAL = { solo: 150, phone: 250, test: 80 };

const $ = (id: string): HTMLElement => document.getElementById(id) as HTMLElement;
const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));
const pct = (v: number | null | undefined): string => (v === null || v === undefined ? '-' : `${(v * 100).toFixed(1)}%`);
const fmt = (v: number): string => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : v.toFixed(3));

/* ======================================================================
 * Shared dashboard
 * ==================================================================== */

const C = 200, RING = 158;
let nodes: { x: number; y: number }[] = [];
let curve: number[] = [];

function topology(n: number): void {
  nodes = Array.from({ length: n }, (_, i) => {
    const a = (i / Math.max(n, 1)) * Math.PI * 2 - Math.PI / 2;
    return { x: C + Math.cos(a) * RING, y: C + Math.sin(a) * RING };
  });
  const r = n > 24 ? 9 : 14;
  $('topo').innerHTML = `
    <circle cx="${C}" cy="${C}" r="${RING}" fill="none" stroke="#e2e8f0" stroke-dasharray="4 5"/>
    ${nodes.map((p) => `<line x1="${C}" y1="${C}" x2="${p.x.toFixed(1)}" y2="${p.y.toFixed(1)}" stroke="#0F766E" stroke-opacity=".15"/>`).join('')}
    <g id="packets"></g>
    ${nodes.map((p, i) => `<g><circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${r}" fill="#fff" stroke="#0F766E" stroke-width="2"/>
      <text x="${p.x.toFixed(1)}" y="${(p.y + 4).toFixed(1)}" text-anchor="middle" class="fill-ink font-mono text-[10px]">${n > 24 ? '' : `H${i + 1}`}</text></g>`).join('')}
    <circle id="server-ring" cx="${C}" cy="${C}" r="46" fill="none" stroke="#14b8a6" stroke-width="2" opacity="0"/>
    <circle cx="${C}" cy="${C}" r="44" fill="#fff" stroke="#0F766E" stroke-width="2.5"/>
    <text x="${C}" y="${C - 3}" text-anchor="middle" class="fill-ink font-display text-[13px] font-semibold">Server</text>
    <text x="${C}" y="${C + 13}" text-anchor="middle" class="fill-slate font-mono text-[9px]">sum + noise</text>`;
  $('k-hospitals').textContent = String(n);
}

/** Dots flying along every spoke: up (hospital to server) or down (broadcast). */
function packets(up: boolean, ms = 420): void {
  const layer = document.getElementById('packets');
  if (!layer) return;
  const color = up ? '#E69F00' : '#0F766E';
  layer.innerHTML = nodes.map((p) => {
    const [x0, y0, x1, y1] = up ? [p.x, p.y, C, C] : [C, C, p.x, p.y];
    return `<circle r="5" fill="${color}"><animate attributeName="cx" from="${x0}" to="${x1}" dur="${ms}ms" fill="freeze"/>
      <animate attributeName="cy" from="${y0}" to="${y1}" dur="${ms}ms" fill="freeze"/>
      <animate attributeName="opacity" values="0;1;1;0" dur="${ms}ms" fill="freeze"/></circle>`;
  }).join('');
  if (up) {
    const ring = document.getElementById('server-ring');
    ring?.animate([{ opacity: 0.9, r: '46' }, { opacity: 0, r: '70' }], { duration: ms + 250, delay: ms - 60 });
  }
}

function chartFrame(): void {
  const W = 560, H = 220, L = 40, R = 12, T = 12, B = 28;
  const y = (v: number): number => T + (1 - (v - 0.3) / 0.5) * (H - T - B);
  const grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8].map((v) => `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="#e2e8f0" stroke-dasharray="3 3"/>
    <text x="${L - 6}" y="${y(v) + 4}" text-anchor="end" class="fill-slate font-mono text-[10px]">${Math.round(v * 100)}%</text>`).join('');
  const chance = y(0.42);
  $('chart').innerHTML = `${grid}
    <line x1="${L}" x2="${W - R}" y1="${chance}" y2="${chance}" stroke="#D55E00" stroke-dasharray="5 4"/>
    <text x="${W - R}" y="${chance - 5}" text-anchor="end" class="font-mono text-[10px]" fill="#D55E00">chance 42%</text>
    <text x="${L}" y="${H - 6}" class="fill-slate font-mono text-[10px]">round 1</text>
    <text x="${W - R}" y="${H - 6}" text-anchor="end" class="fill-slate font-mono text-[10px]">round ${ROUNDS}</text>
    <polyline id="acc-line" fill="none" stroke="#0F766E" stroke-width="2.5" stroke-linejoin="round" points=""/>
    <circle id="acc-dot" r="4.5" fill="#0F766E" opacity="0"/>`;
  curve = [];
}

function plot(acc: number): void {
  const W = 560, H = 220, L = 40, R = 12, T = 12, B = 28;
  curve.push(acc);
  const pts = curve.map((v, i) => {
    const x = L + (i / (ROUNDS - 1)) * (W - L - R);
    const y = T + (1 - (Math.min(Math.max(v, 0.3), 0.8) - 0.3) / 0.5) * (H - T - B);
    return [x, y];
  });
  $('acc-line').setAttribute('points', pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' '));
  const [lx, ly] = pts[pts.length - 1];
  const dot = $('acc-dot');
  dot.setAttribute('cx', lx.toFixed(1));
  dot.setAttribute('cy', ly.toFixed(1));
  dot.setAttribute('opacity', '1');
}

interface Progress { t: number; accuracy: number | null; epsilon: number | null; budget: Budget; seen: number[]; sum: number[] }

function report(p: Progress): void {
  $('k-round').textContent = `${p.t} / ${ROUNDS}`;
  $('k-acc').textContent = pct(p.accuracy);
  $('k-eps').textContent = p.epsilon === null ? 'no limit' : `${p.epsilon.toFixed(2)} of ${p.budget.label.replace('ε = ', '')}`;
  const bar = $('eps-bar');
  bar.style.width = p.epsilon === null ? '0%' : `${Math.min(1, p.epsilon / Number(p.budget.key)) * 100}%`;
  if (p.accuracy !== null) plot(p.accuracy);
  $('seen').innerHTML = p.seen.map((v) => `<span class="rounded bg-ad/10 px-1.5 py-0.5 text-ad">${fmt(v)}</span>`).join(' ');
  $('sum').innerHTML = p.sum.map((v) => `<span class="rounded bg-teal/10 px-1.5 py-0.5 text-teal">${fmt(v)}</span>`).join(' ');
}

function showImportance(w: Float64Array): void {
  const top = F.importance(w).slice(0, 8);
  const max = top[0].value || 1;
  $('importance').innerHTML = top.map((f) => `<div class="flex items-center gap-3 text-sm">
      <span class="w-56 shrink-0 truncate">${f.name}</span>
      <span class="track flex-1"><span class="fill bg-teal" style="width:${(f.value / max) * 100}%"></span></span>
      <span class="num w-14 text-right text-slate">${f.value.toFixed(2)}</span></div>`).join('');
  $('importance-card').classList.remove('hidden');
}

function resetDashboard(n: number): void {
  topology(n);
  chartFrame();
  $('k-round').textContent = `0 / ${ROUNDS}`;
  $('k-acc').textContent = '-';
  $('k-eps').textContent = '0';
  $('eps-bar').style.width = '0%';
  $('seen').textContent = 'waiting for round 1';
  $('sum').textContent = 'waiting for round 1';
  $('importance-card').classList.add('hidden');
}

const tile = (id: string, label: string, sub: string, extra = ''): string => `<div class="card !p-4">
  <p class="font-mono text-[11px] uppercase tracking-wider text-slate">${label}</p>
  <p id="${id}" class="num mt-1 text-2xl font-semibold tracking-tight">-</p>
  <p class="mt-0.5 text-xs text-slate">${sub}</p>${extra}</div>`;

function dashboard(): string {
  return `<div class="grid gap-5 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
    <div class="card !p-4">
      <div class="flex items-center justify-between"><p class="eyebrow">${icon('hub')} The federation</p>
        <p class="flex items-center gap-3 font-mono text-[11px] text-slate"><span><span class="inline-block h-2 w-2 rounded-full bg-mci"></span> masked upload</span>
        <span><span class="inline-block h-2 w-2 rounded-full bg-teal"></span> new model</span></p></div>
      <svg id="topo" viewBox="0 0 400 400" class="mx-auto mt-2 w-full max-w-md" role="img" aria-label="Hospitals around the server"></svg>
    </div>
    <div class="flex flex-col gap-4">
      <div class="grid grid-cols-2 gap-3 md:grid-cols-4">
        ${tile('k-round', 'Round', 'one round = every hospital trains once')}
        ${tile('k-acc', 'Accuracy', 'CN / MCI / AD on held-out patients')}
        ${tile('k-eps', 'Privacy spent', 'ε, tracked by the PRV accountant', '<div class="track mt-2"><span id="eps-bar" class="fill bg-ad" style="width:0%"></span></div>')}
        ${tile('k-hospitals', 'Hospitals', 'each one a separate device or site')}
      </div>
      <div class="card !p-4"><p class="font-mono text-[11px] uppercase tracking-wider text-slate">Accuracy by round</p>
        <svg id="chart" viewBox="0 0 560 220" class="mt-2 w-full" role="img" aria-label="Accuracy by round"></svg></div>
      <div class="card !p-4">
        <p class="font-mono text-[11px] uppercase tracking-wider text-slate">What the server actually sees (first 6 of 84 numbers)</p>
        <p class="mt-3 text-sm font-medium">One hospital's upload: masked, meaningless on its own</p>
        <p id="seen" class="num mt-1 flex flex-wrap gap-1 text-xs"></p>
        <p class="mt-3 text-sm font-medium">The sum of all uploads: the masks cancel, only the total is revealed</p>
        <p id="sum" class="num mt-1 flex flex-wrap gap-1 text-xs"></p>
      </div>
    </div>
  </div>
  <div id="importance-card" class="card mt-5 hidden">
    <p class="eyebrow">${icon('insights')} What the federation learned</p>
    <p class="mt-1 text-sm text-slate">Top features of the trained 84-parameter model (sum of absolute weights). The synthetic patients carry most of their disease signal in medial-temporal volumes, amyloid and tau. Privacy noise reshuffles the ranking from run to run, which is the explanation instability the study measures.</p>
    <div id="importance" class="mt-4 flex flex-col gap-2"></div>
  </div>`;
}

/* ======================================================================
 * Presenter page
 * ==================================================================== */

let mode: 'solo' | 'room' = 'solo';
let chosen: Budget = BUDGETS[0];
let stopSolo = false;
let socket: WebSocket | null = null;
let roster = 0;

const pill = (attr: string, value: string, label: string, on: boolean): string =>
  `<button type="button" data-${attr}="${value}" aria-pressed="${on}" class="rounded-full border px-3.5 py-1.5 text-sm font-medium transition-colors ${on ? 'border-teal bg-teal text-white' : 'border-hairline bg-surface text-ink hover:border-teal/50'}">${label}</button>`;

function controls(): string {
  return `<div class="card flex flex-col gap-5">
    <div class="flex flex-wrap items-center gap-x-8 gap-y-3">
      <div class="flex items-center gap-2"><span class="text-sm text-slate">Where</span>
        <span id="modes" class="flex gap-1.5">${pill('mode', 'solo', 'Simulate in this browser', true)}${pill('mode', 'room', 'With the audience', false)}</span></div>
      <div class="flex items-center gap-2"><span class="text-sm text-slate">Privacy</span>
        <span id="budgets" class="flex gap-1.5">${BUDGETS.map((b, i) => pill('budget', b.key, b.label, i === 0)).join('')}</span></div>
    </div>
    <div id="solo-panel" class="flex flex-wrap items-center gap-3">
      <label class="flex items-center gap-2 text-sm text-slate">Hospitals
        <select id="solo-n" class="rounded-lg border border-hairline bg-surface px-2 py-1.5 text-ink">
          ${[4, 6, 8, 12, 16].map((n) => `<option ${n === 8 ? 'selected' : ''}>${n}</option>`).join('')}</select></label>
      <button id="solo-run" class="btn btn-primary">${icon('play_arrow')} Train the model</button>
      <button id="solo-stop" class="btn hidden">${icon('stop')} Stop</button>
      <p class="text-sm text-slate">Every hospital is simulated here, with the same arithmetic the phones use.</p>
    </div>
    <div id="room-panel" class="hidden">
      <div id="room-start" class="flex flex-wrap items-center gap-3">
        <button id="room-open" class="btn btn-primary">${icon('qr_code_2')} Open a live room</button>
        <p id="room-status" class="text-sm text-slate">The audience scans a QR code; each phone becomes a hospital with its own patients.</p>
      </div>
      <div id="room-live" class="hidden grid items-center gap-6 md:grid-cols-[auto_1fr]">
        <div id="qr" class="rounded-2xl border border-hairline bg-white p-3"></div>
        <div class="flex flex-col gap-3">
          <p class="font-mono text-xs uppercase tracking-wider text-slate">Scan to join, or open</p>
          <p id="join-url" class="break-all font-mono text-sm text-teal"></p>
          <p class="text-5xl font-semibold tracking-[0.2em]" id="room-code"></p>
          <p class="text-sm"><span id="roster-n" class="num font-semibold">0</span> hospitals joined
            <span class="text-slate">· secure aggregation needs at least 2</span></p>
          <div class="flex flex-wrap gap-2">
            <button id="room-run" class="btn btn-primary" disabled>${icon('play_arrow')} Train together</button>
            <button id="room-stop" class="btn">${icon('stop')} Stop</button>
            <button id="room-end" class="btn">${icon('close')} Close the room</button>
          </div>
          <p id="room-note" class="text-sm text-slate"></p>
        </div>
      </div>
    </div>
  </div>`;
}

function steps(): string {
  const items = [
    ['download', 'The server sends the model', 'Every hospital receives the same 84 numbers: the current global model.'],
    ['person', 'Each patient trains it once', 'A random half of each hospital\'s patients (q = 0.5) each take 3 training steps; every patient\'s change is clipped to length 1.'],
    ['lock', 'Uploads are masked in pairs', 'Each pair of hospitals agrees a secret with ECDH. One adds the mask, the other subtracts it, so a single upload is noise.'],
    ['functions', 'The server adds noise once', 'Summing the uploads cancels the masks. The server adds Gaussian noise to the total and updates the model.'],
  ];
  return `<div class="grid gap-4 md:grid-cols-4">${items.map(([ic, h, b], i) => `<div class="card !p-5">
    <span class="grid h-9 w-9 place-items-center rounded-lg bg-mint text-teal">${icon(ic)}</span>
    <p class="mt-3 font-mono text-xs text-slate">Step 0${i + 1}</p>
    <p class="mt-1 font-display text-lg font-semibold">${h}</p>
    <p class="mt-1 text-sm text-slate">${b}</p></div>`).join('')}</div>`;
}

function presenter(): string {
  return `${pageHead('Live · federated learning you can join',
    'Train the model <span class="text-gradient">together, live</span>',
    'Run the paper\'s subject-level DP-FedAvg in front of an audience. Each phone that joins becomes a hospital, trains on its own patients and uploads only a masked update. No patient record ever leaves a device.',
    `<span class="badge">${icon('shield')} 84 parameters · ε = 1 · δ = 0.001</span>`)}
  <section class="shell mt-10" data-reveal>${controls()}</section>
  <section class="shell mt-5" data-reveal>${dashboard()}</section>
  ${section('How it works', 'What happens in every round', steps(),
    'The same four steps the study ran over 48 real ADNI sites, repeated for 60 rounds.')}
  <section class="shell mt-10"><p class="notice">${icon('info')} The patients on this page are synthetic: each device generates its own from a fixed model of Alzheimer's progression, so no ADNI data is used or sent here.
    The study's own results on 1,262 real ADNI subjects are on the <a class="underline" href="results.html">Results</a> page.</p></section>`;
}

function pick(attr: string, value: string): void {
  document.querySelectorAll<HTMLButtonElement>(`[data-${attr}]`).forEach((b) => {
    const on = b.dataset[attr] === value;
    b.setAttribute('aria-pressed', String(on));
    b.className = b.className.replace(/border-teal bg-teal text-white|border-hairline bg-surface text-ink hover:border-teal\/50/,
      on ? 'border-teal bg-teal text-white' : 'border-hairline bg-surface text-ink hover:border-teal/50');
  });
}

async function runSolo(): Promise<void> {
  const n = Number(($('solo-n') as HTMLSelectElement).value);
  const b = chosen;
  stopSolo = false;
  $('solo-run').classList.add('hidden');
  $('solo-stop').classList.remove('hidden');
  resetDashboard(n);
  const seed = Math.floor(Math.random() * 2 ** 31);
  const hospitals = Array.from({ length: n }, (_, i) => F.hospital(seed + i * 7919, PER_HOSPITAL.solo, PER_HOSPITAL.test));
  const test = hospitals.flatMap((h) => h.test);
  const expected = Q * hospitals.reduce((a, h) => a + h.train.length, 0);
  const r = F.rng(seed ^ 0x5bd1e995);
  let w = F.zeros();
  for (let t = 0; t < ROUNDS && !stopSolo; t++) {
    packets(false, 300);
    await sleep(300);
    const s = F.score(w, test);
    const masked = hospitals.map((h, i) => {
      const peers = hospitals.map((_, j) => ({ id: j, seed: F.pairSeed(seed, i, j, t) })).filter((p) => p.id !== i);
      return F.mask(F.encode(F.clippedSum(w, h.train, Q, r).sum), i, peers);
    });
    packets(true, 300);
    await sleep(330);
    const sum = F.decode(F.sumMod(masked));
    w = F.serverStep(w, sum, expected, b.sigma, r);
    report({ t: t + 1, accuracy: s.correct / s.total, epsilon: spent(b, t + 1), budget: b,
      seen: Array.from(F.decode(masked[0]).slice(0, 6)), sum: Array.from(sum.slice(0, 6)) });
  }
  if (!stopSolo) showImportance(w);
  $('solo-run').classList.remove('hidden');
  $('solo-stop').classList.add('hidden');
}

function openRoom(): void {
  const status = $('room-status');
  status.textContent = 'Connecting to the federation server…';
  ($('room-open') as HTMLButtonElement).disabled = true;
  socket = new WebSocket(`${WS_URL}?role=host`);
  const timer = setTimeout(() => {
    if (socket?.readyState !== WebSocket.OPEN) {
      status.textContent = 'The server did not answer. Try again in a moment, or use "Simulate in this browser".';
      ($('room-open') as HTMLButtonElement).disabled = false;
    }
  }, 75_000);
  socket.onopen = (): void => { clearTimeout(timer); socket?.send(JSON.stringify({ type: 'host' })); };
  socket.onclose = (): void => {
    $('room-note').textContent = 'The room is closed.';
    ($('room-open') as HTMLButtonElement).disabled = false;
  };
  socket.onmessage = async (event): Promise<void> => {
    const msg = JSON.parse(String(event.data));
    if (msg.type === 'room') {
      const url = `${location.origin}${location.pathname}?room=${msg.code}`;
      $('room-start').classList.add('hidden');
      $('room-live').classList.remove('hidden');
      $('room-code').textContent = msg.code;
      $('join-url').textContent = url;
      $('qr').innerHTML = await QRCode.toString(url, { type: 'svg', margin: 1, width: 220, color: { dark: '#0f172a', light: '#ffffff' } });
      resetDashboard(0);
    } else if (msg.type === 'roster') {
      roster = msg.members.length;
      $('roster-n').textContent = String(roster);
      ($('room-run') as HTMLButtonElement).disabled = roster < 2;
      topology(roster);
    } else if (msg.type === 'progress') {
      packets(false, 280);
      setTimeout(() => packets(true, 280), 300);
      report({ t: msg.t, accuracy: msg.accuracy, epsilon: msg.epsilon, budget: chosen, seen: msg.server_sees, sum: msg.server_sum });
    } else if (msg.type === 'done') {
      showImportance(Float64Array.from(msg.w));
      $('room-note').textContent = 'Training complete. Close the room or train again.';
    } else if (msg.type === 'stopped' || msg.type === 'error') {
      $('room-note').textContent = msg.reason;
      status.textContent = msg.reason;
      ($('room-open') as HTMLButtonElement).disabled = false;
    }
  };
}

function wirePresenter(): void {
  resetDashboard(8);
  document.querySelectorAll<HTMLButtonElement>('[data-mode]').forEach((b) => b.addEventListener('click', () => {
    mode = b.dataset.mode as 'solo' | 'room';
    pick('mode', mode);
    $('solo-panel').classList.toggle('hidden', mode !== 'solo');
    $('room-panel').classList.toggle('hidden', mode !== 'room');
    resetDashboard(mode === 'solo' ? Number(($('solo-n') as HTMLSelectElement).value) : roster);
  }));
  document.querySelectorAll<HTMLButtonElement>('[data-budget]').forEach((b) => b.addEventListener('click', () => {
    chosen = budget(b.dataset.budget ?? '1');
    pick('budget', chosen.key);
  }));
  $('solo-n').addEventListener('change', () => resetDashboard(Number(($('solo-n') as HTMLSelectElement).value)));
  $('solo-run').addEventListener('click', () => { void runSolo(); });
  $('solo-stop').addEventListener('click', () => { stopSolo = true; });
  $('room-open').addEventListener('click', openRoom);
  $('room-run').addEventListener('click', () => {
    resetDashboard(roster);
    $('room-note').textContent = `Training with ${roster} hospitals at ${chosen.label}.`;
    socket?.send(JSON.stringify({ type: 'start', budget: chosen.key }));
  });
  $('room-stop').addEventListener('click', () => socket?.send(JSON.stringify({ type: 'stop' })));
  $('room-end').addEventListener('click', () => {
    socket?.send(JSON.stringify({ type: 'end' }));
    $('room-live').classList.add('hidden');
    $('room-start').classList.remove('hidden');
    $('room-status').textContent = 'Room closed. Open a new one any time.';
  });
}

/* ======================================================================
 * Phone (hospital) page
 * ==================================================================== */

function hospitalPage(code: string): string {
  return `<section class="shell max-w-xl py-10">
    <p class="eyebrow"><span class="live-dot"></span>Live room ${code.replace(/[^A-Z0-9]/gi, '')}</p>
    <h1 id="h-name" class="mt-2 text-4xl">Joining the federation…</h1>
    <p id="h-status" class="mt-2 text-slate">Generating your hospital's patients and a key pair.</p>
    <div class="mt-6 grid grid-cols-2 gap-3">
      ${tile('h-patients', 'Your patients', 'synthetic, they stay on this phone')}
      ${tile('h-round', 'Round', `of ${ROUNDS}`)}
      ${tile('h-acc', 'Shared model accuracy', 'on everyone\'s held-out patients')}
      ${tile('h-eps', 'Privacy spent', 'ε so far')}
    </div>
    <div class="card mt-4 !p-4">
      <p class="font-mono text-[11px] uppercase tracking-wider text-slate">Your last upload (first 4 of 84 numbers)</p>
      <p id="h-upload" class="num mt-2 flex flex-wrap gap-1 text-xs text-slate">nothing sent yet</p>
      <p class="mt-2 text-xs text-slate">Masked with a secret shared with each other hospital. On its own it is random; only the sum over all hospitals means anything.</p>
    </div>
    <p class="mt-6 text-xs text-slate">Synthetic patients only. Research demonstration, not a medical device.</p>
  </section>`;
}

async function runHospital(code: string): Promise<void> {
  const keys = await newKeyPair();
  const data = F.hospital(Math.floor(Math.random() * 2 ** 31), PER_HOSPITAL.phone, PER_HOSPITAL.test);
  const r = F.rng(Math.floor(Math.random() * 2 ** 31));
  const secrets = new Map<number, Uint8Array>();
  let me = 0;
  let b: Budget = BUDGETS[0];
  $('h-patients').textContent = String(data.train.length + data.test.length);
  try { await (navigator as Navigator & { wakeLock?: { request(t: string): Promise<unknown> } }).wakeLock?.request('screen'); } catch { /* optional */ }
  const status = $('h-status');
  status.textContent = 'Connecting to the federation server…';
  const ws = new WebSocket(`${WS_URL}?room=${encodeURIComponent(code)}`);
  ws.onopen = (): void => ws.send(JSON.stringify({ type: 'join', code, pub: keys.publicKey, n_train: data.train.length }));
  ws.onclose = (): void => { status.textContent = 'Disconnected from the room.'; };
  ws.onmessage = async (event): Promise<void> => {
    const msg = JSON.parse(String(event.data));
    if (msg.type === 'joined') {
      me = msg.id;
      $('h-name').textContent = `You are ${msg.name}`;
      status.textContent = 'Waiting for the presenter to start training.';
    } else if (msg.type === 'round') {
      status.textContent = 'Training on your patients…';
      const w = Float64Array.from(msg.w);
      const s = F.score(w, data.test);
      const update = F.encode(F.clippedSum(w, data.train, msg.q, r).sum);
      const peers = await Promise.all((msg.peers as { id: number; pub: string }[]).filter((p) => p.id !== me).map(async (p) => {
        if (!secrets.has(p.id)) secrets.set(p.id, await sharedSecret(keys.privateKey, p.pub));
        return { id: p.id, seed: await roundSeed(secrets.get(p.id) as Uint8Array, msg.t) };
      }));
      const masked = F.mask(update, me, peers);
      ws.send(JSON.stringify({ type: 'update', t: msg.t, masked: Array.from(masked), correct: s.correct, total: s.total }));
      $('h-round').textContent = String(msg.t + 1);
      $('h-upload').innerHTML = Array.from(F.decode(masked).slice(0, 4)).map((v) => `<span class="rounded bg-ad/10 px-1.5 py-0.5 text-ad">${fmt(v)}</span>`).join(' ');
      status.textContent = 'Masked update sent. Waiting for the other hospitals…';
    } else if (msg.type === 'progress') {
      $('h-acc').textContent = pct(msg.accuracy);
      if (msg.epsilon !== null) b = BUDGETS.find((x) => Math.abs(x.sigma - msg.sigma) < 1e-6) ?? b;
      $('h-eps').textContent = msg.epsilon === null ? 'no limit' : `${msg.epsilon.toFixed(2)} of ${b.key}`;
    } else if (msg.type === 'done') {
      status.textContent = 'Training complete. Your patients never left this phone.';
    } else if (msg.type === 'dropped') {
      status.textContent = 'This phone answered too slowly and was dropped from the round. Reload to rejoin.';
    } else if (msg.type === 'stopped' || msg.type === 'error') {
      status.textContent = msg.reason;
    } else if (msg.type === 'closed') {
      status.textContent = 'The presenter closed the room. Thanks for taking part!';
    }
  };
}

/* ====================================================================== */

if (ROOM) {
  void mountPage('live', () => hospitalPage(ROOM)).then(() => runHospital(ROOM.toUpperCase()));
} else {
  void mountPage('live', presenter).then((data) => { if (data) wirePresenter(); });
}
