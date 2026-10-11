import { bestCentralised, pct, pick, points, type Condition, type SiteData } from '../lib/site';
import { bar, icon, mountPage, pageHead, section } from '../ui/layout';

const SCHEMES = [
  { key: 'iid', name: 'IID simulation', note: 'subjects shuffled evenly across clients' },
  { key: 'dirichlet', name: 'Dirichlet non-IID (α = 0.5)', note: 'label skew drawn at random' },
  { key: 'natural', name: 'Real ADNI sites', note: 'one client per acquisition site' },
];

/** Star topology: one node per real client, radius by subject count. */
function topology(data: SiteData): string {
  const clients = data.partition.clients;
  const max = Math.max(...clients.map((c) => c.n_subjects));
  const C = 210;
  const nodes = clients.map((c, i) => {
    const a = (i / clients.length) * Math.PI * 2 - Math.PI / 2;
    const ring = i % 2 ? 165 : 185;
    const x = C + Math.cos(a) * ring, y = C + Math.sin(a) * ring;
    const r = 3 + 9 * Math.sqrt(c.n_subjects / max);
    const missing = c.class_counts.some((n) => n === 0);
    return `<line x1="${C}" y1="${C}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="#0F766E" stroke-opacity=".18"/>
      <circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="${missing ? '#FFFBEB' : '#fff'}" stroke="${missing ? '#B45309' : '#0F766E'}" stroke-width="1.5">
      <title>Client: ${c.n_subjects} subjects (CN ${c.class_counts[0]} / MCI ${c.class_counts[1]} / AD ${c.class_counts[2]})</title></circle>`;
  }).join('');
  return `<svg viewBox="0 0 420 420" class="mx-auto w-full max-w-md" role="img" aria-label="${clients.length} federated clients around a central server">
    <circle cx="${C}" cy="${C}" r="120" fill="none" stroke="#e2e8f0" stroke-dasharray="4 4"/>
    ${nodes}
    <circle cx="${C}" cy="${C}" r="42" fill="#fff" stroke="#0F766E" stroke-width="2"/>
    <text x="${C}" y="${C - 2}" text-anchor="middle" class="fill-ink font-display text-[13px]">Server</text>
    <text x="${C}" y="${C + 14}" text-anchor="middle" class="fill-slate font-mono text-[9px]">DP-FedAvg</text>
  </svg>`;
}

function realism(data: SiteData): string {
  const fed = (scheme: string, ls: string): Condition | undefined => pick(data, { family: 'federated', label_set: ls, split_scheme: 'subject', client_scheme: scheme });
  return SCHEMES.map((s) => {
    const three = fed(s.key, 'cn-mci-ad');
    const cnad = fed(s.key, 'cn-ad');
    return `<div class="rounded-lg p-3 ${s.key === 'natural' ? 'border border-teal/30 bg-teal/[0.04]' : ''}">
      <p class="text-sm font-medium">${s.name} <span class="font-normal text-slate">· ${s.note}</span></p>
      <div class="mt-2 flex flex-col gap-2">
        ${bar('3-class', three?.accuracy, pct(three?.accuracy), 'bg-teal', three?.chance_accuracy)}
        ${bar('CN vs AD', cnad?.accuracy, pct(cnad?.accuracy), 'bg-cn', cnad?.chance_accuracy)}
      </div></div>`;
  }).join('');
}

function clientTable(data: SiteData): string {
  const clients = data.partition.clients;
  const shown = clients.slice(0, 10);
  const sizes = clients.map((c) => c.n_subjects);
  const rows = shown.map((c, i) => `<tr class="border-t border-hairline">
    <td class="p-3 font-mono text-teal">Client ${String(i + 1).padStart(2, '0')}</td>
    <td class="num p-3 text-right">${c.n_sites}</td>
    <td class="num p-3 text-right">${c.n_subjects}</td>
    <td class="p-3"><div class="flex h-2 w-40 overflow-hidden rounded">
      ${c.class_counts.map((n, k) => `<span class="${['bg-cn', 'bg-mci', 'bg-ad'][k]}" style="width:${(n / c.n_subjects) * 100}%"></span>`).join('')}</div>
      <span class="num text-xs text-slate">${c.class_counts.join(' / ')}</span></td>
    <td class="num p-3 text-right">${c.label_entropy.toFixed(2)}</td></tr>`).join('');
  return `<div class="overflow-x-auto rounded-lg border border-hairline bg-surface"><table class="w-full text-sm">
    <thead class="bg-muted text-left font-mono text-xs uppercase tracking-wider text-slate"><tr>
      <th class="p-3">Client (by size)</th><th class="p-3 text-right">Sites pooled</th><th class="p-3 text-right">Subjects</th>
      <th class="p-3">CN / MCI / AD</th><th class="p-3 text-right">Label entropy</th></tr></thead>
    <tbody>${rows}</tbody></table></div>
    <div class="mt-2 flex flex-wrap justify-between gap-2 text-xs text-slate">
      <span>Showing ${shown.length} of ${clients.length} clients. Sizes range from ${Math.min(...sizes)} to ${Math.max(...sizes)} subjects.</span>
      <span>${clients.filter((c) => c.class_counts.some((n) => n === 0)).length} clients lack at least one class (amber in the diagram).</span></div>`;
}

function algorithm(data: SiteData): string {
  const d = data.dp_config;
  return `<pre class="overflow-x-auto rounded-xl bg-paper p-4 font-mono text-xs leading-6"><span class="text-slate">// server round t = 1..${d.rounds}</span>
<b>for</b> each client k, each subject i of k
    sampled with probability q = ${d.subject_sample_rate}:
  Δ_i ← LocalTrain(w_t, subject i, E = ${d.local_epochs}) − w_t
  Δ_i ← Δ_i · min(1, C / ‖Δ_i‖)    <span class="text-slate">// C = ${d.clip_norm}</span>
<b>end</b>
Δ ← Σ Δ_i + N(0, σ²C²I)        <span class="text-slate">// noise added once, at the server</span>
w_t+1 ← w_t + Δ / participants</pre>
    <p class="mt-3 rounded-xl bg-muted p-3 text-xs text-slate">${icon('info', '!text-sm')} Federation is simulated in one process: clients
      exchange only clipped, noised updates, and a Rényi DP accountant tracks (ε, δ = ${d.delta}) across all ${d.rounds} rounds.</p>`;
}

function render(data: SiteData): string {
  const fed = (s: string): Condition | undefined => pick(data, { family: 'federated', label_set: 'cn-mci-ad', split_scheme: 'subject', client_scheme: s });
  const central = pick(data, { family: 'centralised', model: 'logreg_cv', label_set: 'cn-mci-ad', split_scheme: 'subject' });
  const best = bestCentralised(data, 'cn-mci-ad', 'subject');
  return `
  ${pageHead('Federated topology · real sites vs simulated partitions', 'Hospital topology vs simulated benchmarks',
    'Federated training over the cohort\'s real ADNI acquisition sites, compared with the IID and Dirichlet partitions the literature usually simulates.',
    `<div class="flex flex-wrap gap-2 md:max-w-xs md:justify-end"><span class="badge">${data.partition.n_clients} federated clients</span>
      <span class="badge">${data.partition.n_sites} acquisition sites</span><span class="badge">Subject-level DP-FedAvg</span></div>`)}
  <section class="shell mt-10 grid gap-6 lg:grid-cols-2">
    <div class="card">
      <p class="font-mono text-xs uppercase tracking-wider text-slate">Figure · physical topology</p>
      <h2 class="mt-1 font-display text-xl">One client per real site</h2>
      <p class="mt-1 text-sm text-slate">Sites with fewer than eight subjects are pooled so none is discarded. Node area ∝ subjects.</p>
      ${topology(data)}
    </div>
    <div class="flex flex-col gap-6">
      <div class="card">
        <p class="font-mono text-xs uppercase tracking-wider text-slate">Empirical evaluation</p>
        <h2 class="mt-1 font-display text-xl">Accuracy across partition realism</h2>
        <p class="mt-1 text-sm text-slate">Subject-disjoint folds, three seeds. Dashed ticks: chance.</p>
        <div class="mt-4 flex flex-col gap-3">${realism(data)}</div>
      </div>
      <blockquote class="card font-display text-lg leading-8">
        <p class="italic">Real hospital partitions are at least as hard as the standard simulated benchmark.</p>
        <p class="mt-3 font-sans text-sm text-slate">Three-class accuracy on real sites is ${pct(fed('natural')?.accuracy)}, against
          ${pct(fed('dirichlet')?.accuracy)} for the Dirichlet draw and ${pct(fed('iid')?.accuracy)} for IID: federating over real sites costs
          ${points(fed('natural')?.accuracy, central?.accuracy)} against the same model trained centrally
          (${pct(best.accuracy)} for the best centralised model). Client-size imbalance and scanner variation sit on top of label skew.</p>
      </blockquote>
    </div>
  </section>
  <section class="shell mt-14 grid gap-6 lg:grid-cols-[1.6fr_1fr]">
    <div><p class="eyebrow">Client partition</p><h2 class="mt-2 text-2xl">The largest real-site clients</h2>
      <div class="mt-6">${clientTable(data)}</div></div>
    <div><p class="eyebrow">Algorithm</p><h2 class="mt-2 text-2xl">DP-FedAvg, subject level</h2>
      <div class="card mt-6">${algorithm(data)}</div></div>
  </section>
  ${section('Why site-disjoint evaluation', 'A federated model is deployed at hospitals that did not train it', `<p class="max-w-3xl text-slate">The
    Results page reports every task under both subject-disjoint and site-disjoint folds, so the cost of meeting an unseen scanner is
    measured rather than assumed.</p>`)}`;
}

void mountPage('federation', render);
