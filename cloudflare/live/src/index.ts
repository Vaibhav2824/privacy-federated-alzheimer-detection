/**
 * Live federation rooms on Cloudflare: a Worker routes each websocket to a
 * Durable Object, one per room, which runs the same protocol as
 * backend/live.py. Hospitals upload the sum of their subjects' clipped updates
 * under pairwise masks; the room adds the uploads modulo 2^32 (the masks
 * cancel), adds Gaussian noise once and updates the 84-parameter model.
 *
 *   host:   wss://…/live/ws?role=host      then {"type": "host"}
 *   member: wss://…/live/ws?room=CODE      then {"type": "join", "code", "pub", "n_train"}
 */

import BUDGETS from '../../../backend/live_budgets.json';

export interface Env { ROOMS: DurableObjectNamespace; ALLOWED_ORIGINS?: string }

const P = 84;
const SCALE = 1e5;
const Q = 0.5;
const ROUNDS = 60;
const MAX_MEMBERS = 40;
const MAX_TRAIN = 1000;
const ROUND_TIMEOUT_MS = 12_000;
const MIN_ROUND_MS = 600;
const CODE_ALPHABET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789';
const DEFAULT_ORIGINS = String.raw`https://([\w-]+\.)*pages\.dev|https://[\w-]+\.vercel\.app|http://localhost:\d+`;

type Msg = Record<string, unknown>;
const budgets = BUDGETS as Record<string, { sigma: number; eps_by_round: number[] }>;

/* ---- arithmetic (mirrors ui/src/lib/fedcore.ts) ---- */

export function sumMod(vectors: number[][]): number[] {
  const out = new Array<number>(P).fill(0);
  for (const v of vectors) for (let i = 0; i < P; i++) out[i] = (out[i] + v[i]) % 4294967296;
  return out;
}

export function decode(words: number[]): number[] {
  return words.map((w) => (w >= 2147483648 ? w - 4294967296 : w) / SCALE);
}

function gauss(): number {
  const u = new Uint32Array(2);
  crypto.getRandomValues(u);
  const a = (u[0] + 1) / 4294967297, b = u[1] / 4294967296;
  return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
}

export function validUpdate(msg: Msg, t: number): boolean {
  const m = msg.masked;
  const ints = Array.isArray(m) && m.length === P && m.every((v) => Number.isInteger(v) && v >= 0 && v < 4294967296);
  const counts = ['correct', 'total'].every((k) => Number.isInteger(msg[k]) && (msg[k] as number) >= 0 && (msg[k] as number) <= MAX_TRAIN);
  return msg.t === t && ints && counts && (msg.correct as number) <= (msg.total as number);
}

const epsilon = (budget: string, rounds: number): number | null => budgets[budget]?.eps_by_round[rounds - 1] ?? null;
const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

function newCode(): string {
  const bytes = new Uint8Array(5);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => CODE_ALPHABET[b % CODE_ALPHABET.length]).join('');
}

/* ---- the Worker: origin check and routing ---- */

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === '/health') return Response.json({ status: 'ok' });
    if (url.pathname !== '/live/ws') return new Response('Not found', { status: 404 });
    if (request.headers.get('Upgrade') !== 'websocket') return new Response('Expected a websocket', { status: 426 });
    const origin = request.headers.get('Origin') ?? '';
    if (!new RegExp(`^(?:${env.ALLOWED_ORIGINS ?? DEFAULT_ORIGINS})$`).test(origin)) return new Response('Origin not allowed', { status: 403 });
    const host = url.searchParams.get('role') === 'host';
    const code = host ? newCode() : (url.searchParams.get('room') ?? '').toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 8);
    if (!code) return new Response('Missing room', { status: 400 });
    const room = env.ROOMS.get(env.ROOMS.idFromName(code));
    const forward = new URL(request.url);
    forward.searchParams.set('code', code);
    forward.searchParams.set('role', host ? 'host' : 'member');
    return room.fetch(new Request(forward.toString(), request));
  },
};

/* ---- one room ---- */

interface Member { ws: WebSocket; id: number; name: string; pub: string; n: number }

export class Room {
  private host: WebSocket | null = null;
  private members = new Map<number, Member>();
  private nextId = 1;
  private running = false;
  private cancelled = false;
  private t = 0;
  private updates = new Map<number, Msg>();
  private waiting = new Set<number>();
  private left = false;

  constructor(_state: DurableObjectState, _env: Env) {}

  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const code = url.searchParams.get('code') ?? '';
    const pair = new WebSocketPair();
    const [client, server] = [pair[0], pair[1]];
    server.accept();
    if (url.searchParams.get('role') === 'host') this.acceptHost(server, code);
    else this.acceptMember(server, code);
    return new Response(null, { status: 101, webSocket: client });
  }

  private send(ws: WebSocket | null, msg: Msg): void {
    try { ws?.send(JSON.stringify(msg)); } catch { /* closed sockets are cleaned up by their own close handler */ }
  }

  private broadcast(msg: Msg, host = true): void {
    for (const m of this.members.values()) this.send(m.ws, msg);
    if (host) this.send(this.host, msg);
  }

  private roster(): Msg {
    return { type: 'roster', members: [...this.members.values()].map((m) => ({ id: m.id, name: m.name, n: m.n })) };
  }

  private acceptHost(ws: WebSocket, code: string): void {
    if (this.host) { this.send(ws, { type: 'error', reason: 'Room code clash; open a new room.' }); ws.close(1000); return; }
    this.host = ws;
    const close = (): void => {
      if (this.host !== ws) return;
      this.host = null;
      this.cancelled = true;
      this.broadcast({ type: 'closed' }, false);
      for (const m of this.members.values()) { try { m.ws.close(1000); } catch { /* gone */ } }
      this.members.clear();
    };
    ws.addEventListener('message', (event) => {
      const msg = parse(event.data);
      if (msg.type === 'host') this.send(ws, { type: 'room', code });
      else if (msg.type === 'start' && !this.running) void this.run(String(msg.budget));
      else if (msg.type === 'stop' && this.running) {
        this.cancelled = true;
        this.broadcast({ type: 'stopped', reason: 'The presenter stopped the run.' });
      } else if (msg.type === 'end') { close(); ws.close(1000); }
    });
    ws.addEventListener('close', close);
    ws.addEventListener('error', close);
  }

  private acceptMember(ws: WebSocket, code: string): void {
    let member: Member | null = null;
    ws.addEventListener('message', (event) => {
      const msg = parse(event.data);
      if (msg.type === 'join' && !member) {
        const { pub, n_train: n } = msg;
        if (!this.host || String(msg.code ?? '').toUpperCase() !== code) {
          this.send(ws, { type: 'error', reason: 'No live room with that code.' }); ws.close(1000); return;
        }
        if (this.members.size >= MAX_MEMBERS || this.running) {
          this.send(ws, { type: 'error', reason: 'This room is full or already training.' }); ws.close(1000); return;
        }
        if (typeof pub !== 'string' || !pub || pub.length > 200 || !Number.isInteger(n) || (n as number) <= 0 || (n as number) > MAX_TRAIN) {
          this.send(ws, { type: 'error', reason: 'Malformed join request.' }); ws.close(1000); return;
        }
        member = { ws, id: this.nextId, name: `Hospital ${this.nextId}`, pub, n: n as number };
        this.nextId++;
        this.members.set(member.id, member);
        this.send(ws, { type: 'joined', id: member.id, name: member.name, code });
        this.broadcast(this.roster());
      } else if (msg.type === 'update' && member && this.waiting.has(member.id) && validUpdate(msg, this.t)) {
        this.updates.set(member.id, msg);
        this.waiting.delete(member.id);
      } else if (msg.type === 'leave') {
        ws.close(1000);
      }
    });
    const gone = (): void => {
      if (!member || !this.members.delete(member.id)) return;
      this.broadcast(this.roster());
      if (this.waiting.has(member.id)) this.left = true;
    };
    ws.addEventListener('close', gone);
    ws.addEventListener('error', gone);
  }

  private async run(budget: string): Promise<void> {
    this.running = true;
    this.cancelled = false;
    const sigma = budgets[budget]?.sigma ?? 0;
    let w = new Array<number>(P).fill(0);
    let t = 0;
    try {
      while (t < ROUNDS && !this.cancelled) {
        const start = Date.now();
        const cohort = new Map(this.members);
        if (cohort.size < 2) {
          this.broadcast({ type: 'stopped', reason: 'Secure aggregation needs at least two hospitals.' });
          return;
        }
        this.t = t;
        this.updates = new Map();
        this.waiting = new Set(cohort.keys());
        this.broadcast({ type: 'round', t, rounds: ROUNDS, q: Q, w, peers: [...cohort.values()].map((m) => ({ id: m.id, pub: m.pub })) }, false);
        const deadline = Date.now() + ROUND_TIMEOUT_MS;
        while (this.waiting.size && !this.left && !this.cancelled && Date.now() < deadline) await sleep(50);
        if (this.cancelled) return;
        const left = this.left;
        this.left = false;
        if (this.waiting.size) {
          // A hospital left or went silent, so its masks would not cancel: rerun without it.
          if (!left) {
            for (const id of this.waiting) {
              const m = this.members.get(id);
              this.members.delete(id);
              this.send(m?.ws ?? null, { type: 'dropped' });
            }
          }
          this.broadcast(this.roster());
          continue;
        }
        const ups = [...cohort.keys()].map((id) => this.updates.get(id) as Msg);
        const total = decode(sumMod(ups.map((u) => u.masked as number[])));
        const expected = Q * [...cohort.values()].reduce((a, m) => a + m.n, 0);
        w = w.map((v, i) => v + (total[i] + sigma * gauss()) / expected);
        const correct = ups.reduce((a, u) => a + (u.correct as number), 0);
        const seen = ups.reduce((a, u) => a + (u.total as number), 0);
        t++;
        this.broadcast({
          type: 'progress', t, rounds: ROUNDS, hospitals: cohort.size, accuracy: seen ? correct / seen : null,
          epsilon: epsilon(budget, t), sigma, server_sees: decode((ups[0].masked as number[]).slice(0, 6)), server_sum: total.slice(0, 6),
        });
        await sleep(Math.max(0, MIN_ROUND_MS - (Date.now() - start)));
      }
      if (!this.cancelled) this.broadcast({ type: 'done', w, epsilon: epsilon(budget, ROUNDS) });
    } finally {
      this.running = false;
    }
  }
}

function parse(data: unknown): Msg {
  try {
    const value = JSON.parse(typeof data === 'string' ? data : new TextDecoder().decode(data as ArrayBuffer));
    return value && typeof value === 'object' ? value as Msg : {};
  } catch { return {}; }
}
