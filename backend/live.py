"""
Live federation rooms: an audience trains the 84-parameter model together.

A presenter opens a room; every phone that joins becomes a hospital holding
its own synthetic patients (generated in the browser, never sent here). Each
round this server broadcasts the global weights, every hospital uploads the
sum of its subjects' clipped updates hidden under pairwise masks, and the
server adds the masked uploads modulo 2^32. The masks cancel, so the server
learns only the total; it then adds Gaussian noise once and updates the model,
exactly as the paper's subject-level DP-FedAvg does.

Hospitals agree their pairwise mask seeds with ECDH in the browser; the
server only relays the public keys. A hospital that drops mid-round is removed
and the round is rerun by the rest, since its masks would no longer cancel.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

P = 84
SCALE = 1e5
Q = 0.5
ROUNDS = 60
MAX_ROOMS = 30
MAX_MEMBERS = 40
MAX_TRAIN = 1000
ROUND_TIMEOUT = float(os.environ.get("LIVE_ROUND_TIMEOUT", "12"))
MIN_ROUND_SECONDS = float(os.environ.get("LIVE_MIN_ROUND_SECONDS", "0.6"))
ROOM_TTL = 3600
BUDGETS = json.loads((Path(__file__).parent / "live_budgets.json").read_text(encoding="utf-8"))
ORIGIN = re.compile(os.environ.get("ALLOWED_ORIGIN_REGEX", r"https://[\w-]+\.vercel\.app|http://localhost:\d+"))
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"

router = APIRouter()


# ---- arithmetic (mirrors ui/src/lib/fedcore.ts) ----

def sum_mod(vectors: list[list[int]]) -> list[int]:
    return [sum(column) % 2**32 for column in zip(*vectors)]


def decode(words: list[int]) -> list[float]:
    return [(w - 2**32 if w >= 2**31 else w) / SCALE for w in words]


def server_step(w: list[float], total: list[float], expected: float, sigma: float,
                rand: random.Random) -> list[float]:
    return [wi + (ti + sigma * rand.gauss(0.0, 1.0)) / expected for wi, ti in zip(w, total)]


def valid_update(msg: dict, t: int) -> bool:
    masked = msg.get("masked")
    ints = isinstance(masked, list) and len(masked) == P and all(
        isinstance(v, int) and 0 <= v < 2**32 for v in masked)
    counts = all(isinstance(msg.get(k), int) and 0 <= msg[k] <= MAX_TRAIN for k in ("correct", "total"))
    return msg.get("t") == t and ints and counts and msg["correct"] <= msg["total"]


def epsilon(budget: str, rounds: int) -> float | None:
    if budget not in BUDGETS:
        return None
    return BUDGETS[budget]["eps_by_round"][rounds - 1]


# ---- rooms ----

@dataclass
class Member:
    ws: WebSocket
    id: int
    name: str
    pub: str
    n_train: int


@dataclass
class Room:
    code: str
    host: WebSocket
    created: float = field(default_factory=time.monotonic)
    members: dict[int, Member] = field(default_factory=dict)
    next_id: int = 1
    task: asyncio.Task | None = None
    t: int = 0
    updates: dict[int, dict] = field(default_factory=dict)
    waiting: set[int] = field(default_factory=set)
    left: bool = False

    def roster(self) -> dict:
        return {"type": "roster", "members": [
            {"id": m.id, "name": m.name, "n": m.n_train} for m in self.members.values()]}


ROOMS: dict[str, Room] = {}


async def _send(ws: WebSocket, msg: dict) -> None:
    try:
        await ws.send_json(msg)
    except Exception:  # noqa: BLE001 - a closed socket is handled by its own receive loop
        pass


async def _broadcast(room: Room, msg: dict, host: bool = True) -> None:
    targets = [m.ws for m in room.members.values()] + ([room.host] if host else [])
    await asyncio.gather(*(_send(ws, msg) for ws in targets))


def _new_code() -> str:
    while True:
        code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(5))
        if code not in ROOMS:
            return code


def _expire(now: float) -> None:
    for code in [c for c, r in ROOMS.items() if now - r.created > ROOM_TTL]:
        room = ROOMS.pop(code)
        if room.task:
            room.task.cancel()


async def run(room: Room, budget: str) -> None:
    """Drive ROUNDS rounds of masked DP-FedAvg over the room's hospitals."""
    sigma = BUDGETS[budget]["sigma"] if budget in BUDGETS else 0.0
    rand = random.Random(secrets.randbits(64))
    w = [0.0] * P
    t = 0
    while t < ROUNDS:
        start = time.monotonic()
        cohort = dict(room.members)
        if len(cohort) < 2:
            await _broadcast(room, {"type": "stopped", "reason": "Secure aggregation needs at least two hospitals."})
            return
        room.t, room.updates, room.waiting = t, {}, set(cohort)
        await _broadcast(room, {"type": "round", "t": t, "rounds": ROUNDS, "q": Q, "w": w,
                                "peers": [{"id": m.id, "pub": m.pub} for m in cohort.values()]}, host=False)
        deadline = time.monotonic() + ROUND_TIMEOUT
        while room.waiting and not room.left and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        left, room.left = room.left, False
        if room.waiting:
            # A hospital left or went silent, so its masks would not cancel: rerun
            # this round without it. Silent ones are dropped; ones that left are gone.
            if not left:
                for missing in list(room.waiting):
                    gone = room.members.pop(missing, None)
                    if gone:
                        await _send(gone.ws, {"type": "dropped"})
            await _broadcast(room, room.roster())
            continue
        updates = [room.updates[i] for i in cohort]
        total = decode(sum_mod([u["masked"] for u in updates]))
        expected = Q * sum(m.n_train for m in cohort.values())
        w = server_step(w, total, expected, sigma, rand)
        correct = sum(u["correct"] for u in updates)
        seen = sum(u["total"] for u in updates)
        t += 1
        await _broadcast(room, {
            "type": "progress", "t": t, "rounds": ROUNDS, "hospitals": len(cohort),
            "accuracy": correct / seen if seen else None, "epsilon": epsilon(budget, t), "sigma": sigma,
            "server_sees": decode(updates[0]["masked"][:6]), "server_sum": total[:6]})
        await asyncio.sleep(max(0.0, MIN_ROUND_SECONDS - (time.monotonic() - start)))
    await _broadcast(room, {"type": "done", "w": w, "epsilon": epsilon(budget, ROUNDS)})


async def _host(ws: WebSocket) -> None:
    _expire(time.monotonic())
    if len(ROOMS) >= MAX_ROOMS:
        await ws.send_json({"type": "error", "reason": "Too many live rooms right now; try again shortly."})
        return
    room = Room(code=_new_code(), host=ws)
    ROOMS[room.code] = room
    await ws.send_json({"type": "room", "code": room.code})
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "start" and not (room.task and not room.task.done()):
                room.task = asyncio.create_task(run(room, str(msg.get("budget"))))
            elif msg.get("type") == "stop" and room.task:
                room.task.cancel()
                await _broadcast(room, {"type": "stopped", "reason": "The presenter stopped the run."})
            elif msg.get("type") == "end":
                break
    except WebSocketDisconnect:
        pass
    finally:
        if room.task:
            room.task.cancel()
        ROOMS.pop(room.code, None)
        await _broadcast(room, {"type": "closed"}, host=False)


async def _member(ws: WebSocket, hello: dict) -> None:
    room = ROOMS.get(str(hello.get("code", "")).upper())
    pub, n = hello.get("pub"), hello.get("n_train")
    if room is None:
        await ws.send_json({"type": "error", "reason": "No live room with that code."})
        return
    if len(room.members) >= MAX_MEMBERS or (room.task and not room.task.done()):
        await ws.send_json({"type": "error", "reason": "This room is full or already training."})
        return
    if not (isinstance(pub, str) and 0 < len(pub) <= 200 and isinstance(n, int) and 0 < n <= MAX_TRAIN):
        await ws.send_json({"type": "error", "reason": "Malformed join request."})
        return
    member = Member(ws=ws, id=room.next_id, name=f"Hospital {room.next_id}", pub=pub, n_train=n)
    room.next_id += 1
    room.members[member.id] = member
    await ws.send_json({"type": "joined", "id": member.id, "name": member.name, "code": room.code})
    await _broadcast(room, room.roster())
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "update" and member.id in room.waiting and valid_update(msg, room.t):
                room.updates[member.id] = msg
                room.waiting.discard(member.id)
            elif msg.get("type") == "leave":
                break
    except WebSocketDisconnect:
        pass
    finally:
        if room.members.pop(member.id, None):
            await _broadcast(room, room.roster())
        if member.id in room.waiting:
            room.left = True  # end the round early so it reruns without this hospital


@router.websocket("/live/ws")
async def live_socket(ws: WebSocket) -> None:
    if not ORIGIN.fullmatch(ws.headers.get("origin", "")):
        await ws.close(code=1008)
        return
    await ws.accept()
    try:
        hello = await ws.receive_json()
    except (WebSocketDisconnect, ValueError):
        return
    if hello.get("type") == "host":
        await _host(ws)
    elif hello.get("type") == "join":
        await _member(ws, hello)
    else:
        await ws.send_json({"type": "error", "reason": "Unknown hello."})
    try:
        await ws.close()
    except RuntimeError:
        pass  # already closed by the client
