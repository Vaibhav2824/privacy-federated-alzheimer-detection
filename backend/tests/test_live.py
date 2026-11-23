"""Live federation rooms: masks cancel, dropouts rerun the round, joins are checked.

These run a real uvicorn server in a thread and talk to it with real websocket
clients on one event loop, the way browsers do; Starlette's TestClient gives
every socket its own loop, which a multi-hospital room cannot be tested on.
"""

import asyncio
import json
import random
import socket
import threading
import time

import pytest
import uvicorn
import websockets

import app as api
import live

ORIGIN = "http://localhost:5173"


@pytest.fixture(scope="module")
def url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(api.app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.02)
    yield f"ws://127.0.0.1:{port}/live/ws"
    server.should_exit = True
    thread.join(5)


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(live, "ROUNDS", 2)
    monkeypatch.setattr(live, "MIN_ROUND_SECONDS", 0.0)
    monkeypatch.setattr(live, "ROUND_TIMEOUT", 2.0)
    live.ROOMS.clear()


def connect(url, origin=ORIGIN):
    return websockets.connect(url, origin=origin)


async def send(ws, msg):
    await ws.send(json.dumps(msg))


async def until(ws, kind):
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), 5))
        if msg["type"] == kind:
            return msg


async def open_room(url):
    host = await connect(url)
    await send(host, {"type": "host"})
    return host, (await until(host, "room"))["code"]


async def join(url, code, n=50):
    ws = await connect(url)
    await send(ws, {"type": "join", "code": code, "pub": "k", "n_train": n})
    return ws, await until(ws, "joined")


def encode(values):
    return [round(v * live.SCALE) % 2**32 for v in values]


def test_arithmetic_round_trip():
    a, b = [0.5] * live.P, [-0.25] * live.P
    mask = [random.randrange(2**32) for _ in range(live.P)]
    masked_a = [(x + m) % 2**32 for x, m in zip(encode(a), mask)]
    masked_b = [(x - m) % 2**32 for x, m in zip(encode(b), mask)]
    assert live.decode(live.sum_mod([masked_a, masked_b])) == pytest.approx([0.25] * live.P)
    assert live.server_step([1.0, 1.0], [2.0, 4.0], 2.0, 0.0, random.Random(0)) == [2.0, 3.0]
    assert live.epsilon("1", 60) == pytest.approx(1.0, abs=0.01)
    assert live.epsilon("none", 3) is None


def test_update_validation():
    good = {"t": 0, "masked": [0] * live.P, "correct": 3, "total": 5}
    assert live.valid_update(good, 0)
    assert not live.valid_update(good, 1)
    assert not live.valid_update({**good, "masked": [0] * 3}, 0)
    assert not live.valid_update({**good, "masked": [-1] + [0] * (live.P - 1)}, 0)
    assert not live.valid_update({**good, "correct": 6}, 0)
    assert not live.valid_update({**good, "total": "5"}, 0)


def test_full_run_masks_cancel(url):
    async def scenario():
        host, code = await open_room(url)
        (one, joined), (two, _) = await join(url, code), await join(url, code)
        assert joined["name"] == "Hospital 1"
        await send(host, {"type": "start", "budget": "none"})
        for t in range(2):
            r1, r2 = await until(one, "round"), await until(two, "round")
            assert r1["t"] == r2["t"] == t and len(r1["peers"]) == 2
            mask = [random.randrange(2**32) for _ in range(live.P)]
            a, b = encode([0.1] * live.P), encode([0.3] * live.P)
            await send(one, {"type": "update", "t": t, "masked": [(x + m) % 2**32 for x, m in zip(a, mask)],
                             "correct": 4, "total": 5})
            await send(two, {"type": "update", "t": t, "masked": [(x - m) % 2**32 for x, m in zip(b, mask)],
                             "correct": 1, "total": 5})
            progress = await until(host, "progress")
            assert progress["t"] == t + 1 and progress["accuracy"] == pytest.approx(0.5)
            assert progress["server_sum"] == pytest.approx([0.4] * 6)
            assert progress["epsilon"] is None
        done = await until(host, "done")
        # Two rounds of 0.4 summed over the hospitals, over q * 100 expected subjects.
        assert done["w"] == pytest.approx([2 * 0.4 / 50] * live.P)
        for ws in (one, two, host):
            await ws.close()
    asyncio.run(scenario())


def test_leaving_mid_round_reruns_then_stops(url):
    async def scenario():
        host, code = await open_room(url)
        (one, _), (two, _), (three, _) = [await join(url, code) for _ in range(3)]
        await send(host, {"type": "start", "budget": "1"})
        for ws in (one, two, three):
            await until(ws, "round")
        await three.close()  # leaves mid-round
        rerun = await until(one, "round")
        assert rerun["t"] == 0 and len(rerun["peers"]) == 2
        await send(two, {"type": "leave"})
        assert (await until(one, "stopped"))["reason"].startswith("Secure aggregation")
        await one.close()
        await host.close()
    asyncio.run(scenario())


def test_silent_hospital_is_dropped(url, monkeypatch):
    monkeypatch.setattr(live, "ROUND_TIMEOUT", 0.3)

    async def scenario():
        host, code = await open_room(url)
        (one, _), (two, _), (three, _) = [await join(url, code) for _ in range(3)]
        await send(host, {"type": "start", "budget": "10"})
        for ws in (one, two, three):
            await until(ws, "round")
        for ws in (one, two):
            await send(ws, {"type": "update", "t": 0, "masked": [0] * live.P, "correct": 1, "total": 1})
        assert (await until(three, "dropped"))["type"] == "dropped"
        rerun = await until(one, "round")
        assert rerun["t"] == 0 and len(rerun["peers"]) == 2
        await send(host, {"type": "stop"})
        assert (await until(one, "stopped"))["reason"].startswith("The presenter")
        for ws in (one, two, three, host):
            await ws.close()
    asyncio.run(scenario())


def test_join_checks_and_host_leaving(url, monkeypatch):
    async def scenario():
        host, code = await open_room(url)
        for hello, reason in (
            ({"type": "join", "code": "NOPE1", "pub": "k", "n_train": 5}, "No live room"),
            ({"type": "join", "code": code, "pub": "", "n_train": 5}, "Malformed join"),
            ({"type": "hello"}, "Unknown hello"),
        ):
            async with connect(url) as ws:
                await send(ws, hello)
                assert (await until(ws, "error"))["reason"].startswith(reason)
        monkeypatch.setattr(live, "MAX_MEMBERS", 1)
        member, _ = await join(url, code)
        async with connect(url) as late:
            await send(late, {"type": "join", "code": code, "pub": "k", "n_train": 5})
            assert (await until(late, "error"))["reason"].startswith("This room is full")
        monkeypatch.setattr(live, "MAX_ROOMS", 1)
        async with connect(url) as other:
            await send(other, {"type": "host"})
            assert (await until(other, "error"))["reason"].startswith("Too many live rooms")
        await send(host, {"type": "end"})
        assert (await until(member, "closed"))["type"] == "closed"
        await member.close()
    asyncio.run(scenario())


def test_host_disconnect_closes_room(url):
    async def scenario():
        host, code = await open_room(url)
        member, _ = await join(url, code)
        await host.close()
        assert (await until(member, "closed"))["type"] == "closed"
        assert code not in live.ROOMS
        await member.close()
    asyncio.run(scenario())


def test_foreign_origin_is_refused(url):
    async def scenario():
        with pytest.raises(websockets.exceptions.InvalidStatus):
            async with connect(url, origin="https://evil.example"):
                pass
    asyncio.run(scenario())


def test_old_rooms_expire():
    room = live.Room(code="OLD11", host=None)
    room.created -= live.ROOM_TTL + 1
    live.ROOMS["OLD11"] = room
    live._expire(time.monotonic())
    assert "OLD11" not in live.ROOMS
