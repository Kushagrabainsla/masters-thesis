"""Unit tests for sim.py (version 2). They check the code, not the E-28 hypothesis.

Run: python3 -I test_sim.py   (standard library only)
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import Params, Req, Session, SessionSpec, Sim, Turn  # noqa: E402

SH = 4000


def check_invariant(sim):
    run_kv = sum(r.kv for r in sim.running)
    idle = sum(s.cached for s in sim.lru.values())
    assert idle == sim.idle, (idle, sim.idle)
    assert sim.used == run_kv + idle, (sim.used, run_kv, idle)
    assert 0 <= sim.used <= sim.cap, (sim.used, sim.cap)
    for s in sim.lru.values():
        assert s.cached > 0
    for r in sim.waiting:
        assert r.kv == 0, "a waiting request holds no KV"
    for r in sim.running:
        assert r.decoded <= r.out


class Checked(Sim):
    def step(self):
        out = super().step()
        check_invariant(self)
        return out


def spec(turns):
    return SessionSpec("test", SH, turns)


def manual(turns_list, **kw):
    """A simulator with no random arrivals; sessions are injected by hand."""
    p = Params(base_rate=1e-9, spike_len_s=0, horizon_s=kw.pop("horizon_s", 600), **kw)
    sim = Checked(p, [spec(turns_list[0])], [1.0])
    sim.next_arrival = lambda t0: (1e18, 0)
    for i, turns in enumerate(turns_list):
        s = Session(1000 + i, SH, [Turn(t.prompt, t.out, t.gap) for t in turns])
        sim.sessions_alive[s.sid] = s
        sim.admit(s)
    return sim


def total(sim, key):
    return sum(x[key] for x in sim.bins.values())


def test_single_turn():
    sim = manual([[Turn(12_000, 10)]])
    sim.run()
    assert total(sim, "turn_ok") == 1 and total(sim, "sess_ok") == 1
    assert total(sim, "prefill_tok") == 8_000
    assert sim.used == 8_000 + 10


def test_second_turn_hits_cache():
    sim = manual([[Turn(12_000, 100, 1.0), Turn(12_400, 50)]])
    sim.run()
    assert total(sim, "hit_tok") == 8_100
    assert total(sim, "prefill_tok") == 8_000 + 300


def test_full_hit_computes_last_token():
    sim = manual([[Turn(12_000, 10, 1.0), Turn(12_000, 10)]])
    sim.run()
    assert total(sim, "prefill_tok") == 8_000 + 1


def test_blocked_head_does_not_evict():
    # R-15 B2: a waiting head that cannot fit must not flush the idle caches.
    p = Params(kv_capacity_tok=SH + 20_000, base_rate=1e-9, spike_len_s=0)
    sim = Sim(p, [spec([Turn(5_000, 1)])], [1.0])
    a, b = Session(1, SH, []), Session(2, SH, [])
    sim.used = 4_000
    sim.to_cache(a, 2_000)
    sim.to_cache(b, 2_000)
    big = Session(3, SH, [Turn(SH + 30_000, 1)])
    sim.waiting.append(Req(big, 30_000, 1, 0, 0.0, orig_prompt=30_000))
    # Free 16,000 + idle 4,000 < first chunk 8,192? No: 20,000 >= 8,192, so it is admitted.
    # Make it infeasible: occupy free space.
    sim.used += 15_000
    sim.step()
    assert a.cached == 2_000 and b.cached == 2_000, (a.cached, b.cached)
    assert len(sim.waiting) == 1


def test_preemption_keeps_output():
    # R-15 B3: a preempted request re-prefills prompt plus output and never decodes more than out.
    cap = SH + 2 * 5_000 + 1_500
    sim = manual([[Turn(SH + 5_000, 3_000)], [Turn(SH + 5_000, 3_000)]], kv_capacity_tok=cap)
    seen = []
    orig = sim.preempt

    def pre(r):
        seen.append((r.sess.sid, r.decoded))
        orig(r)
    sim.preempt = pre
    sim.run()
    assert total(sim, "sess_ok") == 2
    assert seen, "expected a preemption"
    # Victim is the most recently admitted session (1001).
    assert seen[0][0] == 1001
    decode_steps = total(sim, "turn_ok")
    assert decode_steps == 2


def test_no_admission_in_preemption_step():
    cap = SH + 9_000
    p = Params(kv_capacity_tok=cap, base_rate=1e-9, spike_len_s=0)
    sim = Sim(p, [spec([Turn(5_000, 1)])], [1.0])
    s1, s2, s3 = (Session(i, SH, [Turn(SH + 4_500, 100)]) for i in (1, 2, 3))
    for s in (s1, s2):
        r = Req(s, 4_500, 100, 0, 0.0, orig_prompt=4_500, computed=4_500)
        sim.running.append(r)
        sim.used += 4_500
    sim.waiting.append(Req(s3, 100, 10, 0, 0.0, orig_prompt=100))
    sim.step()
    assert sim.b()["preempt"] == 1
    assert all(r.sess is not s3 for r in sim.running), "no admission in a preemption step"


def test_timeout_and_retry_fail():
    sim = manual([[Turn(5_000, 100_000)]], timeout_s=5.0, max_retries=2)
    sim.run()
    assert total(sim, "timeouts") == 3
    assert total(sim, "sess_fail") == 1
    assert sim.running == [] and not sim.waiting
    assert total(sim, "hit_tok") == 2 * 999   # prompt KV survives the abort


def test_late_completion_is_timeout():
    # R-15 m2: a turn that ends after its deadline is a timeout, not a success.
    p_tok = SH + 8_000          # prefill about 0.37 s, then 20 decode steps
    sim = manual([[Turn(p_tok, 20)]], timeout_s=0.3, max_retries=0)
    sim.run()
    assert total(sim, "turn_ok") == 0 and total(sim, "timeouts") == 1


def pool(seed=0, n=60):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        k = rng.randint(2, 25)
        base = rng.randint(1_000, 20_000)
        out.append(SessionSpec("test", SH, [Turn(SH + base + 900 * i, rng.randint(5, 600))
                                            for i in range(k)]))
    return out


def test_random_load_invariants():
    for cap in (40_000, 120_000):
        for pol, sc in (("fcfs", 0), ("cap", 3)):
            p = Params(base_rate=0.2, spike_mult=4, spike_start_s=100, spike_len_s=50,
                       horizon_s=500, kv_capacity_tok=cap, timeout_s=20, seed=3, policy=pol,
                       session_cap=sc, gw_patience_s=30)
            sim = Checked(p, pool(), [0.5, 2.0, 10.0])
            sim.run()
            assert total(sim, "turn_ok") > 0


def test_paired_streams():
    # R-15 M3: spike and control runs share arrivals outside the spike and session content.
    def arrivals(spike_len):
        p = Params(base_rate=0.1, spike_mult=5, spike_start_s=200, spike_len_s=spike_len,
                   horizon_s=600, kv_capacity_tok=10_000_000, seed=7)
        sim = Sim(p, pool(), [1.0])
        rec = []
        orig = sim.new_session

        def ns(c):
            orig(c)
            rec.append((round(sim.t, 6), c, len(sim.sessions_alive[c].turns)))
        sim.new_session = ns
        sim.run()
        return rec
    a, b = arrivals(60), arrivals(0)
    out = lambda rec: [x for x in rec if not 200 <= x[0] < 260]
    assert out(a) == out(b), "arrivals outside the spike must match"
    assert set(b) <= set(a), "every control arrival also happens in the spike run"
    assert len(a) > len(b)


def test_cap_limits_and_sheds():
    pl = [SessionSpec("test", SH, [Turn(6_000, 50, 0)] * 10)]
    p = Params(base_rate=0.5, spike_len_s=0, horizon_s=200, policy="cap", session_cap=1,
               gw_patience_s=10.0, gap_scale=1.0)
    sim = Checked(p, pl, [30.0])
    peak = []
    orig = sim.admit

    def admit(s):
        orig(s)
        peak.append(sim.n_admitted)
    sim.admit = admit
    sim.run()
    assert max(peak) <= 1
    assert total(sim, "shed") > 10


def test_deterministic():
    def once():
        p = Params(base_rate=0.2, horizon_s=400, spike_start_s=100, kv_capacity_tok=60_000,
                   timeout_s=15, seed=5)
        return Sim(p, pool(), [0.5, 3.0]).run()
    assert once() == once()


def test_knee_rule_passes_stable_system():
    # R-15 N1: on a stable system with window noise CV 0.21, the knee rule must pass a load
    # at least 90% of the time and must not stop the scan early.
    import run_e28
    rng = random.Random(0)
    g = 1.0

    def rows(f, healthy=True):
        out = []
        for _ in range(10):
            mu = f * g if healthy else 0.5 * f * g
            out.append({"late_turn_ok": mu * rng.gauss(1, 0.21), "pre_turn_ok": f * g * rng.gauss(1, 0.21),
                        "late_timeouts": 0.0, "late_preempt": 0.0})
        return out
    passes = sum(run_e28.knee_ok(rows(0.3), 0.3 * g) for _ in range(500))
    assert passes >= 450, passes
    by = {f: rows(f, healthy=f <= 0.7) for f in run_e28.KNEE_GRID}
    k = run_e28.knee_from(by, g)
    assert 0.6 <= k <= 0.7, k


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
