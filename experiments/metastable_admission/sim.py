"""E-28 discrete-event simulator: one LLM serving replica with a prefix cache and agent clients.

Version 2 (amendment A1 after the pre-run review R-15). The scheduler follows vLLM v1
(`vllm/v1/core/sched/scheduler.py` and `kv_cache_manager.py`, main branch, read 2026-10-07):
- Running requests are served in admission order. If a request cannot get KV slots, the most
  recently admitted running request is preempted (`self.running.pop()`), and the next try follows.
- A preempted request keeps its output tokens. It frees its KV, which stays in the prefix cache
  until it is evicted, and it goes to the front of the waiting queue. On resume it prefills
  prompt plus output again (prefix hits make this cheap if the cache survived).
- In a step with a preemption, no waiting request is admitted.
- An allocation fails without eviction if the free plus evictable slots are not enough.

Model, in short:
- KV memory is a fixed number of token slots. Each harness has one shared prefix (system prompt
  and tool definitions) that always stays resident. Each idle session keeps one private cache entry
  (the KV of its last turn). Idle entries are evicted in LRU order, from the tail, when an
  allocation needs memory.
- Clients: agent sessions arrive as a Poisson process (open loop). Inside a session, turns are
  closed loop: turn i+1 starts after turn i returns plus a tool gap. Each request has a client
  timeout. On timeout the client cancels the request (the server aborts it; the prefilled prompt
  KV stays in the cache), waits a backoff, and retries. After the last retry fails, the session fails.
- Step time: a roofline-style cost model for one GPU (defaults: an 8B GQA model on one H100).

Random streams are separate, so a spike run and its control run see the same arrival candidates,
the same session content and the same tool gaps (only the extra spike arrivals differ).

Policies (selected by `policy`):
- "fcfs": the default engine behavior.
- "cap": "fcfs" plus a gateway that admits at most `session_cap` sessions. A queued session is
  shed after `gw_patience_s`.
"""
from __future__ import annotations

import csv
import heapq
import random
from collections import OrderedDict, deque
from dataclasses import dataclass, field

SHARED_BY_HARNESS = {"claude_code": 14_000, "smolagents_code": 2_500}
SHARED_DEFAULT = 800


# ----------------------------------------------------------------------------- parameters
@dataclass
class Params:
    # Hardware and model cost (defaults: Llama-3.1-8B class, BF16, one H100 SXM).
    kv_capacity_tok: int = 380_000      # about 50 GB / 131,072 bytes per token
    step_overhead_s: float = 0.006      # weight read and launch cost per step
    prefill_s_per_tok: float = 1 / 30_000   # linear (GEMM) prefill cost
    attn_s_per_tok_ctx: float = 1.3e-9      # prefill attention cost per (new token x context token)
    decode_s_per_ctx_tok: float = 4.4e-8    # KV read per context token per decode step
    decode_s_per_seq: float = 0.0001        # per-sequence compute in a decode step
    token_budget: int = 8192            # max tokens per step (chunked prefill)
    max_num_seqs: int = 256
    max_context_tok: int = 128_000
    out_clip: int = 0                   # 0: use trace output lengths; >0: clip each output

    # Clients.
    timeout_s: float = 120.0
    max_retries: int = 2
    backoff_base_s: float = 1.0         # attempt k waits base * 2**k * U(0.5, 1.5)
    tool_gap_cap_s: float = 120.0
    gap_scale: float = 1.0              # multiplies every sampled tool gap

    # Workload.
    base_rate: float = 0.05             # sessions per second
    spike_mult: float = 3.0
    spike_start_s: float = 2400.0
    spike_len_s: float = 60.0           # 0 means no spike (control run)
    horizon_s: float = 6000.0
    seed: int = 1
    thin_mult: float = 0.0              # thinning bound as a multiple of base_rate (0: spike_mult)

    # Policy.
    policy: str = "fcfs"
    session_cap: int = 0                # for policy "cap"
    gw_patience_s: float = 1e9          # gateway queue patience for policy "cap"


# ----------------------------------------------------------------------------- workload
@dataclass
class Turn:
    prompt: int     # tokens of the full prompt (shared prefix included)
    out: int
    gap: float = 0.0  # tool and think time after this turn returns


@dataclass
class SessionSpec:
    harness: str
    shared: int
    turns: list


def shared_for(harness: str) -> int:
    return SHARED_BY_HARNESS.get(harness, SHARED_DEFAULT)


def clean_calls(calls):
    """The calls that the simulator replays: drop parallel subagent calls and spans with no usage."""
    calls = sorted(calls, key=lambda r: int(r["call_idx"]))
    return [r for r in calls if not int(r["overlaps_prev"]) and int(r["in_tok"]) > 0]


def load_sessions(path: str, max_context: int, keep=None):
    """Read session_calls.csv (extract_sessions.py). `keep(clean_calls) -> bool` selects sessions
    for a workload mix. Tool gaps are pooled from harnesses whose spans have real durations."""
    rows: dict[str, list[dict]] = {}
    gaps: list[float] = []
    for r in csv.DictReader(open(path)):
        rows.setdefault(r["session_id"], []).append(r)
        if r["harness"] != "claude_code" and r["gap_before_s"] and float(r["gap_before_s"]) > 0:
            gaps.append(float(r["gap_before_s"]))
    specs = []
    for calls in rows.values():
        cc = clean_calls(calls)
        if len(cc) < 2 or (keep is not None and not keep(cc)):
            continue
        h = cc[0]["harness"]
        sh = shared_for(h)
        turns = [Turn(min(max(int(r["in_tok"]), sh + 2), max_context), max(int(r["out_tok"]), 1))
                 for r in cc]
        specs.append(SessionSpec(h, sh, turns))
    return specs, gaps


# ----------------------------------------------------------------------------- simulator state
@dataclass
class Session:
    sid: int
    shared: int
    turns: list
    idx: int = 0
    cached: int = 0         # private KV tokens in the idle cache (0 when absent or in use)
    admitted: bool = False
    t_arrive: float = 0.0


@dataclass
class Req:
    sess: Session
    prompt_priv: int        # private tokens to compute (grows by the output after a preemption)
    out: int                # output tokens still to decode
    attempt: int
    t_submit: float
    orig_prompt: int = 0    # private prompt tokens of the original request
    computed: int = 0       # private tokens whose KV exists
    decoded: int = 0
    dead: bool = False      # cancelled by the client
    finished: bool = False

    @property
    def kv(self) -> int:
        return self.computed + self.decoded


class Sim:
    def __init__(self, p: Params, specs: list[SessionSpec], gaps: list[float]):
        self.p = p
        self.pool = specs
        self.gaps = [g for g in gaps if g <= p.tool_gap_cap_s]
        self.t = 0.0
        self.ev: list = []          # (time, seq, kind, payload)
        self.seq = 0
        self.waiting: deque[Req] = deque()
        self.running: list[Req] = []   # admission order
        self.lru: OrderedDict[int, Session] = OrderedDict()   # idle caches, oldest first
        self.idle = 0               # private KV tokens in idle caches
        self.used = 0               # private KV tokens in use (running plus idle cache)
        reserved = sum({s.harness: s.shared for s in specs}.values())
        self.cap = p.kv_capacity_tok - reserved
        self.sessions_alive: dict[int, Session] = {}
        self.admit_q: deque[Session] = deque()
        self.n_admitted = 0
        self.n_cand = 0             # arrival candidates (the same in a spike run and its control)
        self.arr_rng = random.Random(f"arrivals-{p.seed}")
        self.bins: dict[int, dict] = {}

    # --- helpers
    def push(self, t, kind, x):
        self.seq += 1
        heapq.heappush(self.ev, (t, self.seq, kind, x))

    def b(self) -> dict:
        k = int(self.t // 10)
        if k not in self.bins:
            self.bins[k] = dict(turn_ok=0, sess_ok=0, sess_fail=0, timeouts=0, arrivals=0,
                                admitted=0, gw_wait=0.0, shed=0, prefill_tok=0, hit_tok=0,
                                preempt=0, busy=0.0, active=0, waiting=0, running=0, admitq=0)
        return self.bins[k]

    def rate(self, t):
        p = self.p
        if p.spike_len_s > 0 and p.spike_start_s <= t < p.spike_start_s + p.spike_len_s:
            return p.base_rate * p.spike_mult
        return p.base_rate

    def make_session(self, cand: int) -> Session:
        rng = random.Random(f"session-{self.p.seed}-{cand}")
        spec = rng.choice(self.pool)
        g, c = self.p.gap_scale, self.p.out_clip
        turns = [Turn(x.prompt, min(x.out, c) if c else x.out, g * rng.choice(self.gaps))
                 for x in spec.turns]
        return Session(cand, spec.shared, turns, t_arrive=self.t)

    # --- memory
    def free(self) -> int:
        return self.cap - self.used

    def evict(self, need: int):
        """Free idle cache from the LRU end until `need` slots are free. Tail-first per session.
        The caller has checked that free + evictable >= need."""
        while self.free() < need:
            sid, s = next(iter(self.lru.items()))
            take = min(s.cached, need - self.free())
            s.cached -= take
            self.used -= take
            self.idle -= take
            if s.cached == 0:
                del self.lru[sid]
        assert self.free() >= need

    def to_cache(self, s: Session, tokens: int):
        """Make `tokens` private KV of session s (already counted in `used`) an idle entry (MRU)."""
        assert s.cached == 0 and s.sid not in self.lru
        if tokens <= 0:
            return
        s.cached = tokens
        self.idle += tokens
        self.lru[s.sid] = s

    def take_cache(self, s: Session) -> int:
        c = s.cached
        if c:
            del self.lru[s.sid]
            self.idle -= c
        s.cached = 0
        return c

    # --- clients
    def new_session(self, cand: int):
        s = self.make_session(cand)
        self.b()["arrivals"] += 1
        self.sessions_alive[s.sid] = s
        if self.p.policy == "cap" and self.n_admitted >= self.p.session_cap:
            self.admit_q.append(s)
            self.push(self.t + self.p.gw_patience_s, "gw_timeout", s)
            return
        self.admit(s)

    def admit(self, s: Session):
        s.admitted = True
        self.n_admitted += 1
        bb = self.b()
        bb["admitted"] += 1
        bb["gw_wait"] += self.t - s.t_arrive
        self.submit(s, attempt=0)

    def submit(self, s: Session, attempt: int):
        tr = s.turns[s.idx]
        priv = tr.prompt - s.shared
        r = Req(s, priv, tr.out, attempt, self.t, orig_prompt=priv)
        self.waiting.append(r)
        self.push(self.t + self.p.timeout_s, "timeout", r)

    def end_session(self, s: Session, ok: bool):
        self.b()["sess_ok" if ok else "sess_fail"] += 1
        del self.sessions_alive[s.sid]
        if s.admitted:
            self.n_admitted -= 1
            if self.admit_q:
                self.admit(self.admit_q.popleft())

    def on_gw_timeout(self, s: Session):
        if s.admitted or s.sid not in self.sessions_alive:
            return
        self.admit_q.remove(s)
        self.b()["shed"] += 1
        self.end_session(s, ok=False)

    def on_timeout(self, r: Req):
        if r.dead or r.finished:
            return
        r.dead = True
        self.b()["timeouts"] += 1
        if r in self.running:
            # Abort. The prompt KV stays in the prefix cache; the rest is freed.
            self.running.remove(r)
            keep = min(r.computed, r.orig_prompt)
            self.used -= r.kv - keep
            self.to_cache(r.sess, keep)
        else:
            self.waiting.remove(r)   # a waiting request holds no KV
        s = r.sess
        if r.attempt < self.p.max_retries:
            rng = random.Random(f"jitter-{self.p.seed}-{s.sid}-{s.idx}-{r.attempt}")
            d = self.p.backoff_base_s * (2 ** r.attempt) * rng.uniform(0.5, 1.5)
            self.push(self.t + d, "retry", (s, r.attempt + 1))
        else:
            self.end_session(s, ok=False)

    def on_done(self, r: Req):
        r.finished = True
        self.b()["turn_ok"] += 1
        s = r.sess
        self.running.remove(r)
        self.to_cache(s, r.kv)
        s.idx += 1
        if s.idx >= len(s.turns):
            self.end_session(s, ok=True)
        else:
            self.push(self.t + s.turns[s.idx - 1].gap, "turn", s)

    # --- server step
    def preempt(self, r: Req):
        """vLLM v1 recompute preemption: free the KV (it stays as evictable cache), keep the
        output tokens, and prefill prompt plus output again on resume."""
        self.running.remove(r)
        kv = r.kv
        r.prompt_priv += r.decoded
        r.out -= r.decoded
        r.computed = r.decoded = 0
        self.to_cache(r.sess, kv)
        self.waiting.appendleft(r)
        self.b()["preempt"] += 1

    def step(self):
        p = self.p
        budget = p.token_budget
        sched: dict[int, tuple] = {}   # id(req) -> (req, n, is_prefill)
        preempted = False
        # 1. Running requests in admission order.
        for r in list(self.running):
            if budget <= 0:
                break
            if r not in self.running:
                continue
            is_pre = r.computed < r.prompt_priv
            n = min(budget, r.prompt_priv - r.computed) if is_pre else 1
            while self.free() + self.idle < n:
                victim = self.running[-1]
                if id(victim) in sched:   # give back what the victim got in this step
                    _, vn, _ = sched.pop(id(victim))
                    self.used -= vn
                    budget += vn
                self.preempt(victim)
                preempted = True
                if victim is r:
                    break
            if r not in self.running:
                continue
            self.evict(n)
            self.used += n
            sched[id(r)] = (r, n, is_pre)
            budget -= n
        # 2. Waiting requests in FCFS order, only in a step with no preemption.
        while (not preempted and self.waiting and budget > 0
               and len(self.running) < p.max_num_seqs):
            r = self.waiting[0]
            s = r.sess
            hit = min(s.cached, r.prompt_priv - 1)   # the last prompt token is always computed
            n = min(budget, r.prompt_priv - hit)
            if self.free() + self.idle - hit < n:   # the unmatched cache tail is freed below
                break                                 # cannot allocate: no eviction
            cached = self.take_cache(s)
            self.used -= cached - hit                 # cached tokens past the prompt do not match
            self.evict(n)
            self.waiting.popleft()
            r.computed = hit
            self.running.append(r)
            self.used += n
            self.b()["hit_tok"] += hit
            sched[id(r)] = (r, n, True)
            budget -= n
        # 3. Step time (roofline-style cost model).
        t = p.step_overhead_s
        for r, n, is_pre in sched.values():
            if is_pre:
                ctx = r.sess.shared + r.computed
                t += n * p.prefill_s_per_tok + p.attn_s_per_tok_ctx * n * (ctx + n / 2)
            else:
                t += p.decode_s_per_seq + p.decode_s_per_ctx_tok * (r.sess.shared + r.kv)
        # 4. Apply progress at the end of the step.
        done = []
        for r, n, is_pre in sched.values():
            if is_pre:
                r.computed += n
                self.b()["prefill_tok"] += n
            else:
                r.decoded += 1
                if r.decoded >= r.out:
                    done.append(r)
        return t, done, bool(sched)

    # --- main loop
    def next_arrival(self, t0: float) -> float:
        """Thinning for the non-homogeneous Poisson process. The candidate stream does not depend
        on the rate, so a spike run and its control share all candidates."""
        m = self.p.thin_mult if self.p.thin_mult > 0 else self.p.spike_mult
        assert m >= self.p.spike_mult or self.p.spike_len_s == 0
        lam_max = self.p.base_rate * max(1.0, m)
        t = t0
        while True:
            t += self.arr_rng.expovariate(lam_max)
            u = self.arr_rng.random()
            self.n_cand += 1
            if u < self.rate(t) / lam_max:
                return t, self.n_cand

    def process_events(self, until: float):
        """Handle client events with time <= until, each at its own event time."""
        while self.ev and self.ev[0][0] <= until:
            te, _, kind, x = heapq.heappop(self.ev)
            self.t = max(self.t, te)
            if kind == "arrive":
                self.new_session(x)
                tn, cand = self.next_arrival(te)
                self.push(tn, "arrive", cand)
            elif kind == "timeout":
                self.on_timeout(x)
            elif kind == "retry":
                s, att = x
                self.submit(s, att)
            elif kind == "turn":
                self.submit(x, 0)
            elif kind == "gw_timeout":
                self.on_gw_timeout(x)
        self.t = max(self.t, until)

    def run(self):
        p = self.p
        tn, cand = self.next_arrival(0.0)
        self.push(tn, "arrive", cand)
        while self.t < p.horizon_s:
            if self.running or self.waiting:
                t_start = self.t
                dt, done, busy = self.step()
                # Client events inside the step happen at their own time. Timeouts that fall
                # inside the step win over a completion at the step end.
                self.t = t_start
                self.process_events(t_start + dt)
                if busy:
                    self.b()["busy"] += dt
                for r in done:
                    if not r.dead:
                        self.on_done(r)
            else:
                self.process_events(min(self.ev[0][0], p.horizon_s) if self.ev else p.horizon_s)
            bb = self.b()
            bb["active"] = len(self.sessions_alive)
            bb["waiting"] = len(self.waiting)
            bb["running"] = len(self.running)
            bb["admitq"] = len(self.admit_q)
        return self.bins
