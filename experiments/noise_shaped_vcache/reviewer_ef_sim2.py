import random, math
random.seed(1)
D, G, N, TR = 128, 32, 512, 20
W, S = 64, 4   # FP16 recent window, FP16 sink tokens

def qg(u, bits):
    lo, hi = min(u), max(u); L = (1 << bits) - 1
    if hi == lo: return list(u)
    s = (hi - lo) / L
    return [lo + s * round((x - lo) / s) for x in u]
def qt(u, bits):
    o = []
    for g in range(0, D, G): o += qg(u[g:g+G], bits)
    return o
def run(V, bits, fb):
    Q, r = [], [0.0]*D
    for t, v in enumerate(V):
        if t < S or t >= N - W:      # kept exact
            Q.append(list(v)); r = [0.0]*D; continue
        u = [v[i] + r[i] for i in range(D)] if fb else v
        q = qt(u, bits)
        if fb: r = [u[i]-q[i] for i in range(D)]
        Q.append(q)
    return Q
def err(V, Q, w):
    e2 = 0.0
    for c in range(D):
        e = sum(w[i]*(Q[i][c]-V[i][c]) for i in range(N)); e2 += e*e
    return math.sqrt(e2)
def norm(w):
    s = sum(w); return [x/s for x in w]
def mk(spike_frac, scale=64.0, nspk=6):
    a = norm([math.exp(-(N-1-i)/scale) for i in range(N)])
    b = [0.0]*N
    for p in random.sample(range(S, N-W), nspk): b[p] = 1.0
    b = norm(b)
    return [(1-spike_frac)*x + spike_frac*y for x, y in zip(a, b)]
def uniform_mid():
    w = [0.0]*N
    for i in range(S, N-W): w[i] = 1.0
    return norm(w)
print("N=%d, FP16 sink=%d, FP16 window=%d, 2-bit g=%d" % (N, S, W, G))
print("%-34s %8s %8s %8s" % ("weights", "FB/noFB", "3b/2b", "FB/3bit"))
cases = [("diffuse (uniform over middle)", lambda: uniform_mid())]
for f in (0.0, 0.1, 0.3, 0.6):
    cases.append(("exp64 tail + spikes mass %.1f" % f, (lambda f=f: mk(f))))
for name, fn in cases:
    a=b=c=0.0
    for _ in range(TR):
        V = [[random.gauss(0,1) for _ in range(D)] for _ in range(N)]
        w = fn()
        a += err(V, run(V,2,False), w); b += err(V, run(V,2,True), w); c += err(V, run(V,3,False), w)
    print("%-34s %8.2f %8.2f %8.2f" % (name, b/a, c/a, b/c))
