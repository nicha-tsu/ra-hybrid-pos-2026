"""Round-3 experiments requested by reviewers (checklist items 6-12, 17).
Main protocol: global temporal split (cut 2023-08-01), validation May-Jul 2023 for all tuning.
C16 tuned Personal | C15 TIFU-KNN, UP-CF@r | C11 ablation of p_u | C14 5 seeds | C18 ex-ante groups + switching
C27 2+1 slots | C26 timing | C17 cold-start | C19 P-TopFreq, NDCG@5, MRR"""
import os, sys, time, pickle, itertools, platform
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp2
from exp2 import *

# ---------- extra metrics
D5 = DISC[:5]
def evaluate3(S, te, X):
    r = evaluate(S, te, X)
    S = S[te.ui.values]; order = np.argsort(-S, 1)[:, :10]
    n5, mrr = [], []
    for top, its in zip(order, te['items'].values):
        hit = np.isin(top, its)
        n5.append((hit[:5] * D5).sum() / D5[:min(len(its), 5)].sum())
        f = np.where(hit)[0]; mrr.append(1 / (f[0] + 1) if len(f) else 0.0)
    r['ndcg5'] = np.array(n5); r['mrr'] = np.array(mrr); return r

# ---------- tunable Personal (C16): X + beta * sum_r rho^r
def add_last(c, tr, m, rho):
    last = np.zeros((c.U, c.I)); r = tr.groupby('ui').cumcount(ascending=False).values
    for ui, its, rr in zip(tr.ui.values, tr['items'].values, r): last[ui, its] += rho ** rr
    m['last'] = last; return m
def personal_t(c, m, te, beta=0.5, **k):
    X = m['X']; return X + beta * m['last'] + 1e-6 * X.sum(0)[None, :] / max(X.sum(), 1)
def ptop(c, m, te, **k):
    S = m['X'].astype(float).copy(); S[S == 0] = -np.inf; return S

# ---------- user basket sequences
def seqs(c, tr):
    d = {}
    for ui, its in zip(tr.ui.values, tr['items'].values): d.setdefault(ui, []).append(its)
    return d

def tifu_vectors(c, sq, gsize, rb, rg):
    V = np.zeros((c.U, c.I))
    for u, bl in sq.items():
        groups = [bl[i:i + gsize] for i in range(0, len(bl), gsize)]
        G = len(groups); acc = np.zeros(c.I)
        for gi, g in enumerate(groups):
            gv = np.zeros(c.I); L = len(g)
            for j, its in enumerate(g): gv[its] += rb ** (L - 1 - j)
            acc += rg ** (G - 1 - gi) * gv / L
        V[u] = acc / G
    return V
def tifu(c, m, te, V=None, k=300, alpha=0.7, **kw):
    """TIFU-KNN (Hu et al., SIGIR 2020): alpha * own time-decayed vector + (1-alpha) * mean of k nearest users."""
    q = te.ui.values; n = np.linalg.norm(V, axis=1).clip(1e-12); Vn = V / n[:, None]
    sim = Vn[q] @ Vn.T; sim[np.arange(len(q)), q] = -1
    idx = np.argpartition(-sim, k, 1)[:, :k]
    W = np.zeros((len(q), V.shape[0])); np.put_along_axis(W, idx, 1.0 / k, 1); nb = W @ V
    S = np.zeros((c.U, c.I)); S[q] = alpha * V[q] + (1 - alpha) * nb; return S

def upcf_vectors(c, sq, r):
    P = np.zeros((c.U, c.I))
    for u, bl in sq.items():
        bl = bl if r is None else bl[-r:]
        for its in bl: P[u, its] += 1
        P[u] /= len(bl)
    return P
def upcf(c, m, te, P=None, a=0.75, q=5, lam=0.5, **kw):
    """UP-CF@r (Faggioli et al., UMAP 2020): recency-aware user popularity P_u^r plus asymmetric-cosine
    user-based CF with locality q; score = P_u + lam * sum_v sim(u,v)^q P_v."""
    qs = te.ui.values; n = np.linalg.norm(P, axis=1).clip(1e-12)
    sim = (P[qs] @ P.T) / (n[qs, None] ** (2 * a) * n[None, :] ** (2 * (1 - a)))
    sim[np.arange(len(qs)), qs] = 0; sim = np.maximum(sim, 0) ** q
    cf = sim @ P; cf = cf / cf.max(1, keepdims=True).clip(1e-12)
    S = np.zeros((c.U, c.I)); S[qs] = P[qs] + lam * cf; return S

def ra_const(c, m, te, pbar=None, **kw):   # ablation: p_u replaced by a constant
    m2 = dict(m); m2['prop'] = np.full(c.U, pbar); return ra_hybrid(c, m2, te, **kw)

def nd(S, te, X): return evaluate(S, te, X)['ndcg'].mean()

def setup(o, cut=MAIN_CUT, val_lo=VAL_CUT):
    B, items, U = build(o); c = Ctx(B, len(items), U)
    return B, items, c

def tune_all(c, B):
    trv, tev = temporal(B, MAIN_CUT, lo=VAL_CUT); mv = mats(c, trv); X = mv['X']; T = {}
    # C16 Personal
    best = (-1, None)
    for rho in [0.5, 0.7, 0.8, 0.9, 1.0]:
        add_last(c, trv, mv, rho)
        for beta in [0, 0.25, 0.5, 1, 2, 4, 8]:
            s = nd(personal_t(c, mv, tev, beta), tev, X)
            if s > best[0]: best = (s, dict(rho=rho, beta=beta))
    T['pers'] = best[1]; add_last(c, trv, mv, T['pers']['rho'])
    exp2.personal = lambda c, m, te, **k: personal_t(c, m, te, beta=T['pers']['beta'])   # used inside ctx/ra hybrids
    T['ctx'] = max([dict(g=g) for g in [0, 0.15, 0.3, 0.6, 1, 1.5, 2.5]], key=lambda p: nd(ctx_hybrid(c, mv, tev, **p), tev, X))
    T['lam'] = max([0.1, 0.3, 1, 3, 10, 50, 200, 1000], key=lambda l: nd(ease(c, mv, tev, l), tev, X))
    T['knn_k'] = max([3, 5, 10, 30, 100, 0], key=lambda k: nd(itemknn(c, mv, tev, k), tev, X))
    grid = [dict(base=b, e=e, g=g, lam=T['lam'], k=T['knn_k']) for b in ['ease', 'knn'] for e in [0, 0.1, 0.25, 0.5, 1, 2] for g in [0, 0.15, 0.3, 0.6, 1]]
    T['ra'] = max(grid, key=lambda p: nd(ra_hybrid(c, mv, tev, **p), tev, X))
    pbar = float(mv['prop'][np.unique(trv.ui)].mean()); T['pbar'] = pbar
    T['ra_const'] = max([dict(T['ra'], g=g, pbar=pbar) for g in [0, 0.05, 0.1, 0.15, 0.3, 0.6, 1]], key=lambda p: nd(ra_const(c, mv, tev, **p), tev, X))
    # C15 TIFU-KNN
    sq = seqs(c, trv); best = (-1, None)
    for gs, rb, rg in itertools.product([3, 7], [0.9, 1.0], [0.6, 0.9]):
        V = tifu_vectors(c, sq, gs, rb, rg)
        for k, al in itertools.product([100, 300], [0.5, 0.7, 0.9]):
            s = nd(tifu(c, mv, tev, V, k, al), tev, X)
            if s > best[0]: best = (s, dict(gsize=gs, rb=rb, rg=rg, k=k, alpha=al))
    T['tifu'] = best[1]
    best = (-1, None)
    for r in [1, 3, 5, None]:
        P = upcf_vectors(c, sq, r)
        for a, q, lam in itertools.product([0.5, 0.75], [1, 5], [0.1, 0.5, 1.0]):
            s = nd(upcf(c, mv, tev, P, a, q, lam), tev, X)
            if s > best[0]: best = (s, dict(r=r, a=a, q=q, lam=lam))
    T['upcf'] = best[1]
    # C18 switching threshold (ex-ante, on validation): Popularity if p_u < tau else RA-Hybrid
    Sp, Sr = pop(c, mv, tev), ra_hybrid(c, mv, tev, **T['ra'])
    def sw(tau):
        lowp = mv['prop'] < tau; S = Sr.copy(); S[lowp] = Sp[lowp]; return S
    T['tau'] = max([0, 0.1, 0.2, 0.3, 0.4, 0.5], key=lambda t: nd(sw(t), tev, X))
    return T

def models(c, tr, te, T, timing=False):
    m = mats(c, tr); add_last(c, tr, m, T['pers']['rho']); sq = seqs(c, tr)
    exp2.personal = lambda c, m, te, **k: personal_t(c, m, te, beta=T['pers']['beta'])
    F = {'Popularity': lambda: pop(c, m, te), 'Hour-Popularity': lambda: hourpop(c, m, te),
         'P-TopFreq': lambda: ptop(c, m, te),
         'Personal (tuned)': lambda: personal_t(c, m, te, **{'beta': T['pers']['beta']}),
         'TIFU-KNN': lambda: tifu(c, m, te, tifu_vectors(c, sq, T['tifu']['gsize'], T['tifu']['rb'], T['tifu']['rg']), T['tifu']['k'], T['tifu']['alpha']),
         'UP-CF@r': lambda: upcf(c, m, te, upcf_vectors(c, sq, T['upcf']['r']), T['upcf']['a'], T['upcf']['q'], T['upcf']['lam']),
         'Context Hybrid': lambda: ctx_hybrid(c, m, te, **T['ctx']),
         'RA-Hybrid': lambda: ra_hybrid(c, m, te, **T['ra']),
         'RA: p_u=const': lambda: ra_const(c, m, te, **T['ra_const']),
         'RA: gamma=0': lambda: ra_hybrid(c, m, te, **dict(T['ra'], g=0)),
         'Switch (Pop if p_u<tau)': None}
    S, tm = {}, {}
    for k, f in F.items():
        if f is None: continue
        t0 = time.perf_counter(); S[k] = f(); tm[k] = time.perf_counter() - t0
    lowp = m['prop'] < T['tau']; Ssw = S['RA-Hybrid'].copy(); Ssw[lowp] = S['Popularity'][lowp]; S['Switch (Pop if p_u<tau)'] = Ssw
    return S, m, tm

if __name__ == '__main__':
    out = {}
    o = load_orders(); B, items, c = setup(o)
    t0 = time.time(); T = tune_all(c, B); print('T', T, '%.0fs' % (time.time() - t0), flush=True)
    tr, te = temporal(B, MAIN_CUT)
    S, m, tm = models(c, tr, te, T); X = m['X']
    R = {k: evaluate3(v, te, X) for k, v in S.items()}
    for k, v in R.items():
        print(f"{k:26s} NDCG {v['ndcg'].mean():.4f} N@5 {v['ndcg5'].mean():.4f} MRR {v['mrr'].mean():.4f} HR3 {v['hr3'].mean():.3f} "
              f"HR5 {v['hr5'].mean():.3f} HR10 {v['hr10'].mean():.3f} RecRep {np.nanmean(v['rec_rep']):.3f} RecExp {np.nanmean(v['rec_exp']):.3f} t {tm.get(k, np.nan):.2f}s", flush=True)
    # paired tests vs tuned Personal
    base = 'Personal (tuned)'; names = [k for k in R if k != base]
    P = pd.DataFrame([{'A': a, 'B': base, **paired(R[a]['ndcg'], R[base]['ndcg'])} for a in names])
    P['w_p_holm'] = holm(P.w_p_pratt.values); P['t_p_holm'] = holm(P.t_p.values)
    print(P[['A', 'mean', 'ci', 'w_p_pratt', 't_p', 'w_p_holm', 't_p_holm', 'win', 'tie', 'loss']].to_string(), flush=True)
    # C18 ex-ante groups by training repeat propensity p_u
    pu = m['prop'][te.ui.values]
    bins = [(0, .2), (.2, .4), (.4, .6), (.6, 1.01)]
    G = []
    for lo, hi in bins:
        mk = (pu >= lo) & (pu < hi)
        G.append(dict(group=f'[{lo},{min(hi,1)})', n=int(mk.sum()), **{k: R[k]['ndcg'][mk].mean() for k in ['Popularity', 'Personal (tuned)', 'Context Hybrid', 'RA-Hybrid', 'TIFU-KNN', 'UP-CF@r']}))
    G = pd.DataFrame(G); print(G.round(3).to_string(), flush=True)
    # C27 2+1 slots at K=3: two best history items from RA-Hybrid + best new item by hour-popularity
    q = te.ui.values; hist = X[q] > 0
    Sra = S['RA-Hybrid'][q]; Shp = S['Hour-Popularity'][q]
    rows = []
    def score3(tops, name):
        hr = []; rr = []; re_ = []; anynew = []
        for k, (top, its) in enumerate(zip(tops, te['items'].values)):
            hit = np.isin(top, its); hr.append(hit.any())
            rep = its[hist[k][its]]; ex = its[~hist[k][its]]
            rr.append(np.isin(rep, top).mean() if len(rep) else np.nan); re_.append(np.isin(ex, top).mean() if len(ex) else np.nan)
            anynew.append(np.isin(top[~hist[k][top]], its).any())
        return dict(slots=name, HR3=np.mean(hr), Rec_rep3=np.nanmean(rr), Rec_expl3=np.nanmean(re_), new_hit=np.mean(anynew), hit=np.array(hr, float))
    ra3 = np.argsort(-Sra, 1)[:, :3]
    t21 = []
    for k in range(len(q)):
        hrank = [i for i in np.argsort(-np.where(hist[k], Sra[k], -np.inf)) if hist[k][i]][:2]
        nrank = [i for i in np.argsort(-np.where(hist[k], -np.inf, Shp[k])) if not hist[k][i] and i not in hrank]
        pick = hrank + nrank[:3 - len(hrank)]
        t21.append(pick[:3])
    t21 = np.array(t21)
    pop3 = np.argsort(-S['Popularity'][q], 1)[:, :3]
    slots = [score3(ra3, 'RA-Hybrid top-3'), score3(t21, '2 repeat + 1 new'), score3(pop3, 'Popularity top-3')]
    for s in slots: print({k: (round(v, 3) if not isinstance(v, (str, np.ndarray)) else v) for k, v in s.items() if k != 'hit'}, flush=True)
    slot_test = paired(slots[1]['hit'], slots[0]['hit'])
    # C17 cold-start users: first basket after cut, no training history
    after = B[B.d >= MAIN_CUT].groupby('ui').head(1); cold = after[~after.ui.isin(set(tr.ui))]
    Rc = {k: evaluate3(S[k], cold, X) for k in ['Popularity', 'Hour-Popularity']}
    cold_res = dict(n=len(cold), n_warm=len(te), **{k: (v['ndcg'].mean(), v['hr5'].mean()) for k, v in Rc.items()})
    print('cold', cold_res, flush=True)
    cpu = [l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')]
    out['main'] = dict(T=T, R=R, P=P, G=G, slots=[{k: v for k, v in s.items() if k != 'hit'} for s in slots], slot_test=slot_test,
                       cold=cold_res, time=tm, cpu=(cpu[0] if cpu else platform.processor()), ncpu=os.cpu_count(), n=len(te))
    pickle.dump(out, open('exp3_main.pkl', 'wb'))
    # C14: 5 simulator seeds at w=0.6 (+ main seed), hyperparameters fixed from main tuning
    AS = pd.read_csv(os.path.join(exp2.DATA, 'simulated_customer_assignments.csv'), dtype=str, encoding='utf-8-sig')
    o0 = o.drop(columns='customer_id'); seeds = []
    for s, a in AS[(AS.set == 'sensitivity') & (AS.item_w == '0.6')].groupby('seed'):
        oo = o0.merge(a[['order_id', 'customer_id']], on='order_id')
        B2, _, c2 = setup(oo); tr2, te2 = temporal(B2, MAIN_CUT)
        S2, m2, _ = models(c2, tr2, te2, T)
        row = dict(seed=int(s), n=len(te2), **{k: evaluate(v, te2, m2['X'])['ndcg'].mean() for k, v in S2.items()})
        seeds.append(row); print(row, flush=True)
    out['seeds'] = pd.DataFrame(seeds)
    pickle.dump(out, open('exp3_main.pkl', 'wb'))
