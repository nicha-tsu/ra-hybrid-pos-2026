"""Experiment v2: global temporal split (main), leave-last-basket-out (secondary),
RA-Hybrid, repeat-aware baselines, repeat/explore recall, HR@K with CI."""
import os
import pandas as pd, numpy as np
DATA = os.environ.get('DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
from scipy import stats

MAIN_CUT, VAL_CUT = '2023-08-01', '2023-05-01'

def load_orders():
    # reads the released data; ASSIGN env var = 'main' or 'w<item_w>_s<seed>' selects a simulated customer-ID set
    o = pd.read_csv(os.path.join(DATA, 'orders.csv'), dtype=str, encoding='utf-8-sig').sort_values('row_seq', key=lambda s: s.astype(int))
    o = o.sort_values(['order_date', 'order_time', 'order_id'], kind='stable').reset_index(drop=True)
    for col in ['qty', 'unit_price', 'line_no']: o[col] = o[col].astype(int)
    o['hour'] = o.order_time.str[:2].astype(int)
    return o

def build(o):
    items = sorted(o.sku.unique()); ix = {s: i for i, s in enumerate(items)}
    o = o.assign(ii=o.sku.map(ix))
    B = o.groupby('order_id').agg(u=('customer_id', 'first'), d=('order_date', 'first'),
                                  t=('order_time', 'first'), h=('hour', 'first')).reset_index()
    it = o.groupby('order_id').ii.apply(lambda x: np.array(sorted(set(x))))
    B['items'] = B.order_id.map(it)
    B = B.sort_values(['u', 'd', 't', 'order_id']).reset_index(drop=True)
    users = sorted(B.u.unique()); B['ui'] = B.u.map({u: i for i, u in enumerate(users)})
    B['k'] = B.groupby('u').cumcount(); B['n'] = B.groupby('u').u.transform('size')
    return B, items, len(users)

class Ctx:
    def __init__(self, B, I, U): self.B, self.I, self.U = B, I, U

def temporal(B, cut, lo=None):
    tr = B[B.d < cut] if lo is None else B[B.d < lo]
    after = B[B.d >= (cut if lo is None else lo)]
    if lo is not None: after = after[after.d < cut]
    te = after.groupby('ui').head(1)
    te = te[te.ui.isin(set(tr.ui))]
    return tr, te

def leave_last(B, val=False):
    if not val:
        te = B[(B.n >= 2) & (B.k == B.n - 1)]; tr = B.drop(te.index)
    else:
        te = B[(B.n >= 3) & (B.k == B.n - 2)]; tr = B[B.k < B.n - 2]
    return tr, te

def mats(c, tr):
    U, I = c.U, c.I
    X = np.zeros((U, I)); H = np.zeros((24, I)); last = np.zeros((U, I))
    for ui, its in zip(tr.ui.values, tr['items'].values): X[ui, its] += 1
    for h, its in zip(tr.h.values, tr['items'].values): H[h, its] += 1
    r = tr.groupby('ui').cumcount(ascending=False).values
    for ui, its, rr in zip(tr.ui.values, tr['items'].values, r): last[ui, its] += 0.8 ** rr
    # repeat propensity: share of basket items already bought earlier (within training)
    rep = np.zeros(U); cnt = np.zeros(U); seen = {}
    for ui, its in zip(tr.ui.values, tr['items'].values):
        s = seen.setdefault(ui, set())
        if s:
            rep[ui] += np.mean([i in s for i in its]); cnt[ui] += 1
        s.update(its.tolist())
    g = rep.sum() / max(cnt.sum(), 1)
    prop = (rep + 5 * g) / (cnt + 5)
    return dict(X=X, H=H, last=last, prop=prop)

def nrm(M):
    m = M.max(1, keepdims=True); return np.divide(M, m, out=np.zeros_like(M), where=m > 0)

# ---------------- models
def pop(c, m, te, **k): return np.tile(m['X'].sum(0), (c.U, 1))
def hourpop(c, m, te, **k):
    S = np.tile(m['X'].sum(0) * 1e-9, (c.U, 1)); S[te.ui.values] += nrm(m['H'])[te.h.values]; return S
def personal(c, m, te, **k):
    X = m['X']; return X + 0.5 * m['last'] + 1e-6 * X.sum(0)[None, :] / max(X.sum(), 1)
def knn_sim(X, k, self_sim=False):
    Xb = (X > 0).astype(float); C = Xb.T @ Xb; n = np.sqrt(np.diag(C)); S = C / np.outer(n, n).clip(1e-9)
    if not self_sim: np.fill_diagonal(S, 0)
    if k:
        thr = -np.sort(-S, 1)[:, k - 1:k]; S = np.where(S >= thr, S, 0)
    return S
def itemknn(c, m, te, k=30, **kw): return (m['X'] > 0).astype(float) @ knn_sim(m['X'], k)
def userknn(c, m, te, k=50, **kw):
    Xb = (m['X'] > 0).astype(float); n = np.linalg.norm(Xb, axis=1).clip(1e-9)
    S = (Xb @ Xb.T) / np.outer(n, n); np.fill_diagonal(S, 0)
    idx = np.argpartition(-S, k, 1)[:, :k]; M = np.zeros_like(S)
    np.put_along_axis(M, idx, np.take_along_axis(S, idx, 1), 1); return M @ Xb
def ease_B(X, lam):
    Xb = (X > 0).astype(float); G = Xb.T @ Xb + lam * np.eye(X.shape[1]); P = np.linalg.inv(G)
    Bm = -P / np.diag(P); np.fill_diagonal(Bm, 0); return Bm
def ease(c, m, te, lam=200, **kw): return (m['X'] > 0).astype(float) @ ease_B(m['X'], lam)
def ials(c, m, te, f=32, reg=0.1, alpha=10, it=12, **kw):
    X = m['X']; U, I = X.shape; r = np.random.default_rng(7); C = 1 + alpha * np.log1p(X); Pm = (X > 0).astype(float)
    Uf = r.normal(0, .1, (U, f)); V = r.normal(0, .1, (I, f))
    for _ in range(it):
        for A, Bf, Cm, Pp in ((Uf, V, C, Pm), (V, Uf, C.T, Pm.T)):
            BtB = Bf.T @ Bf
            for a in range(A.shape[0]):
                cc = Cm[a]; nz = np.where(cc > 1)[0]
                if len(nz) == 0: A[a] = 0; continue
                Bn = Bf[nz]; M = BtB + (Bn.T * (cc[nz] - 1)) @ Bn + reg * np.eye(f)
                A[a] = np.linalg.solve(M, (Bn.T * cc[nz]) @ Pp[a, nz])
    return Uf @ V.T
def content(c, m, te, cidx=None, **kw):
    X = m['X']; K = cidx.max() + 1; Cu = np.zeros((c.U, K))
    for j in range(K): Cu[:, j] = X[:, cidx == j].sum(1)
    Cu = Cu / Cu.sum(1, keepdims=True).clip(1e-9); ip = X.sum(0); w = np.zeros(c.I)
    for j in range(K):
        mm = cidx == j; w[mm] = ip[mm] / max(ip[mm].sum(), 1e-9)
    return Cu[:, cidx] * w[None, :]
def knn_r(c, m, te, k=30, w=0.5, **kw):   # repeat-aware ItemKNN: personal + kNN (self-similarity kept)
    return nrm(personal(c, m, te)) + w * nrm((m['X'] > 0).astype(float) @ knn_sim(m['X'], k, self_sim=True))
def ease_r(c, m, te, lam=200, w=0.5, **kw):  # repeat-aware EASE: personal + EASE
    return nrm(personal(c, m, te)) + w * nrm(ease(c, m, te, lam))
def ctx_hybrid(c, m, te, g=0.6, **kw):
    return nrm(personal(c, m, te)) + g * hourpop(c, m, te)
def ra_hybrid(c, m, te, g=0.6, e=1.0, base='ease', lam=200, k=30, **kw):
    """RA-Hybrid: s = p_u * R(u,i) + (1-p_u) * e * E(u,i) + g * T(i|h)
    R = repeat score (frequency + recency) on items in the user's history,
    E = explore score (EASE or ItemKNN) on items NOT in the history,
    p_u = smoothed repeat propensity of user u, T = hour-of-day popularity."""
    hist = m['X'] > 0
    R = nrm(np.where(hist, personal(c, m, te), 0))
    raw = ease(c, m, te, lam) if base == 'ease' else itemknn(c, m, te, k)
    E = nrm(np.where(hist, 0, np.maximum(raw, 0)))
    p = m['prop'][:, None]
    return p * R + (1 - p) * e * E + g * hourpop(c, m, te)

# ---------------- metrics
DISC = 1 / np.log2(np.arange(2, 12))
def evaluate(S, te, X):
    S = S[te.ui.values]; order = np.argsort(-S, 1)[:, :10]
    out = {k: [] for k in ['ndcg', 'hr3', 'hr5', 'hr10', 'rec', 'rec_rep', 'rec_exp']}
    for ui, top, its in zip(te.ui.values, order, te['items'].values):
        hit = np.isin(top, its); idcg = DISC[:min(len(its), 10)].sum()
        out['ndcg'].append((hit * DISC).sum() / idcg)
        out['hr3'].append(float(hit[:3].any())); out['hr5'].append(float(hit[:5].any())); out['hr10'].append(float(hit.any()))
        out['rec'].append(hit.sum() / len(its))
        h = X[ui] > 0; rep = its[h[its]]; exp_ = its[~h[its]]
        out['rec_rep'].append(np.isin(rep, top).mean() if len(rep) else np.nan)
        out['rec_exp'].append(np.isin(exp_, top).mean() if len(exp_) else np.nan)
    return {k: np.array(v) for k, v in out.items()}

def boot_ci(x, B=4000, seed=1):
    r = np.random.default_rng(seed); x = x[~np.isnan(x)]
    idx = r.integers(0, len(x), (B, len(x))); m = x[idx].mean(1); return np.percentile(m, [2.5, 97.5])

def paired(a, b):
    d = a - b
    return dict(mean=d.mean(), sd=d.std(ddof=1), ci=boot_ci(d), t_p=stats.ttest_rel(a, b).pvalue,
                w_p=stats.wilcoxon(d, zero_method='wilcox').pvalue if (d != 0).any() else 1.0,
                w_p_pratt=stats.wilcoxon(d, zero_method='pratt').pvalue if (d != 0).any() else 1.0,
                win=(d > 0).mean(), tie=(d == 0).mean(), loss=(d < 0).mean(), n=len(d))

def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj
