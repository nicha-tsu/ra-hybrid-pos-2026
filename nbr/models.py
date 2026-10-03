"""Next-basket models. Every scorer returns a (len(q), I) matrix of scores for query users q."""
import itertools
import numpy as np


def nrm(M):
    m = M.max(1, keepdims=True)
    return np.divide(M, m, out=np.zeros_like(M), where=m > 0)


class State:
    """Sufficient statistics of the training baskets."""
    def __init__(self, tr, U, I):
        self.U, self.I = U, I
        X = np.zeros((U, I)); H = np.zeros((24, I))
        for ui, its, h in zip(tr.ui.values, tr['items'].values, tr.h.values):
            X[ui, its] += 1
            if h >= 0: H[h, its] += 1
        self.X, self.H = X, H; self.has_hour = bool(H.sum() > 0)
        self.tr = tr; self.rank = tr.groupby('ui').cumcount(ascending=False).values
        self.seqs = {}
        for ui, its in zip(tr.ui.values, tr['items'].values): self.seqs.setdefault(ui, []).append(its)
        rep = np.zeros(U); cnt = np.zeros(U); seen = {}
        for ui, its in zip(tr.ui.values, tr['items'].values):      # repeat propensity p_u
            s = seen.setdefault(ui, set())
            if s: rep[ui] += np.mean([i in s for i in its]); cnt[ui] += 1
            s.update(its.tolist())
        g = rep.sum() / max(cnt.sum(), 1)
        self.prop = (rep + 5 * g) / (cnt + 5); self.pbar = g
        self._m = {}

    def memo(self, key, fn):
        if key not in self._m: self._m[key] = fn()
        return self._m[key]

    def last(self, rho):
        def f():
            L = np.zeros((self.U, self.I))
            for ui, its, r in zip(self.tr.ui.values, self.tr['items'].values, self.rank): L[ui, its] += rho ** r
            return L
        return self.memo(('last', rho), f)


def _hour_t(s, hrs, n):
    if not s.has_hour or hrs is None: return np.zeros((n, s.I))
    hrs = np.asarray(hrs); T = nrm(s.H)[np.clip(hrs, 0, 23)]; T[hrs < 0] = 0; return T


# ---------------- scorers
def popularity(s, q, hrs=None, **p):
    return np.tile(s.X.sum(0), (len(q), 1))

def personal(s, q, hrs=None, beta=0.5, rho=0.8, **p):
    X = s.X[q]; return X + beta * s.last(rho)[q] + 1e-6 * s.X.sum(0)[None, :] / max(s.X.sum(), 1)

def _tifu_vectors(s, gsize, rb, rg):
    V = np.zeros((s.U, s.I))
    for u, bl in s.seqs.items():
        groups = [bl[i:i + gsize] for i in range(0, len(bl), gsize)]; G = len(groups); acc = np.zeros(s.I)
        for gi, g in enumerate(groups):
            gv = np.zeros(s.I); L = len(g)
            for j, its in enumerate(g): gv[its] += rb ** (L - 1 - j)
            acc += rg ** (G - 1 - gi) * gv / L
        V[u] = acc / G
    return V

def tifu_knn(s, q, hrs=None, gsize=3, rb=0.9, rg=0.9, k=100, alpha=0.7, **p):
    """TIFU-KNN (Hu et al., SIGIR 2020)."""
    V = s.memo(('tifu', gsize, rb, rg), lambda: _tifu_vectors(s, gsize, rb, rg))
    k = min(k, s.U - 2)
    if k < 1: return V[q]
    Vn = V / np.linalg.norm(V, axis=1).clip(1e-12)[:, None]
    sim = Vn[q] @ Vn.T; sim[np.arange(len(q)), q] = -1
    idx = np.argpartition(-sim, k, 1)[:, :k]
    W = np.zeros((len(q), s.U)); np.put_along_axis(W, idx, 1.0 / k, 1)
    return alpha * V[q] + (1 - alpha) * (W @ V)

def _upcf_vectors(s, r):
    P = np.zeros((s.U, s.I))
    for u, bl in s.seqs.items():
        bl = bl if r is None else bl[-r:]
        for its in bl: P[u, its] += 1
        P[u] /= len(bl)
    return P

def upcf(s, q, hrs=None, r=5, a=0.75, qq=5, lam=0.5, **p):
    """UP-CF@r (Faggioli et al., UMAP 2020)."""
    P = s.memo(('upcf', r), lambda: _upcf_vectors(s, r)); n = np.linalg.norm(P, axis=1).clip(1e-12)
    sim = (P[q] @ P.T) / (n[q, None] ** (2 * a) * n[None, :] ** (2 * (1 - a)))
    sim[np.arange(len(q)), q] = 0; sim = np.maximum(sim, 0) ** qq
    cf = sim @ P; cf = cf / cf.max(1, keepdims=True).clip(1e-12)
    return P[q] + lam * cf

def _ease_B(s, lam):
    def f():
        Xb = (s.X > 0).astype(float); P = np.linalg.inv(Xb.T @ Xb + lam * np.eye(s.I))
        Bm = -P / np.diag(P); np.fill_diagonal(Bm, 0); return Bm
    return s.memo(('ease', lam), f)

def _knn_S(s, k):
    def f():
        Xb = (s.X > 0).astype(float); C = Xb.T @ Xb; n = np.sqrt(np.diag(C)); S = C / np.outer(n, n).clip(1e-9)
        np.fill_diagonal(S, 0)
        if k and k < s.I:
            thr = -np.sort(-S, 1)[:, k - 1:k]; S = np.where(S >= thr, S, 0)
        return S
    return s.memo(('knnS', k), f)

def ra_hybrid(s, q, hrs=None, base='ease', lam=200, k=30, e=1.0, g=0.3, pers=None, **p):
    """RA-Hybrid: s = p_u*R + (1-p_u)*e*E + g*T(hour).
    R: repeat score on the user's history; E: ease/knn score on items NOT in history; T: hour-of-day popularity."""
    hist = s.X[q] > 0
    R = nrm(np.where(hist, personal(s, q, **(pers or {})), 0))
    Xb = (s.X[q] > 0).astype(float)
    raw = Xb @ (_ease_B(s, lam) if base == 'ease' else _knn_S(s, k))
    E = nrm(np.where(hist, 0, np.maximum(raw, 0)))
    pu = s.prop[q][:, None]
    return pu * R + (1 - pu) * e * E + g * _hour_t(s, hrs, len(q))


# ---------------- tuning grids (kept small: SME data, laptop CPU)
def grids(has_hour):
    G = {
        'Popularity': [{}],
        'Personal': [dict(beta=b, rho=r) for r in (0.5, 0.8, 1.0) for b in (0, 0.5, 1, 2, 4)],
        'TIFU-KNN': [dict(gsize=gs, rb=rb, rg=rg, k=k, alpha=a) for gs, rb, rg, k, a in
                     itertools.product((3, 7), (0.9, 1.0), (0.6, 0.9), (30, 100, 300), (0.5, 0.7, 0.9))],
        'UP-CF@r': [dict(r=r, a=a, qq=qq, lam=l) for r, a, qq, l in
                    itertools.product((1, 3, 5, None), (0.5, 0.75), (1, 5), (0.1, 0.5, 1.0))],
    }
    gs = (0, 0.15, 0.3, 0.6, 1) if has_hour else (0,)
    G['RA-Hybrid'] = [dict(base='ease', lam=l, e=e, g=g) for l in (10, 200) for e in (0, 0.1, 0.25, 0.5, 1, 2) for g in gs] + \
                     [dict(base='knn', k=k, e=e, g=g) for k in (10, 30) for e in (0, 0.1, 0.25, 0.5, 1, 2) for g in gs]
    return G

SCORERS = {'Popularity': popularity, 'Personal': personal, 'TIFU-KNN': tifu_knn, 'UP-CF@r': upcf, 'RA-Hybrid': ra_hybrid}
