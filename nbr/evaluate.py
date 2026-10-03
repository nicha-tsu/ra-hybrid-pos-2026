"""Splits, metrics, validation-based method selection, paired statistics, sample-size formula."""
import numpy as np, pandas as pd
from scipy import stats
from .models import State, SCORERS, grids

DISC = 1 / np.log2(np.arange(2, 12))


# ---------------- splits
def split_leave_last(B):
    """Hold out each customer's last basket (customers with >= 2 baskets)."""
    te = B[(B.n >= 2) & (B.k == B.n - 1)]
    return B.drop(te.index), te


def split_temporal(B, cut, lo=None):
    """Train on baskets before `cut`; test = first basket on/after cut per customer with history."""
    lo = pd.Timestamp(lo) if lo is not None else None
    tr = B[B.d < (lo if lo is not None else pd.Timestamp(cut))]
    after = B[B.d >= (lo if lo is not None else pd.Timestamp(cut))]
    if lo is not None: after = after[after.d < pd.Timestamp(cut)]
    te = after.groupby('ui').head(1)
    return tr, te[te.ui.isin(set(tr.ui))]


def make_splits(B, cut=None, val_cut=None):
    """Returns (train, test) and (val_train, val_test) built from `train` only (no test leakage)."""
    if cut is None:
        tr, te = split_leave_last(B)
        tr = tr.assign(k=tr.groupby('ui').cumcount(), n=tr.groupby('ui').ui.transform('size'))
        vtr, vte = split_leave_last(tr)
    else:
        cut = pd.Timestamp(cut)
        if val_cut is None:
            val_cut = cut - (cut - B.d.min()) * 0.25
        tr, te = split_temporal(B, cut)
        val_cut = pd.Timestamp(val_cut)
        vtr, vte = split_temporal(B, cut=cut, lo=val_cut)      # train < val_cut, test in [val_cut, cut)
    return (tr, te), (vtr, vte)


# ---------------- metrics
def evaluate(S, te, X):
    """S: scores aligned with te rows; X: training count matrix (to split repeat vs explore)."""
    order = np.argsort(-S, 1)[:, :10]
    out = {k: [] for k in ('ndcg', 'ndcg5', 'mrr', 'hr3', 'hr5', 'hr10', 'rec_rep', 'rec_exp')}
    for ui, top, its in zip(te.ui.values, order, te['items'].values):
        hit = np.isin(top, its); idcg = DISC[:min(len(its), 10)].sum()
        out['ndcg'].append((hit * DISC).sum() / idcg)
        out['ndcg5'].append((hit[:5] * DISC[:5]).sum() / DISC[:min(len(its), 5)].sum())
        f = np.where(hit)[0]; out['mrr'].append(1 / (f[0] + 1) if len(f) else 0.0)
        out['hr3'].append(float(hit[:3].any())); out['hr5'].append(float(hit[:5].any())); out['hr10'].append(float(hit.any()))
        h = X[ui] > 0; rep = its[h[its]]; exp_ = its[~h[its]]
        out['rec_rep'].append(np.isin(rep, top).mean() if len(rep) else np.nan)
        out['rec_exp'].append(np.isin(exp_, top).mean() if len(exp_) else np.nan)
    return {k: np.array(v) for k, v in out.items()}


def boot_ci(x, B=4000, seed=1):
    x = x[~np.isnan(x)]
    if len(x) < 2: return (np.nan, np.nan)
    idx = np.random.default_rng(seed).integers(0, len(x), (B, len(x)))
    return tuple(np.percentile(x[idx].mean(1), [2.5, 97.5]))


def paired(a, b):
    d = a - b
    nz = (d != 0).any()
    return dict(mean=d.mean(), sd=d.std(ddof=1) if len(d) > 1 else np.nan, ci=boot_ci(d),
                t_p=stats.ttest_rel(a, b).pvalue if nz and len(d) > 1 else 1.0,
                w_p=stats.wilcoxon(d, zero_method='pratt').pvalue if nz else 1.0,
                win=(d > 0).mean(), tie=(d == 0).mean(), loss=(d < 0).mean(), n=len(d))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj


def sample_size(sd, delta, z=2.8):
    """Customers needed to detect a paired NDCG@10 difference `delta` (alpha .05 two-sided, power .80)."""
    return int(np.ceil((z * sd / delta) ** 2))


def mde(sd, n, z=2.8):
    return z * sd / np.sqrt(n)


# ---------------- tuning + selection
def _score(name, s, te, p):
    return SCORERS[name](s, te.ui.values, te.h.values, **p)


def tune(c, vtr, vte, names=None, log=print):
    """Pick hyper-parameters of every method on validation only. Returns {method: params}."""
    s = State(vtr, c.U, c.I); X = s.X; names = names or list(SCORERS)
    G = grids(s.has_hour); best = {}
    for nm in ['Popularity', 'Personal'] + [n for n in names if n not in ('Popularity', 'Personal')]:
        if nm not in names: continue
        cands = G[nm]
        if nm == 'RA-Hybrid': cands = [dict(p, pers=best['Personal']) for p in cands]
        res = [(evaluate(_score(nm, s, vte, p), vte, X)['ndcg'].mean(), -i) for i, p in enumerate(cands)]
        sc, i = max(res); i = -i; best[nm] = cands[i]       # ties -> first (simplest) grid point
        log(f'  tune {nm:10s} grid={len(cands):3d} val NDCG@10={sc:.4f}')
    return best


def compare(c, tr, te, vtr, vte, params, names=None):
    """Fit every method on tr, test on te; also validation scores for the selection rule."""
    names = names or list(params)
    s, sv = State(tr, c.U, c.I), State(vtr, c.U, c.I)
    R, V = {}, {}
    for nm in names:
        R[nm] = evaluate(_score(nm, s, te, params[nm]), te, s.X)
        V[nm] = evaluate(_score(nm, sv, vte, params[nm]), vte, sv.X)
    return R, V


def select_method(V, base='Personal', delta=0.01):
    """Selection rule (paper, Sec. 3.9/4.2.2): switch from the tuned Personal baseline to a more complex
    method only if its validation gain is BOTH statistically significant (95% paired bootstrap CI lower
    bound > 0) AND practically meaningful (mean gain >= delta NDCG@10); among qualifying methods take the
    one with the highest validation NDCG@10. Otherwise keep Personal (simplest)."""
    best, best_sc = base, V[base]['ndcg'].mean()
    for nm, v in V.items():
        if nm == base or nm == 'Popularity': continue
        d = v['ndcg'] - V[base]['ndcg']; lo, _ = boot_ci(d)
        if lo > 0 and d.mean() >= delta and v['ndcg'].mean() > best_sc: best, best_sc = nm, v['ndcg'].mean()
    return best


def run_experiment(c, cut=None, val_cut=None, names=None, log=print):
    (tr, te), (vtr, vte) = make_splits(c.B, cut, val_cut)
    log(f'train baskets {len(tr)} | test customers {len(te)} | validation customers {len(vte)}')
    if len(te) < 10 or len(vte) < 10:
        raise ValueError('ลูกค้าที่ซื้อซ้ำน้อยเกินไปสำหรับประเมินผล (ต้องมีลูกค้าซื้อ >=3 ครั้งอย่างน้อย 10 ราย)')
    params = tune(c, vtr, vte, names, log)
    R, V = compare(c, tr, te, vtr, vte, params, names or list(params))
    chosen = select_method(V)
    rows = []
    for nm, r in R.items():
        row = dict(method=nm, ndcg10=r['ndcg'].mean(), ndcg5=r['ndcg5'].mean(), mrr10=r['mrr'].mean(), ci_lo=boot_ci(r['ndcg'])[0], ci_hi=boot_ci(r['ndcg'])[1],
                   hr3=r['hr3'].mean(), hr5=r['hr5'].mean(), hr10=r['hr10'].mean(),
                   recall_repeat=np.nanmean(r['rec_rep']), recall_explore=np.nanmean(r['rec_exp']),
                   val_ndcg10=V[nm]['ndcg'].mean())
        if nm not in ('Personal', 'Popularity'):
            dv = V[nm]['ndcg'] - V['Personal']['ndcg']; lo_v, hi_v = boot_ci(dv)
            row.update(val_gain_vs_personal=dv.mean(), val_gain_ci_lo=lo_v, val_gain_ci_hi=hi_v)
        if nm != 'Personal':
            p = paired(r['ndcg'], R['Personal']['ndcg'])
            row.update(diff_vs_personal=p['mean'], diff_ci_lo=p['ci'][0], diff_ci_hi=p['ci'][1],
                       wilcoxon_p=p['w_p'], win=p['win'], loss=p['loss'])
        rows.append(row)
    T = pd.DataFrame(rows)
    mask = T.method != 'Personal'; T.loc[mask, 'wilcoxon_p_holm'] = holm(T.loc[mask, 'wilcoxon_p'])
    sd = float(np.median([paired(R[n]['ndcg'], R['Personal']['ndcg'])['sd'] for n in R if n not in ('Personal',)
                          and not np.isnan(paired(R[n]['ndcg'], R['Personal']['ndcg'])['sd'])] or [0.3]))
    n = len(te)
    size = dict(test_customers=n, sd_paired_ndcg10=sd, mde_current=mde(sd, n),
                customers_needed_delta_0_05=sample_size(sd, 0.05), customers_needed_delta_0_03=sample_size(sd, 0.03),
                customers_needed_delta_0_01=sample_size(sd, 0.01))
    return dict(table=T, params=params, chosen=chosen, sample_size=size, per_user=R)
