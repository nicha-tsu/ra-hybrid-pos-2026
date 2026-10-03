import os, sys, time, pickle, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2 import *

o = load_orders()
B, items, U = build(o)
I = len(items); c = Ctx(B, I, U)
prod = pd.read_csv(os.path.join(DATA, 'products.csv'), encoding='utf-8-sig').set_index('sku').reindex(items)
cats = sorted(prod.category.unique()); cidx = np.array([cats.index(x) for x in prod.category])

def nd(S, te, X): return evaluate(S, te, X)['ndcg'].mean()

def tune(trv, tev):
    m = mats(c, trv); X = m['X']; T = {}
    T['knn_k'] = max([3, 5, 10, 30, 100, 0], key=lambda k: nd(itemknn(c, m, tev, k), tev, X))
    T['uknn_k'] = max([100, 200, 400, 800], key=lambda k: nd(userknn(c, m, tev, k), tev, X))
    T['lam'] = max([0.1, 0.3, 1, 3, 10, 50, 200, 1000], key=lambda l: nd(ease(c, m, tev, l), tev, X))
    T['als'] = max([dict(f=16, reg=0.1, alpha=10), dict(f=32, reg=0.1, alpha=10), dict(f=32, reg=1, alpha=20)],
                   key=lambda p: nd(ials(c, m, tev, **p), tev, X))
    T['knn_r'] = max([dict(k=k, w=w) for k in [5, 10, 30] for w in [0.1, 0.3, 0.6, 1.0, 1.5, 2.5]],
                     key=lambda p: nd(knn_r(c, m, tev, **p), tev, X))
    T['ease_r'] = max([dict(lam=l, w=w) for l in [10, 200, 1000, 3000, 10000] for w in [0.1, 0.3, 0.6, 1.0, 1.5]],
                      key=lambda p: nd(ease_r(c, m, tev, **p), tev, X))
    T['ctx'] = max([dict(g=g) for g in [0, 0.15, 0.3, 0.6, 1, 1.5, 2.5]], key=lambda p: nd(ctx_hybrid(c, m, tev, **p), tev, X))
    grid = [dict(base=b, e=e, g=g, lam=T['lam'], k=T['knn_k']) for b in ['ease', 'knn'] for e in [0, 0.1, 0.25, 0.5, 1, 2] for g in [0, 0.15, 0.3, 0.6, 1]]
    T['ra'] = max(grid, key=lambda p: nd(ra_hybrid(c, m, tev, **p), tev, X))
    return T

def run_all(tr, te, T):
    m = mats(c, tr); X = m['X']
    S = {'Popularity': pop(c, m, te), 'Hour-Popularity': hourpop(c, m, te), 'Personal': personal(c, m, te),
         'ItemKNN': itemknn(c, m, te, T['knn_k']), 'ItemKNN+R': knn_r(c, m, te, **T['knn_r']),
         'UserKNN': userknn(c, m, te, T['uknn_k']), 'EASE': ease(c, m, te, T['lam']),
         'EASE+R': ease_r(c, m, te, **T['ease_r']), 'iALS': ials(c, m, te, **T['als']),
         'Content/Ontology': content(c, m, te, cidx=cidx), 'Context Hybrid': ctx_hybrid(c, m, te, **T['ctx']),
         'RA-Hybrid': ra_hybrid(c, m, te, **T['ra'])}
    return {k: evaluate(v, te, X) for k, v in S.items()}, X

out = {}
for proto in ['temporal', 'leave_last']:
    t0 = time.time()
    if proto == 'temporal':
        trv, tev = temporal(B, MAIN_CUT, lo=VAL_CUT); tr, te = temporal(B, MAIN_CUT)
    else:
        trv, tev = leave_last(B, val=True); tr, te = leave_last(B)
    T = tune(trv, tev)
    R, X = run_all(tr, te, T)
    out[proto] = dict(T=T, R=R, te=te[['u', 'ui', 'd', 'n', 'k']].reset_index(drop=True),
                      ntrain=len(tr), nval=len(tev), ntest=len(te))
    print(proto, 'val users', len(tev), 'test users', len(te), 'train baskets', len(tr), 'T', T, '%.0fs' % (time.time() - t0), flush=True)
    for k, v in R.items():
        print(f"  {k:18s} NDCG {v['ndcg'].mean():.4f} HR3 {v['hr3'].mean():.3f} HR5 {v['hr5'].mean():.3f} HR10 {v['hr10'].mean():.3f} "
              f"Rec {v['rec'].mean():.3f} RecRep {np.nanmean(v['rec_rep']):.3f} RecExp {np.nanmean(v['rec_exp']):.3f}", flush=True)
pickle.dump(out, open('v2_results.pkl', 'wb'))
