import os, sys, glob, pickle, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2 import *

T = pickle.load(open('v2_results.pkl', 'rb'))['temporal']['T']
o0 = load_orders().drop(columns='customer_id')
prod = pd.read_csv(os.path.join(DATA, 'products.csv'), encoding='utf-8-sig')
rows = []
AS = pd.read_csv(os.path.join(DATA, 'simulated_customer_assignments.csv'), dtype=str, encoding='utf-8-sig')
AS = AS[AS.set == 'sensitivity']
for (w, s), a in AS.groupby(['item_w', 'seed'], sort=True):
    o = o0.merge(a[['order_id', 'customer_id']], on='order_id')
    B, items, U = build(o); c = Ctx(B, len(items), U)
    tr, te = temporal(B, MAIN_CUT); m = mats(c, tr); X = m['X']
    S = {'Popularity': pop(c, m, te), 'Hour-Popularity': hourpop(c, m, te), 'Personal': personal(c, m, te),
         'UserKNN': userknn(c, m, te, T['uknn_k']), 'ItemKNN+R': knn_r(c, m, te, **T['knn_r']),
         'Context Hybrid': ctx_hybrid(c, m, te, **T['ctx']), 'RA-Hybrid': ra_hybrid(c, m, te, **T['ra'])}
    R = {k: evaluate(v, te, X) for k, v in S.items()}
    rep_share = np.mean([np.isin(its, np.where(X[u] > 0)[0]).mean() for u, its in zip(te.ui.values, te['items'].values)])
    row = dict(w=float(w), seed=int(s), n=len(te), rep_share=rep_share)
    for k, v in R.items(): row[k] = v['ndcg'].mean()
    for a_, b_ in [('Personal', 'Popularity'), ('Context Hybrid', 'Personal'), ('RA-Hybrid', 'Personal'), ('Context Hybrid', 'Popularity')]:
        d = R[a_]['ndcg'] - R[b_]['ndcg']; row[f'{a_}-{b_}'] = d.mean(); row[f'sd:{a_}-{b_}'] = d.std(ddof=1)
        row[f'p:{a_}-{b_}'] = stats.wilcoxon(d, zero_method='pratt').pvalue
    rows.append(row); print(row, flush=True)
pd.DataFrame(rows).to_pickle('sens_results.pkl')
