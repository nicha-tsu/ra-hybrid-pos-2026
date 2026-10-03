"""Round-4 additions requested in the round-3 review (all customer IDs remain SIMULATED):
  tost  : equivalence test (TOST, delta = 0.01 NDCG@10) on the main temporal split, from run3.pkl
  abl   : RA-Hybrid ablations (p_u constant, gamma = 0) on the 20 simulator-sensitivity sets
  alt   : alternative simulator (category-level preferences, simulate_alt.py), 3 weights x 5 seeds
  perm  : permutation control - customer IDs shuffled among baskets of the same month (main set), 20 permutations
  seedtune : per-seed re-tuning on validation for the 5 sensitivity seeds at item_w = 0.6
Usage: python run4.py <mode>   ->  run4_<mode>.pkl"""
import os, sys, pickle, glob, re
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp2
from exp2 import *
from run3 import tune_on, scores
HERE = os.path.dirname(os.path.abspath(__file__))
R3 = pickle.load(open(os.environ.get('RUN3', os.path.join(HERE, 'run3.pkl')), 'rb'))
T = R3['temporal']['T']; DELTA = 0.01
KEY = ['Popularity', 'Hour-Popularity', 'Personal', 'UP-CF@r', 'TIFU-KNN', 'RA-Hybrid', 'RA: p_u คงที่', 'RA: γ = 0']

def tost(d, delta=DELTA):
    d = np.asarray(d, float); n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    p_lo = 1 - stats.t.cdf((m + delta) / se, n - 1); p_hi = stats.t.cdf((m - delta) / se, n - 1)
    lo, hi = m + np.array(stats.t.interval(0.90, n - 1)) * se
    return dict(mean=m, ci90=(lo, hi), p_tost=max(p_lo, p_hi), equivalent=bool(max(p_lo, p_hi) < 0.05))

def setup_base():
    o = load_orders(); B, items, U = build(o)
    prod = pd.read_csv(os.path.join(exp2.DATA, 'products.csv'), encoding='utf-8-sig').set_index('sku').reindex(items)
    cats = sorted(prod.category.unique()); cidx = np.array([cats.index(x) for x in prod.category])
    return o, cidx

def eval_set(o, cidx, only, TT=T):
    B, it, U = build(o); c = Ctx(B, len(it), U); tr, te = temporal(B, MAIN_CUT)
    S, m, _ = scores(c, tr, te, TT, cidx, only=only)
    return {k: evaluate(v, te, m['X'])['ndcg'] for k, v in S.items()}, len(te), m, te

def summarise(R, extra):
    row = dict(extra, **{k: v.mean() for k, v in R.items()})
    for a, b in [('Personal', 'Popularity'), ('RA-Hybrid', 'Personal'), ('UP-CF@r', 'Personal'), ('TIFU-KNN', 'Personal'),
                 ('RA-Hybrid', 'RA: p_u คงที่'), ('RA-Hybrid', 'RA: γ = 0')]:
        if a in R and b in R:
            d = R[a] - R[b]; lo, hi = boot_ci(d, B=2000)
            row[f'{a}-{b}'] = d.mean(); row[f'lo:{a}-{b}'] = lo; row[f'hi:{a}-{b}'] = hi; row[f'sd:{a}-{b}'] = d.std(ddof=1)
    return row

if __name__ == '__main__':
    mode = sys.argv[1]; out = {}
    if mode == 'tost':
        Rm = R3['temporal']['R']
        out = {f'{a}-{b}': tost(Rm[a]['ndcg'] - Rm[b]['ndcg']) for a, b in
               [('RA-Hybrid', 'Personal'), ('UP-CF@r', 'Personal'), ('TIFU-KNN', 'Personal'), ('EASE+R', 'Personal'), ('ItemKNN+R', 'Personal'),
                ('RA-Hybrid', 'RA: p_u คงที่'), ('Hour-Popularity', 'Popularity')]}
        for k, v in out.items(): print(k, {kk: (np.round(vv, 4) if not isinstance(vv, bool) else vv) for kk, vv in v.items()})
    o, cidx = setup_base(); o0 = o.drop(columns='customer_id')
    if mode == 'abl':
        AS = pd.read_csv(os.path.join(exp2.DATA, 'simulated_customer_assignments.csv'), dtype=str, encoding='utf-8-sig')
        rows = []
        for (w, s), a in AS[AS.set == 'sensitivity'].groupby(['item_w', 'seed'], sort=True):
            R, n, _, _ = eval_set(o0.merge(a[['order_id', 'customer_id']], on='order_id'), cidx, ['Popularity', 'Personal', 'RA-Hybrid', 'RA: p_u คงที่', 'RA: γ = 0'])
            rows.append(summarise(R, dict(w=float(w), seed=int(s), n=n))); print(rows[-1]['w'], rows[-1]['seed'], round(rows[-1]['RA-Hybrid-RA: p_u คงที่'], 4), round(rows[-1]['RA-Hybrid-RA: γ = 0'], 4), flush=True)
        out = pd.DataFrame(rows)
    if mode == 'alt':
        rows = []
        for f in sorted(glob.glob(os.environ.get('ALT', os.path.join(HERE, 'alt')) + '/alt_w*_s*.csv')):
            w, s = re.findall(r'alt_w([\d.]+)_s(\d+)', f)[0]
            a = pd.read_csv(f, dtype=str, encoding='utf-8-sig')
            R, n, m, te = eval_set(o0.merge(a[['order_id', 'customer_id']], on='order_id'), cidx, KEY)
            rep = np.mean([np.isin(its, np.where(m['X'][u] > 0)[0]).mean() for u, its in zip(te.ui.values, te['items'].values)])
            rows.append(summarise(R, dict(w=float(w), seed=int(s), n=n, rep_share=rep)))
            print(w, s, n, round(rep, 3), {k: round(v, 4) for k, v in rows[-1].items() if '-' in k and not k.startswith(('lo:', 'hi:', 'sd:'))}, flush=True)
        out = pd.DataFrame(rows)
    if mode == 'perm':
        a0 = o[['order_id', 'customer_id', 'order_date']].drop_duplicates('order_id')
        R, n, _, _ = eval_set(o, cidx, ['Popularity', 'Personal', 'RA-Hybrid'])
        out['observed'] = summarise(R, dict(perm=-1, n=n)); rows = []
        for p in range(20):
            rng = np.random.default_rng(1000 + p); a = a0.copy(); a['ym'] = a.order_date.str[:7]
            a['customer_id'] = a.groupby('ym').customer_id.transform(lambda x: rng.permutation(x.values))
            Rp, n_p, _, _ = eval_set(o0.merge(a[['order_id', 'customer_id']], on='order_id'), cidx, ['Popularity', 'Personal', 'RA-Hybrid'])
            rows.append(summarise(Rp, dict(perm=p, n=n_p))); print(p, n_p, round(rows[-1]['Personal-Popularity'], 4), round(rows[-1]['RA-Hybrid-Personal'], 4), flush=True)
        out['perm'] = pd.DataFrame(rows)
        obs = out['observed']['Personal-Popularity']; null = out['perm']['Personal-Popularity'].values
        out['p_value'] = (1 + (null >= obs).sum()) / (1 + len(null)); print('p =', out['p_value'])
    if mode == 'seedtune':
        AS = pd.read_csv(os.path.join(exp2.DATA, 'simulated_customer_assignments.csv'), dtype=str, encoding='utf-8-sig')
        rows = []
        for s, a in AS[(AS.set == 'sensitivity') & (AS.item_w == '0.6')].groupby('seed'):
            oo = o0.merge(a[['order_id', 'customer_id']], on='order_id'); B, it, U = build(oo); c = Ctx(B, len(it), U)
            trv, tev = temporal(B, MAIN_CUT, lo=VAL_CUT); Ts = tune_on(c, trv, tev, cidx)
            R, n, _, _ = eval_set(oo, cidx, KEY, TT=Ts)
            rows.append(summarise(R, dict(seed=int(s), n=n, T=str(Ts)))); print(s, Ts, round(rows[-1]['RA-Hybrid-Personal'], 4), flush=True)
        out = pd.DataFrame(rows)
    pickle.dump(out, open(os.path.join(os.environ.get('OUTDIR', '.'), f'run4_{mode}.pkl'), 'wb')); print('DONE', mode)
