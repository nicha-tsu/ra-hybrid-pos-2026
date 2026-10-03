"""Round-3 full rerun: every table/figure recomputed with the tuned Personal baseline (C16) and the added
NBR baselines TIFU-KNN and UP-CF@r (C15) plus ablations (C11). Output: run3.pkl"""
import os, sys, time, pickle, itertools, glob, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp2
from exp2 import *
from exp3 import evaluate3, add_last, personal_t, ptop, seqs, tifu_vectors, tifu, upcf_vectors, upcf, ra_const, nd

def set_personal(beta):
    exp2.personal = lambda c, m, te, **k: personal_t(c, m, te, beta=beta)

def tune_on(c, trv, tev, cidx):
    mv = mats(c, trv); X = mv['X']; T = {}
    best = (-1, None)
    for rho in [0.5, 0.7, 0.8, 0.9, 1.0]:
        add_last(c, trv, mv, rho)
        for beta in [0, 0.25, 0.5, 1, 2, 4, 8]:
            s = nd(personal_t(c, mv, tev, beta), tev, X)
            if s > best[0]: best = (s, dict(rho=rho, beta=beta))
    T['pers'] = best[1]; add_last(c, trv, mv, T['pers']['rho']); set_personal(T['pers']['beta'])
    T['knn_k'] = max([3, 5, 10, 30, 100, 0], key=lambda k: nd(itemknn(c, mv, tev, k), tev, X))
    T['uknn_k'] = max([100, 200, 400, 800], key=lambda k: nd(userknn(c, mv, tev, k), tev, X))
    T['lam'] = max([0.1, 0.3, 1, 3, 10, 50, 200, 1000], key=lambda l: nd(ease(c, mv, tev, l), tev, X))
    T['als'] = max([dict(f=16, reg=0.1, alpha=10), dict(f=32, reg=0.1, alpha=10), dict(f=32, reg=1, alpha=20)], key=lambda p: nd(ials(c, mv, tev, **p), tev, X))
    T['knn_r'] = max([dict(k=k, w=w) for k in [5, 10, 30] for w in [0.1, 0.3, 0.6, 1.0, 1.5, 2.5]], key=lambda p: nd(knn_r(c, mv, tev, **p), tev, X))
    T['ease_r'] = max([dict(lam=l, w=w) for l in [10, 200, 1000, 3000, 10000, 30000] for w in [0.1, 0.3, 0.6, 1.0, 1.5]], key=lambda p: nd(ease_r(c, mv, tev, **p), tev, X))
    T['ctx'] = max([dict(g=g) for g in [0, 0.15, 0.3, 0.6, 1, 1.5, 2.5]], key=lambda p: nd(ctx_hybrid(c, mv, tev, **p), tev, X))
    grid = [dict(base=b, e=e, g=g, lam=T['lam'], k=T['knn_k']) for b in ['ease', 'knn'] for e in [0, 0.1, 0.25, 0.5, 1, 2] for g in [0, 0.05, 0.15, 0.3, 0.6, 1]]
    T['ra'] = max(grid, key=lambda p: nd(ra_hybrid(c, mv, tev, **p), tev, X))
    pbar = float(mv['prop'][np.unique(trv.ui)].mean())
    T['ra_const'] = max([dict(T['ra'], g=g, pbar=pbar) for g in [0, 0.05, 0.1, 0.15, 0.3, 0.6, 1]], key=lambda p: nd(ra_const(c, mv, tev, **p), tev, X))
    sq = seqs(c, trv); best = (-1, None)
    for gs, rb, rg in itertools.product([3, 7], [0.9, 1.0], [0.6, 0.9]):
        V = tifu_vectors(c, sq, gs, rb, rg)
        for k, al in itertools.product([100, 300], [0.5, 0.7, 0.9]):
            s = nd(tifu(c, mv, tev, V, k, al), tev, X)
            if s > best[0]: best = (s, dict(gsize=gs, rb=rb, rg=rg, k=k, alpha=al))
    T['tifu'] = best[1]; best = (-1, None)
    for r in [1, 3, 5, None]:
        P = upcf_vectors(c, sq, r)
        for a, q, lam in itertools.product([0.5, 0.75], [1, 5], [0.1, 0.5, 1.0]):
            s = nd(upcf(c, mv, tev, P, a, q, lam), tev, X)
            if s > best[0]: best = (s, dict(r=r, a=a, q=q, lam=lam))
    T['upcf'] = best[1]
    return T

MAIN = ['Popularity', 'Hour-Popularity', 'P-TopFreq', 'Personal', 'ItemKNN', 'UserKNN', 'ItemKNN+R', 'EASE', 'EASE+R', 'iALS',
        'Content/Ontology', 'TIFU-KNN', 'UP-CF@r', 'Context Hybrid', 'RA-Hybrid', 'RA: p_u คงที่', 'RA: γ = 0']
def scores(c, tr, te, T, cidx, only=None):
    m = mats(c, tr); add_last(c, tr, m, T['pers']['rho']); set_personal(T['pers']['beta']); sq = seqs(c, tr)
    F = {'Popularity': lambda: pop(c, m, te), 'Hour-Popularity': lambda: hourpop(c, m, te), 'P-TopFreq': lambda: ptop(c, m, te),
         'Personal': lambda: personal_t(c, m, te, T['pers']['beta']),
         'ItemKNN': lambda: itemknn(c, m, te, T['knn_k']), 'UserKNN': lambda: userknn(c, m, te, T['uknn_k']),
         'ItemKNN+R': lambda: knn_r(c, m, te, **T['knn_r']), 'EASE': lambda: ease(c, m, te, T['lam']),
         'EASE+R': lambda: ease_r(c, m, te, **T['ease_r']), 'iALS': lambda: ials(c, m, te, **T['als']),
         'Content/Ontology': lambda: content(c, m, te, cidx=cidx),
         'TIFU-KNN': lambda: tifu(c, m, te, tifu_vectors(c, sq, T['tifu']['gsize'], T['tifu']['rb'], T['tifu']['rg']), T['tifu']['k'], T['tifu']['alpha']),
         'UP-CF@r': lambda: upcf(c, m, te, upcf_vectors(c, sq, T['upcf']['r']), T['upcf']['a'], T['upcf']['q'], T['upcf']['lam']),
         'Context Hybrid': lambda: ctx_hybrid(c, m, te, **T['ctx']), 'RA-Hybrid': lambda: ra_hybrid(c, m, te, **T['ra']),
         'RA: p_u คงที่': lambda: ra_const(c, m, te, **T['ra_const']), 'RA: γ = 0': lambda: ra_hybrid(c, m, te, **dict(T['ra'], g=0))}
    S, tm = {}, {}
    for k in (only or MAIN):
        t0 = time.perf_counter(); S[k] = F[k](); tm[k] = time.perf_counter() - t0
    return S, m, tm

def deep(S, te, X, I):
    """Li et al. [1] style: RepR/ExplR, repeat/explore recall & PHR, NDCG split, coverage, exposure."""
    ui = te.ui.values; its = te['items'].values; n = len(te); H = X[ui] > 0; DISC10 = DISC
    rows, per = [], {}
    for name, Sm in S.items():
        Su = Sm[ui]; top = np.argsort(-Su, 1)[:, :10]; valid = np.isfinite(np.take_along_axis(Su, top, 1))
        isrep = np.take_along_axis(H, top, 1)
        RepR = (isrep & valid).sum(1) / 10; ExplR = ((~isrep) & valid).sum(1) / 10
        ndl, nr, ne, rr, pr, re_, pe = [], [], [], [], [], [], []
        for k in range(n):
            t = top[k][valid[k]]; hit = np.isin(t, its[k]); d = DISC10[:len(t)]; idcg = DISC10[:min(len(its[k]), 10)].sum(); rp = isrep[k][valid[k]]
            ndl.append((hit * d).sum() / idcg); nr.append((hit * rp * d).sum() / idcg); ne.append((hit * (~rp) * d).sum() / idcg)
            g_rep = its[k][H[k][its[k]]]; g_exp = its[k][~H[k][its[k]]]
            if len(g_rep): x = np.isin(g_rep, t); rr.append(x.mean()); pr.append(float(x.any()))
            if len(g_exp): x = np.isin(g_exp, t); re_.append(x.mean()); pe.append(float(x.any()))
        per[name] = np.array(ndl)
        cnt = np.bincount(top[valid].ravel(), minlength=I); sc = np.sort(cnt)[::-1]
        gini = 1 - 2 * (np.cumsum(np.sort(cnt)) / cnt.sum()).mean() + 1 / I
        rows.append(dict(model=name, NDCG=np.mean(ndl), NDCG_rep=np.mean(nr), NDCG_expl=np.mean(ne), RepR=RepR.mean(), ExplR=ExplR.mean(),
                         Recall_rep=np.mean(rr), PHR_rep=np.mean(pr), Recall_expl=np.mean(re_), PHR_expl=np.mean(pe),
                         coverage=(cnt > 0).sum() / I, top10_share=sc[:int(np.ceil(I * 0.1))].sum() / cnt.sum(), gini=gini))
    gt = np.array([H[k][it].mean() for k, it in enumerate(its)])
    return pd.DataFrame(rows), per, gt, dict(n_rep=int(sum(H[k][it].any() for k, it in enumerate(its))), n_exp=int(sum((~H[k][it]).any() for k, it in enumerate(its))))

if __name__ == '__main__':
    o = load_orders(); B, items, U = build(o); I = len(items); c = Ctx(B, I, U)
    prod = pd.read_csv(os.path.join(exp2.DATA, 'products.csv'), encoding='utf-8-sig').set_index('sku').reindex(items)
    cats = sorted(prod.category.unique()); cidx = np.array([cats.index(x) for x in prod.category])
    out = {}
    for proto in ['temporal', 'leave_last']:
        t0 = time.time()
        if proto == 'temporal': trv, tev = temporal(B, MAIN_CUT, lo=VAL_CUT); tr, te = temporal(B, MAIN_CUT)
        else: trv, tev = leave_last(B, val=True); tr, te = leave_last(B)
        T = tune_on(c, trv, tev, cidx); ttune = time.time() - t0
        S, m, tm = scores(c, tr, te, T, cidx); X = m['X']
        R = {k: evaluate3(v, te, X) for k, v in S.items()}
        out[proto] = dict(T=T, R={k: {mm: v[mm] for mm in v} for k, v in R.items()}, n=len(te), time=tm, tune_time=ttune, te=te[['ui', 'd', 'h']].reset_index(drop=True))
        print(proto, len(te), T, '%.0fs' % ttune, flush=True)
        for k, v in R.items():
            print(f"  {k:18s} NDCG {v['ndcg'].mean():.4f} N@5 {v['ndcg5'].mean():.4f} MRR {v['mrr'].mean():.4f} HR3 {v['hr3'].mean():.3f} HR5 {v['hr5'].mean():.3f} "
                  f"HR10 {v['hr10'].mean():.3f} RecRep {np.nanmean(v['rec_rep']):.3f} RecExp {np.nanmean(v['rec_exp']):.3f}", flush=True)
        if proto != 'temporal': continue
        # ---- statistics
        pairs = [(k, 'Popularity') for k in MAIN if k != 'Popularity'] + [(k, 'Personal') for k in ['UP-CF@r', 'TIFU-KNN', 'RA-Hybrid', 'Context Hybrid', 'ItemKNN+R', 'EASE+R', 'RA: p_u คงที่', 'RA: γ = 0']] + \
                [('RA-Hybrid', 'RA: p_u คงที่'), ('RA-Hybrid', 'RA: γ = 0'), ('ItemKNN+R', 'ItemKNN'), ('EASE+R', 'EASE'), ('Personal', 'P-TopFreq')]
        P = pd.DataFrame([{'A': a, 'B': b, **paired(R[a]['ndcg'], R[b]['ndcg'])} for a, b in pairs])
        P['t_p'] = P.t_p.fillna(1.0); P['w_p_pratt'] = P.w_p_pratt.fillna(1.0)
        P['w_p_holm'] = holm(P.w_p_pratt.values); P['t_p_holm'] = holm(P.t_p.values)
        out['P'] = P
        print(P[['A', 'B', 'mean', 'ci', 'sd', 'tie', 'w_p_pratt', 't_p', 'w_p_holm', 't_p_holm']].round(4).to_string(), flush=True)
        out['ci'] = {k: {mm: boot_ci(v[mm]) for mm in ['ndcg', 'hr3', 'hr5']} for k, v in R.items()}
        # ---- deep analysis (Tables 4-5)
        D, per, gt, cnts = deep(S, te, X, I); out['deep'] = D; out['gt_rep'] = gt.mean(); out['anyrep'] = (gt > 0).mean(); out['cnts'] = cnts
        H = X[te.ui.values] > 0; out['ptop_slots'] = (np.minimum(H.sum(1), 10) / 10).mean()
        print(D.round(3).to_string(), flush=True)
        bins = [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0]; g = pd.cut(gt, bins, labels=['[0,0.2]', '(0.2,0.4]', '(0.4,0.6]', '(0.6,0.8]', '(0.8,1]'])
        GX = []
        for lab in g.categories:
            mk = np.asarray(g == lab); row = dict(group=lab, n=int(mk.sum()), PAU=mk.mean())
            for k in ['Popularity', 'Personal', 'UP-CF@r', 'RA-Hybrid', 'UserKNN']:
                row[k] = per[k][mk].mean(); row['CAP:' + k] = per[k][mk].sum() / per[k].sum()
            GX.append(row)
        out['G_expost'] = pd.DataFrame(GX); print(out['G_expost'].round(3).to_string(), flush=True)
        # ex-ante groups
        rep, seen = {}, {}
        for ui, its in zip(tr.ui.values, tr['items'].values):
            s = seen.setdefault(ui, set())
            if s: rep.setdefault(ui, []).append(np.mean([i in s for i in its]))
            s.update(its.tolist())
        q = te.ui.values; raw = np.array([np.mean(rep[u]) if u in rep else np.nan for u in q]); nb = tr.groupby('ui').size().reindex(q).values
        cols = ['Popularity', 'Personal', 'UP-CF@r', 'TIFU-KNN', 'RA-Hybrid']
        GA = []
        for by, lab, mk in [('rr', 'มีบิลเดียว', np.isnan(raw)), ('rr', '[0, 0.2]', raw <= .2), ('rr', '(0.2, 0.5]', (raw > .2) & (raw <= .5)), ('rr', '(0.5, 1]', raw > .5),
                            ('nb', '1', nb == 1), ('nb', '2–4', (nb >= 2) & (nb <= 4)), ('nb', '5–9', (nb >= 5) & (nb <= 9)), ('nb', '≥10', nb >= 10)]:
            row = dict(by=by, group=lab, n=int(mk.sum()))
            for k in cols:
                row[k] = R[k]['ndcg'][mk].mean()
            d = R['Personal']['ndcg'][mk] - R['Popularity']['ndcg'][mk]; row['Pers-Pop'] = d.mean(); row['lo'], row['hi'] = boot_ci(d)
            GA.append(row)
        out['G_exante'] = pd.DataFrame(GA); print(out['G_exante'].round(3).to_string(), flush=True)
        # ---- epsilon curve (Fig 2)
        m2 = mats(c, tr); add_last(c, tr, m2, T['pers']['rho']); set_personal(T['pers']['beta'])
        out['curve'] = []
        for e in [0, 0.1, 0.25, 0.5, 1, 2, 4]:
            r = evaluate(ra_hybrid(c, m2, te, **dict(T['ra'], e=e)), te, X)
            out['curve'].append(dict(e=e, ndcg=r['ndcg'].mean(), rec_rep=np.nanmean(r['rec_rep']), rec_exp=np.nanmean(r['rec_exp'])))
        print(out['curve'], flush=True)
        # ---- 2+1 slots (C27)
        Sra = S['RA-Hybrid'][q]; Shp = S['Hour-Popularity'][q]; Hq = X[q] > 0
        def at3(tops):
            hit = np.array([np.isin(t, it).any() for t, it in zip(tops, te['items'].values)], float)
            rr = [np.isin(it[Hq[k][it]], t).mean() for k, (t, it) in enumerate(zip(tops, te['items'].values)) if Hq[k][it].any()]
            re_ = [np.isin(it[~Hq[k][it]], t).mean() for k, (t, it) in enumerate(zip(tops, te['items'].values)) if (~Hq[k][it]).any()]
            newhit = np.array([np.isin(t[~Hq[k][t]], it).any() for k, (t, it) in enumerate(zip(tops, te['items'].values))], float)
            return dict(HR3=hit.mean(), Rec_rep3=np.mean(rr), Rec_expl3=np.mean(re_), new_hit=newhit.mean(), _hit=hit, _new=newhit)
        ra3 = np.argsort(-Sra, 1)[:, :3]; t21 = []
        for k in range(len(q)):
            h = [i for i in np.argsort(-np.where(Hq[k], Sra[k], -np.inf)) if Hq[k][i]][:2]
            nw = [i for i in np.argsort(-np.where(Hq[k], -np.inf, Shp[k])) if not Hq[k][i]]
            t21.append((h + nw)[:3])
        SL = {'RA-Hybrid 3 ช่อง': at3(ra3), 'ซื้อซ้ำ 2 + ใหม่ 1': at3(np.array(t21)), 'Popularity 3 ช่อง': at3(np.argsort(-S['Popularity'][q], 1)[:, :3])}
        out['slots'] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')} for k, v in SL.items()}
        out['slot_test'] = dict(hr=paired(SL['ซื้อซ้ำ 2 + ใหม่ 1']['_hit'], SL['RA-Hybrid 3 ช่อง']['_hit']), new=paired(SL['ซื้อซ้ำ 2 + ใหม่ 1']['_new'], SL['RA-Hybrid 3 ช่อง']['_new']))
        print(out['slots'], {k: (v['mean'], v['ci'], v['t_p']) for k, v in out['slot_test'].items()}, flush=True)
        # ---- robustness: cutoffs (Table 3)
        cuts = []
        for cut in ['2023-05-01', '2023-08-01', '2023-11-01']:
            trc, tec = temporal(B, cut); Sc, mc, _ = scores(c, trc, tec, T, cidx, only=['Popularity', 'Personal', 'UP-CF@r', 'TIFU-KNN', 'RA-Hybrid'])
            Rc = {k: evaluate(v, tec, mc['X'])['ndcg'] for k, v in Sc.items()}
            row = dict(cut=cut, n=len(tec))
            for a, b in [('Personal', 'Popularity'), ('UP-CF@r', 'Personal'), ('TIFU-KNN', 'Personal'), ('RA-Hybrid', 'Personal')]:
                row[f'{a}-{b}'] = paired(Rc[a], Rc[b])
            cuts.append(row); print(cut, len(tec), {k: (round(v['mean'], 4), np.round(v['ci'], 4), round(v['t_p'], 4)) for k, v in row.items() if isinstance(v, dict)}, flush=True)
        out['cuts'] = cuts
        pickle.dump(out, open('run3.pkl', 'wb'))
    # ---- simulator sensitivity (Fig 4) + seeds, parameters fixed from temporal tuning
    T = out['temporal']['T']; o0 = o.drop(columns='customer_id'); rows = []
    AS = pd.read_csv(os.path.join(exp2.DATA, 'simulated_customer_assignments.csv'), dtype=str, encoding='utf-8-sig')
    for (w, s), a in AS[AS.set == 'sensitivity'].groupby(['item_w', 'seed'], sort=True):
        oo = o0.merge(a[['order_id', 'customer_id']], on='order_id'); B2, it2, U2 = build(oo); c2 = Ctx(B2, len(it2), U2)
        tr2, te2 = temporal(B2, MAIN_CUT)
        S2, m2, _ = scores(c2, tr2, te2, T, cidx, only=['Popularity', 'Hour-Popularity', 'Personal', 'UP-CF@r', 'TIFU-KNN', 'RA-Hybrid'])
        R2 = {k: evaluate(v, te2, m2['X'])['ndcg'] for k, v in S2.items()}
        rep_share = np.mean([np.isin(its, np.where(m2['X'][u] > 0)[0]).mean() for u, its in zip(te2.ui.values, te2['items'].values)])
        row = dict(w=float(w), seed=int(s), n=len(te2), rep_share=rep_share, **{k: v.mean() for k, v in R2.items()})
        for a_, b_ in [('Personal', 'Popularity'), ('RA-Hybrid', 'Personal'), ('UP-CF@r', 'Personal'), ('TIFU-KNN', 'Personal')]:
            d = R2[a_] - R2[b_]; row[f'{a_}-{b_}'] = d.mean(); row[f'sd:{a_}-{b_}'] = d.std(ddof=1)
            lo, hi = boot_ci(d, B=2000); row[f'lo:{a_}-{b_}'] = lo; row[f'hi:{a_}-{b_}'] = hi
        rows.append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if not k.startswith(('lo:', 'hi:'))}, flush=True)
    out['sens'] = pd.DataFrame(rows)
    pickle.dump(out, open('run3.pkl', 'wb'))
    print('DONE', flush=True)
