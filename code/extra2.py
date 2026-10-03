import os, sys,pickle; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2 import *
V=pickle.load(open('v2_results.pkl','rb')); T=V['temporal']['T']
o=load_orders(); B,items,U=build(o); c=Ctx(B,len(items),U)
out={'cuts':[],'curve':[]}
for cut in ['2023-05-01','2023-08-01','2023-11-01']:
    tr,te=temporal(B,cut); m=mats(c,tr); X=m['X']
    R={'Popularity':evaluate(pop(c,m,te),te,X),'Personal':evaluate(personal(c,m,te),te,X),
       'Context Hybrid':evaluate(ctx_hybrid(c,m,te,**T['ctx']),te,X),'RA-Hybrid':evaluate(ra_hybrid(c,m,te,**T['ra']),te,X)}
    row={'cut':cut,'n':len(te)}
    for a,b in [('Personal','Popularity'),('Context Hybrid','Personal'),('RA-Hybrid','Personal')]:
        p=paired(R[a]['ndcg'],R[b]['ndcg']); row[f'{a}-{b}']=p
    out['cuts'].append(row)
    print(cut,len(te),{k:(round(v['mean'],4),[round(x,4) for x in v['ci']],round(v['t_p'],4),round(v['w_p_pratt'],4)) for k,v in row.items() if isinstance(v,dict)},flush=True)
tr,te=temporal(B,'2023-08-01'); m=mats(c,tr); X=m['X']
for e in [0,0.1,0.25,0.5,1,2,4]:
    r=evaluate(ra_hybrid(c,m,te,**{**T['ra'],'e':e}),te,X)
    out['curve'].append(dict(e=e,ndcg=r['ndcg'].mean(),rec_rep=np.nanmean(r['rec_rep']),rec_exp=np.nanmean(r['rec_exp']),hr5=r['hr5'].mean()))
    print(out['curve'][-1],flush=True)
pickle.dump(out,open('extra2.pkl','wb'))
