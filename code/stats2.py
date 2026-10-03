import os, sys, pickle; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2 import *
V=pickle.load(open('v2_results.pkl','rb'))
res={}
for proto in ['temporal','leave_last']:
    R=V[proto]['R']; names=list(R)
    met=pd.DataFrame([{'model':k,**{m:np.nanmean(v[m]) for m in ['ndcg','hr3','hr5','hr10','rec','rec_rep','rec_exp']},
        'ndcg_lo':boot_ci(v['ndcg'])[0],'ndcg_hi':boot_ci(v['ndcg'])[1],
        'hr3_lo':boot_ci(v['hr3'])[0],'hr3_hi':boot_ci(v['hr3'])[1],'hr5_lo':boot_ci(v['hr5'])[0],'hr5_hi':boot_ci(v['hr5'])[1],
        'n_rep':int((~np.isnan(v['rec_rep'])).sum()),'n_exp':int((~np.isnan(v['rec_exp'])).sum())} for k,v in R.items()])
    pairs=[(k,'Popularity') for k in names if k!='Popularity']+[('Context Hybrid','Personal'),('RA-Hybrid','Personal'),('RA-Hybrid','Context Hybrid'),('ItemKNN+R','ItemKNN'),('EASE+R','EASE')]
    P=pd.DataFrame([{'A':a,'B':b,**paired(R[a]['ndcg'],R[b]['ndcg'])} for a,b in pairs])
    P['w_p_holm']=holm(P.w_p_pratt.values)
    # hour-pop vs pop on HR@3/5/10
    hp={m:paired(R['Hour-Popularity'][m],R['Popularity'][m]) for m in ['hr3','hr5','hr10','ndcg']}
    res[proto]=dict(met=met,P=P,hp=hp,n=V[proto]['ntest'],T=V[proto]['T'])
    print('=====',proto,'n',V[proto]['ntest']); print(met[['model','ndcg','ndcg_lo','ndcg_hi','hr3','hr5','hr10','rec_rep','rec_exp']].round(3).to_string())
    print(P[['A','B','mean','sd','win','tie','loss','w_p','w_p_pratt','t_p','w_p_holm']].round(4).to_string())
    print({m:(round(v['mean'],4),round(v['w_p_pratt'],4)) for m,v in hp.items()})
    sdpop=P[P.B=='Popularity'].sd; print('median sd vs pop',sdpop.median(),'range',sdpop.min(),sdpop.max())
pickle.dump(res,open('stats2.pkl','wb'))
