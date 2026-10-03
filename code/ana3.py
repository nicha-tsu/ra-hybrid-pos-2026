import os, sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2 import *
V=pickle.load(open('v2_results.pkl','rb'))['temporal']; T=V['T']
o=load_orders(); B,items,U=build(o); I=len(items); c=Ctx(B,I,U)
prod=pd.read_csv(os.path.join(DATA, 'products.csv'), encoding='utf-8-sig').set_index('sku').reindex(items)
cats=sorted(prod.category.unique()); cidx=np.array([cats.index(x) for x in prod.category])
tr,te=temporal(B,MAIN_CUT); m=mats(c,tr); X=m['X']
def ptop(c,m,te): # pure P-TopFreq: only history items, others -inf
    S=m['X'].astype(float).copy(); S[S==0]=-np.inf; return S
S={'Popularity (G-TopFreq)':pop(c,m,te),'Hour-Popularity':hourpop(c,m,te),'P-TopFreq':ptop(c,m,te),'Personal (GP-TopFreq+recency)':personal(c,m,te),
   'ItemKNN':itemknn(c,m,te,T['knn_k']),'UserKNN':userknn(c,m,te,T['uknn_k']),'ItemKNN+R':knn_r(c,m,te,**T['knn_r']),
   'EASE':ease(c,m,te,T['lam']),'EASE+R':ease_r(c,m,te,**T['ease_r']),'iALS':ials(c,m,te,**T['als']),
   'Content/Ontology':content(c,m,te,cidx=cidx),'Context Hybrid':ctx_hybrid(c,m,te,**T['ctx']),'RA-Hybrid':ra_hybrid(c,m,te,**T['ra'])}
ui=te.ui.values; its=te['items'].values; n=len(te)
H=X[ui]>0
gt_rep=np.array([H[k][it].mean() for k,it in enumerate(its)])   # ground-truth repetition ratio
print('n',n,'GT RepR mean',gt_rep.mean().round(3),'users with any repeat',(gt_rep>0).mean().round(3))
slot=np.minimum(H.sum(1),10)/10; print('P-TopFreq slot usage@10',slot.mean().round(4))
DISC=1/np.log2(np.arange(2,12))
rows=[]; per={}
for name,Sm in S.items():
    Su=Sm[ui]; top=np.argsort(-Su,1)[:,:10]
    valid=np.isfinite(np.take_along_axis(Su,top,1))
    isrep=np.take_along_axis(H,top,1)
    RepR=(isrep&valid).sum(1)/10; ExplR=((~isrep)&valid).sum(1)/10
    nd=[];nd_r=[];nd_e=[];rr=[];re=[];pr=[];pe=[];rec=[]
    for k in range(n):
        t=top[k][valid[k]]; hit=np.isin(t,its[k]); d=DISC[:len(t)]; idcg=DISC[:min(len(its[k]),10)].sum()
        rp=isrep[k][valid[k]]
        nd.append((hit*d).sum()/idcg); nd_r.append((hit*rp*d).sum()/idcg); nd_e.append((hit*(~rp)*d).sum()/idcg)
        rec.append(hit.sum()/len(its[k]))
        g_rep=its[k][H[k][its[k]]]; g_exp=its[k][~H[k][its[k]]]
        if len(g_rep): x=np.isin(g_rep,t); rr.append(x.mean()); pr.append(float(x.any()))
        if len(g_exp): x=np.isin(g_exp,t); re.append(x.mean()); pe.append(float(x.any()))
    nd=np.array(nd); per[name]=nd
    exp_items=top[valid]; cnt=np.bincount(exp_items.ravel(),minlength=I); sc=np.sort(cnt)[::-1]
    gini=(np.cumsum(np.sort(cnt))/cnt.sum()); gini=1-2*gini.mean()+1/I
    rows.append(dict(model=name,NDCG=nd.mean(),NDCG_rep=np.mean(nd_r),NDCG_expl=np.mean(nd_e),Recall=np.mean(rec),
        RepR=RepR.mean(),ExplR=ExplR.mean(),Recall_rep=np.mean(rr),PHR_rep=np.mean(pr),Recall_expl=np.mean(re),PHR_expl=np.mean(pe),
        coverage=(cnt>0).sum()/I, top10pct_share=sc[:int(np.ceil(I*0.1))].sum()/cnt.sum(), gini=gini))
D=pd.DataFrame(rows); pd.set_option('display.width',250)
print(D.round(3).to_string())
# user groups by GT repetition ratio
bins=[-0.01,0.2,0.4,0.6,0.8,1.0]; g=pd.cut(gt_rep,bins,labels=['[0,0.2]','(0.2,0.4]','(0.4,0.6]','(0.6,0.8]','(0.8,1]'])
G=[]
for lab in g.categories:
    msk=(g==lab)
    row=dict(group=lab,n=int(msk.sum()),PAU=msk.mean())
    for name in ['Popularity (G-TopFreq)','Personal (GP-TopFreq+recency)','Context Hybrid','RA-Hybrid','UserKNN','ItemKNN+R']:
        row[name]=per[name][msk].mean(); row['CAP:'+name]=per[name][msk].sum()/per[name].sum()
    G.append(row)
G=pd.DataFrame(G); print(G.round(3).to_string())
pickle.dump(dict(D=D,G=G,gt_rep=gt_rep.mean(),slot=slot.mean(),n=n,anyrep=(gt_rep>0).mean()),open('ana3.pkl','wb'))
