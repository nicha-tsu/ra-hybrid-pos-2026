import os
import pandas as pd, numpy as np
rng=np.random.default_rng(int(os.environ.get('SEED','2569')))
# Simulates customer IDs for the released (anonymised) receipts. All customer IDs are SIMULATED, not real behaviour.
DATA=os.environ.get('DATA',os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data'))
d=pd.read_csv(os.path.join(DATA,'orders.csv'),dtype=str,encoding='utf-8-sig')
d=d.sort_values('row_seq',key=lambda s:s.astype(int)).reset_index(drop=True)   # original row order (keeps tie-breaking identical)
d['date']=pd.to_datetime(d.order_date); d['qty']=d.qty.astype(float)
d['hour']=pd.to_numeric(d.order_time.str.split(':').str[0],errors='coerce').fillna(12)
skus=d.product_name.value_counts(); V=len(skus); sid={s:i for i,s in enumerate(skus.index)}
pop=(skus.values+1)/(skus.values.sum()+V)
d['sidx']=d.product_name.map(sid)
R=d.groupby('order_id').agg(date=('date','first'),hour=('hour','first'),pay=('payment_type','first'),
    items=('sidx',lambda x: sorted(set(x))),lines=('sidx','size'),qty=('qty','sum')).sort_values(['date','hour']).reset_index()
R['group']=(R.lines>=6)|(R.qty>=10)
D0,D1=R.date.min(),R.date.max(); days=(D1-D0).days+1
print('receipts',len(R),'group',R.group.sum())
# ---------- customer population
rows=[]
def mk(seg,n,mean_v,sd_v,hmu,hsd,span):
    for _ in range(n):
        if span=='all':
            s=rng.integers(0,days//4) if rng.random()<.3 else 0; e=days-1 if rng.random()<.8 else rng.integers(days//2,days)
        else:
            L=int(rng.uniform(120,540)); s=int(rng.integers(-L//2,days-30)); e=min(days-1,s+L); s=max(0,s)
        rows.append(dict(segment=seg,start=s,end=e,target=max(2,rng.lognormal(np.log(mean_v),sd_v)),
            hmu=rng.normal(*hmu),hsd=rng.uniform(*hsd),ptransfer=rng.beta(*( (4,2) if seg!='บุคลากร' else (2,3)))))
mk('บุคลากร',260,55,0.7,(9.5,1.6),(1.0,2.2),'all')
mk('นักศึกษา',3300,8,0.8,(12.5,2.2),(1.2,2.8),'win')
mk('หน่วยงาน/โครงการ',25,6,0.6,(10,1.5),(1,2),'all')
C=pd.DataFrame(rows); NC=len(C)
# taste seeds: 3 favourite products each, sampled from pop^0.7
pw=pop**0.7; pw/=pw.sum()
seed=np.zeros((NC,V),np.float32)
for c in range(NC):
    for j in rng.choice(V,3,replace=False,p=pw): seed[c,j]+=rng.uniform(2,5)
ALPHA=4.0; ITEM_W=float(os.environ.get('ITEM_W','0.6'))
cnt=np.zeros((NC,V),np.float32); n_c=np.zeros(NC); lastday=np.full(NC,-1); today_n=np.zeros(NC)
seg=C.segment.values; start=C.start.values; end=C.end.values; target=C.target.values
hmu=C.hmu.values; hsd=C.hsd.values; ptr=C.ptransfer.values
rate=target/np.maximum(end-start+1,1)
isorg=seg=='หน่วยงาน/โครงการ'
assign=np.empty(len(R),object); nextvis=0; P_VIS=0.055
for k,r in enumerate(R.itertuples()):
    t=(r.date-D0).days
    if (not r.group) and rng.random()<P_VIS:
        nextvis+=1; assign[k]=('V',nextvis); continue
    act=np.where((start<=t)&(end>=t)&(isorg==r.group))[0]
    tn=np.where(lastday[act]==t,today_n[act],0)
    bal=np.clip((target[act]-n_c[act]+1)/(target[act]+1),0.03,None)
    s=np.log(rate[act])+np.log(bal)
    s+=-0.5*((r.hour-hmu[act])/hsd[act])**2-np.log(hsd[act])
    ist=1.0 if r.pay=='โอนเงิน' else 0.0
    s+=np.log(np.where(ist,ptr[act],1-ptr[act])+1e-3)
    it=r.items
    th=(ALPHA*pop[it][None,:]+cnt[np.ix_(act,it)]+seed[np.ix_(act,it)])/(ALPHA+n_c[act]+seed[act].sum(1))[:,None]
    s+=ITEM_W*np.log(th/pop[it][None,:]).sum(1)
    s-=2.5*tn
    p=np.exp(s-s.max()); p/=p.sum()
    c=act[rng.choice(len(act),p=p)]
    assign[k]=('C',c)
    cnt[c,it]+=1; n_c[c]+=1
    today_n[c]=today_n[c]+1 if lastday[c]==t else 1; lastday[c]=t
# ---------- ids
order=np.argsort(start,kind='stable')
cid={c:f'C{10001+i}' for i,c in enumerate(order)}
R['customer_id']=[cid[a[1]] if a[0]=='C' else f'G{50000+a[1]}' for a in assign]
R['segment_sim']=[seg[a[1]] if a[0]=='C' else 'ผู้มาเยือน (ซื้อครั้งเดียว)' for a in assign]
R[['order_id','customer_id','segment_sim']].rename(columns={'segment_sim':'segment_simulated'}).to_csv(os.environ.get('OUT','assign_sim.csv'),index=False,encoding='utf-8-sig')
print(R.groupby('customer_id').size().describe())
