"""Self-contained HTML dashboard (no server, no external requests) for the shop owner."""
import json
import numpy as np, pandas as pd
from .repeat import repeat_cycles, due_now, repeat_summary
from .recommend import Recommender

HTML = r"""<!doctype html><html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NBR — repeat-purchase recommendations</title><style>
:root{--bg:#fafaf7;--fg:#222;--mut:#6b6b66;--card:#fff;--bd:#e3e1d8;--ac:#1b6b4a;--ac2:#b4570a}
@media(prefers-color-scheme:dark){:root{--bg:#161614;--fg:#eee;--mut:#9a9a92;--card:#1f1f1c;--bd:#33332e;--ac:#5cc79a;--ac2:#f0a257}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,"Noto Sans Thai",sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}h1{font-size:20px;margin:8px 0}h2{font-size:16px;margin:24px 0 8px}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.c{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px}.c b{font-size:22px;display:block}.c span{color:var(--mut);font-size:13px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--bd);border-radius:10px;overflow:hidden}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--bd);font-size:14px}th{color:var(--mut);font-weight:600}
.bar{height:8px;background:var(--ac);border-radius:4px}.w{overflow-x:auto}select,input{font:inherit;padding:6px;border:1px solid var(--bd);border-radius:6px;background:var(--card);color:var(--fg)}
.t1{color:var(--ac);font-weight:600}.t2{color:var(--ac2);font-weight:600}.n{color:var(--mut);font-size:13px}
</style></head><body><main>
<h1>NBR — repeat-purchase recommendations</h1><div class="n" id="meta"></div>
<div class="kpi" id="kpi"></div>
<h2>แนะนำสินค้าตะกร้าถัดไปรายลูกค้า</h2>
<div><select id="cust"></select> <label><input type="checkbox" id="ex" checked> กันช่องสุดท้ายให้สินค้าใหม่</label></div>
<div class="w"><table id="rec"></table></div>
<h2>ลูกค้าที่ถึงรอบสั่งซ้ำ (±<span id="win"></span> วันจากวันอ้างอิง)</h2><div class="w"><table id="due"></table></div>
<h2>อัตราซื้อซ้ำตามหมวดสินค้า</h2><div class="w"><table id="cat"></table></div>
<h2>อัตราซื้อซ้ำตามช่องทางขาย</h2><div class="w"><table id="chn"></table></div>
<h2>ผลประเมินโมเดล</h2><div class="w"><table id="ev"></table></div><p class="n" id="note"></p>
<script>
const D=__DATA__;const $=i=>document.getElementById(i);
const tb=(el,h,rows)=>{$(el).innerHTML='<tr>'+h.map(x=>'<th>'+x+'</th>').join('')+'</tr>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')};
const pc=x=>(x*100).toFixed(1)+'%';
$('meta').textContent='ข้อมูล '+D.s.first+' ถึง '+D.s.last+' | วันอ้างอิง '+D.today+' | วิธี: '+D.method;
$('kpi').innerHTML=[['ลูกค้า',D.s.customers],['คำสั่งซื้อ',D.s.baskets],['สินค้า (SKU)',D.s.products],['อัตราซื้อซ้ำรวม',pc(D.overall)],['ลูกค้าซื้อ ≥2 ครั้ง',D.s.customers_2plus]].map(k=>'<div class="c"><b>'+k[1]+'</b><span>'+k[0]+'</span></div>').join('');
$('cust').innerHTML=Object.keys(D.recs).map(u=>'<option>'+u+'</option>').join('');
function rec(){const r=D.recs[$('cust').value][$('ex').checked?'ex':'plain'];
tb('rec',['#','สินค้า','หมวด','ประเภท','ซื้อมาแล้ว','รอบซื้อเฉลี่ย','ครบรอบวันที่'],r.map(x=>[x.rank,x.name,x.category,'<span class="'+(x.type=='ซื้อซ้ำ'?'t1':'t2')+'">'+x.type+'</span>',x.times_bought||'-',x.median_gap_days?x.median_gap_days+' วัน':'-',x.next_due||'-']))}
$('cust').onchange=rec;$('ex').onchange=rec;rec();
$('win').textContent=D.window;
tb('due',['ลูกค้า','สินค้า','ซื้อแล้ว (ครั้ง)','รอบเฉลี่ย (วัน)','ซื้อล่าสุด','ครบรอบ','เลยกำหนด (วัน)'],D.due.map(x=>[x.customer_id,x.name,x.purchases,x.median_gap_days,x.last_purchase,x.next_due,x.overdue_days]));
const mx=Math.max(...D.cat.map(x=>x.repeat_rate),.01);
tb('cat',['หมวด','รายการที่วิเคราะห์','อัตราซื้อซ้ำ',''],D.cat.map(x=>[x.category,x.lines,pc(x.repeat_rate),'<div class="bar" style="width:'+x.repeat_rate/mx*120+'px"></div>']));
tb('chn',['ช่องทาง','รายการที่วิเคราะห์','อัตราซื้อซ้ำ'],D.chn.map(x=>[x.channel,x.lines,pc(x.repeat_rate)]));
tb('ev',['วิธี','NDCG@10','HR@3','HR@5','NDCG@10 (validation)'],D.ev.map(x=>[x.method,x.ndcg10.toFixed(3),x.hr3.toFixed(3),x.hr5.toFixed(3),x.val_ndcg10.toFixed(3)]));
$('note').textContent=D.note;
</script></main></body></html>"""


def build(corpus, method, params, out_html, today=None, window=30, evaluation=None, top=None):
    today = pd.Timestamp(today) if today else corpus.B.d.max()
    S = repeat_summary(corpus); cy = repeat_cycles(corpus); due = due_now(cy, today, window)
    rec = Recommender(corpus, method, params)
    users = corpus.users if not top else list(corpus.B.groupby('u').size().sort_values(ascending=False).head(top).index)
    recs = {u: dict(plain=rec.recommend(u, 5, 0), ex=rec.recommend(u, 5, 1)) for u in users}
    if top: due = due.sort_values('overdue_days', ascending=False).head(200)
    date_cols = ['last_purchase', 'next_due']
    due = due.assign(**{c: due[c].dt.strftime('%Y-%m-%d') for c in date_cols})
    ev = evaluation if evaluation is not None else pd.DataFrame(columns=['method', 'ndcg10', 'hr3', 'hr5', 'val_ndcg10'])
    ch = S['by_channel'] if len(S['by_channel']) else pd.DataFrame(columns=['channel', 'lines', 'repeat_rate'])
    data = dict(s=corpus.summary(), today=str(today.date()), window=window, method=method, overall=S['overall'], recs=recs,
                due=due[['customer_id', 'name', 'purchases', 'median_gap_days', 'last_purchase', 'next_due', 'overdue_days']].to_dict('records'),
                cat=S['by_category'].to_dict('records'), chn=ch.to_dict('records'),
                ev=ev[['method', 'ndcg10', 'hr3', 'hr5', 'val_ndcg10']].to_dict('records'),
                note='ลูกค้าทดสอบมีจำนวนน้อย แยกความต่างระหว่างวิธีไม่ได้ ใช้ Personal (ซื้อซ้ำตามประวัติ) เป็นฐาน และควรประเมินซ้ำเมื่อมีข้อมูลมากขึ้น')
    js = json.dumps(data, ensure_ascii=False, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else str(o)).replace('</', '<\\/')
    js = js.replace('NaN', 'null')
    open(out_html, 'w', encoding='utf-8').write(HTML.replace('__DATA__', js))
    return out_html
