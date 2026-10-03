"""CLI:  python -m nbr <summary|evaluate|repeat|recommend|serve> <orders.csv|xlsx> [options]"""
import argparse, json, os, pickle, sys, time
import pandas as pd
from .data import read_table, canonical, Corpus
from .evaluate import run_experiment
from .repeat import repeat_cycles, due_now, repeat_summary
from .recommend import Recommender


def load(a):
    m = dict(kv.split('=', 1) for kv in a.map.split(',')) if a.map else None
    lines, rep = canonical(read_table(a.data, a.sheet), m)
    return Corpus(lines), rep


def cmd_summary(a):
    c, rep = load(a)
    print(json.dumps(dict(mapping=rep['mapping'], dropped_no_customer=rep['dropped_no_customer'],
                          has_hour=rep['has_hour'], **c.summary()), ensure_ascii=False, indent=1, default=str))


def cmd_evaluate(a):
    c, rep = load(a); os.makedirs(a.out, exist_ok=True)
    print('data:', c.summary()); t0 = time.time()
    r = run_experiment(c, a.cut, a.val_cut)
    T = r['table']; pd.set_option('display.width', 200)
    print(T[['method', 'ndcg10', 'ci_lo', 'ci_hi', 'hr3', 'hr5', 'recall_repeat', 'recall_explore', 'val_ndcg10']].round(4).to_string(index=False))
    print('sample size:', {k: round(float(v), 4) for k, v in r['sample_size'].items()})
    if c.I < 50: print(f'WARNING: แค่ {c.I} SKU -> top-10 ครอบคลุม {10 / c.I:.0%} ของแคตตาล็อก ตัวเลข NDCG@10 แยกวิธีได้ยาก ควรดู HR@3/HR@5 ประกอบ')
    for nm, p in r['params'].items():
        if nm == 'RA-Hybrid' and p.get('e') == 2: print('WARNING: RA-Hybrid เลือก e=2 (ขอบ grid) น้ำหนักสินค้าใหม่สูงสุด ผลอาจอ่อนไหวต่อขนาดแคตตาล็อก')
    print(f"==> recommended method: {r['chosen']}   ({time.time() - t0:.0f}s)")
    with pd.ExcelWriter(os.path.join(a.out, 'evaluation.xlsx')) as w:
        T.to_excel(w, sheet_name='results', index=False)
        pd.DataFrame([r['sample_size']]).T.reset_index().rename(columns={'index': 'item', 0: 'value'}).to_excel(w, sheet_name='sample_size', index=False)
        pd.DataFrame([dict(method=k, params=json.dumps(v, default=str)) for k, v in r['params'].items()]).to_excel(w, sheet_name='params', index=False)
    json.dump(dict(method=r['chosen'], params=r['params'][r['chosen']], mapping=rep['mapping']),
              open(os.path.join(a.out, 'model.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)


def cmd_repeat(a):
    c, _ = load(a); os.makedirs(a.out, exist_ok=True)
    S = repeat_summary(c); cy = repeat_cycles(c); due = due_now(cy, a.today, a.window)
    print(f"overall repeat rate: {S['overall']:.3f} | customer-SKU pairs with cycles: {len(cy)} | due within +-{a.window}d: {len(due)}")
    with pd.ExcelWriter(os.path.join(a.out, 'repeat_discovery.xlsx')) as w:
        S['by_product'].to_excel(w, sheet_name='by_product', index=False); S['by_category'].to_excel(w, sheet_name='by_category', index=False)
        if len(S['by_channel']): S['by_channel'].to_excel(w, sheet_name='by_channel', index=False)
        cy.to_excel(w, sheet_name='cycles', index=False); due.to_excel(w, sheet_name='due_now', index=False)
    print(S['by_category'].round(3).to_string(index=False))


def _fit_rec(a):
    c, _ = load(a); mj = json.load(open(a.model, encoding='utf-8')) if a.model else dict(method='Personal', params={})
    return Recommender(c, mj['method'], mj['params']), c


def cmd_recommend(a):
    rec, c = _fit_rec(a)
    ids = a.customer or c.users
    res = {u: rec.recommend(u, a.k, a.explore_slots) for u in ids}
    if a.customer:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    else:
        rows = [dict(customer_id=u, **r) for u, rs in res.items() for r in rs]
        os.makedirs(a.out, exist_ok=True); p = os.path.join(a.out, 'recommendations.xlsx')
        pd.DataFrame(rows).to_excel(p, index=False); print('wrote', p, len(rows), 'rows')


def cmd_dashboard(a):
    from .dashboard import build
    c, _ = load(a); mj = json.load(open(a.model, encoding='utf-8')) if a.model else dict(method='Personal', params={})
    ev = None
    if a.model and os.path.exists(os.path.join(os.path.dirname(a.model), 'evaluation.xlsx')):
        ev = pd.read_excel(os.path.join(os.path.dirname(a.model), 'evaluation.xlsx'), sheet_name='results')
    os.makedirs(a.out, exist_ok=True)
    print('wrote', build(c, mj['method'], mj['params'], os.path.join(a.out, 'dashboard.html'), a.today, a.window, ev, a.top))


def cmd_serve(a):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import urlparse, parse_qs
    rec, _ = _fit_rec(a)

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            u = urlparse(self.path); q = parse_qs(u.query)
            if u.path != '/recommend' or 'customer' not in q:
                self.send_response(404); self.end_headers(); return
            hour = int(q['hour'][0]) if 'hour' in q else None
            body = json.dumps(rec.recommend(q['customer'][0], int(q.get('k', [5])[0]), int(q.get('explore', [0])[0]), hour),
                              ensure_ascii=False).encode('utf-8')
            self.send_response(200); self.send_header('Content-Type', 'application/json; charset=utf-8'); self.end_headers(); self.wfile.write(body)
    print(f'GET http://127.0.0.1:{a.port}/recommend?customer=C1001&k=5&explore=1'); HTTPServer(('127.0.0.1', a.port), H).serve_forever()


def main(argv=None):
    p = argparse.ArgumentParser(prog='nbr'); sub = p.add_subparsers(dest='cmd', required=True)
    def common(sp):
        sp.add_argument('data'); sp.add_argument('--sheet'); sp.add_argument('--map', help='canonical=คอลัมน์,... เช่น sku=รหัสสินค้า')
        sp.add_argument('--out', default='out')
    s = sub.add_parser('summary'); common(s); s.set_defaults(f=cmd_summary)
    s = sub.add_parser('evaluate'); common(s); s.add_argument('--cut', help='วันตัดแบ่ง train/test (YYYY-MM-DD); ไม่ระบุ = leave-last-basket-out')
    s.add_argument('--val-cut', dest='val_cut'); s.set_defaults(f=cmd_evaluate)
    s = sub.add_parser('repeat'); common(s); s.add_argument('--today'); s.add_argument('--window', type=int, default=14); s.set_defaults(f=cmd_repeat)
    s = sub.add_parser('recommend'); common(s); s.add_argument('--model', help='model.json จาก evaluate'); s.add_argument('--customer', nargs='*')
    s.add_argument('-k', type=int, default=5); s.add_argument('--explore-slots', type=int, default=0); s.set_defaults(f=cmd_recommend)
    s = sub.add_parser('dashboard'); common(s); s.add_argument('--model'); s.add_argument('--today'); s.add_argument('--window', type=int, default=30); s.add_argument('--top', type=int, help='แสดงเฉพาะลูกค้าที่ซื้อบ่อยที่สุด N ราย (ลดขนาดไฟล์)'); s.set_defaults(f=cmd_dashboard)
    s = sub.add_parser('serve'); common(s); s.add_argument('--model'); s.add_argument('--port', type=int, default=8080); s.set_defaults(f=cmd_serve)
    a = p.parse_args(argv); a.f(a)


if __name__ == '__main__':
    main()
