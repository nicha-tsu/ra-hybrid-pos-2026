"""NBR local web application (stdlib only). Run:  python app/server.py [--port 8080]
Serves a single-page UI and a JSON API on 127.0.0.1 only; data never leaves the machine."""
import argparse, json, os, sys, tempfile, threading, time, traceback, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from nbr.data import read_table, canonical, Corpus
from nbr.evaluate import run_experiment
from nbr.recommend import Recommender
from nbr.repeat import repeat_cycles, due_now, repeat_summary
from i18n_en import name_en, category_en, channel_en, type_en

MAX_UPLOAD = 150 * 1024 * 1024
DEFAULT = os.path.join(ROOT, 'results', 'pos')


def jdefault(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp,)): return str(o.date())
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def clean(df):
    d = df.copy()
    for c in d.columns:
        if str(d[c].dtype).startswith('datetime'): d[c] = d[c].dt.strftime('%Y-%m-%d')
    return json.loads(json.dumps(d.astype(object).where(d.notna(), None).to_dict('records'), default=jdefault))


def en_df(df):
    d = df.copy()
    for col, fn in (('name', name_en), ('category', category_en), ('channel', channel_en)):
        if col in d: d[col] = d[col].map(fn)
    return d


def en_error(msg):
    if 'ไม่พบคอลัมน์จำเป็น' in msg:
        return 'Required columns not found (need customer ID, date, order/bill number, product code). Use the column-mapping box. Details: ' + msg.split(':', 1)[-1].strip()
    if 'ลูกค้าที่ซื้อซ้ำน้อยเกินไป' in msg:
        return 'Too few repeat customers to evaluate (need at least 10 customers with 3 or more orders).'
    return msg


class Store:
    """Everything the UI needs for one dataset."""
    def __init__(self, corpus, rep, method, params, table, sample, name, simulated):
        self.c, self.rep, self.method, self.params = corpus, rep, method, params
        self.table, self.sample, self.name, self.simulated = table, sample, name, simulated
        self.rec = Recommender(corpus, method, params)
        self.summary = repeat_summary(corpus); self.cycles = repeat_cycles(corpus)
        self.today = corpus.B.d.max() + pd.Timedelta(days=1)
        self.nb = corpus.B.groupby('u').size()


STORE = None; LOCK = threading.Lock(); JOBS = {}


def load_default():
    csv = os.path.join(DEFAULT, 'orders_main.csv'); mj = json.load(open(os.path.join(DEFAULT, 'model.json'), encoding='utf-8'))
    lines, rep = canonical(read_table(csv)); c = Corpus(lines)
    ev = os.path.join(DEFAULT, 'evaluation.xlsx')
    table = pd.read_excel(ev, sheet_name='results'); ss = pd.read_excel(ev, sheet_name='sample_size')
    sample = {r['item']: r['value'] for _, r in ss.iterrows()}
    return Store(c, rep, mj['method'], mj['params'], table, sample, 'Main POS dataset', True)


def run_job(jid, path, name, mapping, cut, simulated):
    j = JOBS[jid]; log = lambda s: j['log'].append(str(s))
    try:
        log('Reading file and matching columns...')
        lines, rep = canonical(read_table(path), mapping); c = Corpus(lines)
        log(f"Customers {c.U} | Products {c.I} | Orders {len(c.B)} | Lines dropped (no customer ID) {rep['dropped_no_customer']}")
        r = run_experiment(c, cut or None, None, log=log)
        log('Building model and analysing repeat purchases...')
        s = Store(c, rep, r['chosen'], r['params'][r['chosen']], r['table'], r['sample_size'], name, simulated)
        global STORE
        with LOCK: STORE = s
        j['state'] = 'done'; log(f"Done. Selected method: {r['chosen']}")
    except Exception as e:
        j['state'] = 'error'; log('Error: ' + en_error(str(e))); traceback.print_exc()
    finally:
        try: os.remove(path)
        except OSError: pass


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, code, body, ctype='application/json; charset=utf-8'):
        if not isinstance(body, bytes): body = json.dumps(body, ensure_ascii=False, default=jdefault).encode('utf-8')
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ('/', '/index.html'):
                return self.send(200, open(os.path.join(HERE, 'static', 'index.html'), 'rb').read(), 'text/html; charset=utf-8')
            if u.path == '/manifest.webmanifest':
                return self.send(200, open(os.path.join(HERE, 'static', 'manifest.webmanifest'), 'rb').read(), 'application/manifest+json')
            if u.path == '/icon.svg':
                return self.send(200, open(os.path.join(HERE, 'static', 'icon.svg'), 'rb').read(), 'image/svg+xml')
            if u.path == '/api/job':
                j = JOBS.get(q.get('id')); return self.send(200 if j else 404, j or {'error': 'Job not found'})
            with LOCK: s = STORE
            if s is None: return self.send(503, {'error': 'No data loaded'})
            c = s.c
            if u.path == '/api/status':
                t = s.table
                return self.send(200, dict(name=s.name, simulated=s.simulated, summary=c.summary(), method=s.method, params=s.params,
                                           has_hour=s.rep['has_hour'], dropped_no_customer=s.rep['dropped_no_customer'],
                                           results=clean(t), sample={k: v for k, v in s.sample.items()},
                                           overall_repeat=s.summary['overall'], today=str(s.today.date())))
            if u.path == '/api/customers':
                qq = q.get('q', '').strip().lower(); n = s.nb
                ids = [x for x in n.index if x.lower().startswith(qq)] if qq else list(n.sort_values(ascending=False).index)
                ids = sorted(ids, key=lambda x: -n[x])[:40]
                return self.send(200, [dict(id=x, baskets=int(n[x])) for x in ids])
            if u.path == '/api/customer':
                cid = q.get('id', ''); B = c.B[c.B.u == cid]
                if B.empty: return self.send(404, {'error': 'Customer not found'})
                last = B.tail(8).iloc[::-1]
                hist = [dict(date=str(r.d.date()), order=r.order_id, items=[name_en(c.name[c.items[i]]) for i in r['items']]) for _, r in last.iterrows()]
                rec = s.rec.recommend(cid, int(q.get('k', 5)), int(q.get('explore', 0)), int(q['hour']) if q.get('hour') else None)
                for x in rec: x['name'], x['category'], x['type'] = name_en(x['name']), category_en(x['category']), type_en(x['type'])
                return self.send(200, dict(id=cid, baskets=len(B), first=str(B.d.min().date()), last=str(B.d.max().date()), history=hist, recommendations=rec))
            if u.path == '/api/due':
                w = int(q.get('window', 14)); d = due_now(s.cycles, s.today, w)
                d = d.assign(customer_id=d.customer_id)
                return self.send(200, dict(total=len(d), today=str(s.today.date()), rows=clean(en_df(d.head(int(q.get('limit', 200)))))))
            if u.path == '/api/repeat':
                S = s.summary
                return self.send(200, dict(overall=S['overall'], by_category=clean(en_df(S['by_category'])), by_product=clean(en_df(S['by_product'].head(30))),
                                           by_channel=clean(en_df(S['by_channel'])) if len(S['by_channel']) else []))
            return self.send(404, {'error': 'not found'})
        except Exception as e:
            traceback.print_exc(); self.send(500, {'error': str(e)})

    def do_POST(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path != '/api/upload': return self.send(404, {'error': 'not found'})
        n = int(self.headers.get('Content-Length', 0))
        if n <= 0 or n > MAX_UPLOAD: return self.send(413, {'error': 'File is empty or larger than 150 MB'})
        ext = '.xlsx' if q.get('ext') == 'xlsx' else '.csv'
        fd, path = tempfile.mkstemp(suffix=ext, prefix='nbr_'); os.close(fd)
        with open(path, 'wb') as f:
            left = n
            while left > 0:
                b = self.rfile.read(min(1 << 20, left)); f.write(b); left -= len(b)
        mapping = dict(kv.split('=', 1) for kv in q['map'].split(',') if '=' in kv) if q.get('map') else None
        jid = uuid.uuid4().hex[:8]; JOBS[jid] = dict(state='running', log=[], started=time.time())
        threading.Thread(target=run_job, args=(jid, path, q.get('name', 'uploaded'), mapping, q.get('cut'), q.get('simulated') == '1'), daemon=True).start()
        self.send(200, {'job': jid})


def main():
    global STORE
    ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--lan', action='store_true', help='listen on all network interfaces so a phone on the same Wi-Fi can connect (no login!)')
    a = ap.parse_args()
    print('Loading main dataset...'); STORE = load_default()
    host = '0.0.0.0' if a.lan else '127.0.0.1'
    print(f'Ready: http://127.0.0.1:{a.port}  (Ctrl+C to stop)')
    if a.lan:
        import socket
        ip = socket.gethostbyname(socket.gethostname())
        print(f'LAN mode: open http://{ip}:{a.port} on a phone on the same Wi-Fi. WARNING: there is no login; anyone on this network can see the data.')
    ThreadingHTTPServer((host, a.port), H).serve_forever()


if __name__ == '__main__':
    main()
