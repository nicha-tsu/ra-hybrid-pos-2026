"""Load POS/e-commerce order lines (CSV/XLSX), map columns to a canonical schema, build baskets."""
import unicodedata
import numpy as np, pandas as pd

ALIASES = {
    'order_id': ['order_id', 'รหัสคำสั่งซื้อ', 'เลขที่คำสั่งซื้อ', 'เลขที่บิล', 'bill_no', 'receipt_no'],
    'customer_id': ['customer_id', 'รหัสลูกค้า', 'member_id', 'เลขสมาชิก'],
    'date': ['order_date', 'date', 'วันที่สั่งซื้อ', 'วันที่'],
    'time': ['order_time', 'time', 'เวลา'],
    'sku': ['sku', 'รหัสสินค้า', 'product_code'],
    'name': ['product_name', 'ชื่อสินค้า'],
    'category': ['category', 'หมวดสินค้า'],
    'qty': ['qty', 'จำนวน'],
    'price': ['unit_price', 'ราคาต่อหน่วย'],
    'channel': ['channel', 'ช่องทางการขาย'],
    'promo': ['promo_code', 'รหัสส่วนลด', 'รหัสโค้ดส่วนลด'],
}
REQUIRED = ['order_id', 'customer_id', 'date', 'sku']


def _norm(s):
    # sara-am can be stored as U+0E4D U+0E32; fold to U+0E33 so header matching works
    return unicodedata.normalize('NFC', str(s)).replace('ํา', 'ำ').strip()


def read_table(path, sheet=None):
    if str(path).lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(path, sheet_name=sheet or 0, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str, encoding='utf-8-sig')
    df.columns = [_norm(c) for c in df.columns]
    return df


def parse_dates(s):
    d = pd.to_datetime(s, format='%Y-%m-%d', errors='coerce')
    if d.isna().mean() > 0.5:
        d = pd.to_datetime(s, dayfirst=True, errors='coerce')
    if d.notna().any() and d.dt.year.median() > 2400:      # Buddhist Era
        d = d - pd.DateOffset(years=543)
    return d


def canonical(df, mapping=None):
    """Return (lines, report). mapping: {canonical: source column}; missing keys are auto-detected."""
    mapping = {k: _norm(v) for k, v in (mapping or {}).items()}
    cols = {_norm(c).lower(): c for c in df.columns}
    for k, al in ALIASES.items():
        if k not in mapping:
            for a in al:
                if _norm(a).lower() in cols:
                    mapping[k] = cols[_norm(a).lower()]; break
    miss = [k for k in REQUIRED if k not in mapping]
    if miss:
        raise ValueError(f'ไม่พบคอลัมน์จำเป็น: {miss}. คอลัมน์ในไฟล์: {list(df.columns)}')
    out = pd.DataFrame({k: df[v] for k, v in mapping.items()})
    n0 = len(out)
    out['date'] = parse_dates(out['date'])
    for c in ('order_id', 'customer_id', 'sku'):
        out[c] = out[c].map(lambda x: None if pd.isna(x) else _norm(x))
    out = out[out.customer_id.notna() & ~out.customer_id.isin(['', 'nan', 'None'])]
    n_nocust = n0 - len(out)
    out = out[out.date.notna() & out.sku.notna() & out.order_id.notna()].copy()
    for c in ('qty', 'price'):
        out[c] = pd.to_numeric(out[c], errors='coerce') if c in out else np.nan
    if 'qty' in mapping:
        out = out[~(out.qty <= 0)]                              # drop voids/returns
    if 'time' in out:
        out['hour'] = pd.to_datetime(out['time'], format='%H:%M', errors='coerce').dt.hour.fillna(-1).astype(int)
    else:
        out['hour'] = -1
    for c in ('name', 'category', 'channel', 'promo'):
        if c not in out: out[c] = None
    rep = dict(mapping=mapping, lines_in=n0, lines_used=len(out), dropped_no_customer=n_nocust,
               has_hour=bool((out.hour >= 0).any()))
    return out.reset_index(drop=True), rep


class Corpus:
    """Baskets (one row per order) indexed by integer user/item ids."""
    def __init__(self, lines):
        self.lines = lines
        self.items = sorted(lines.sku.unique()); self.ix = {s: i for i, s in enumerate(self.items)}
        meta = lines.drop_duplicates('sku').set_index('sku')
        txt = lambda c, s: meta.at[s, c] if isinstance(meta.at[s, c], str) else ''
        self.name = {s: txt('name', s) or s for s in self.items}
        self.category = {s: txt('category', s) for s in self.items}
        l = lines.assign(ii=lines.sku.map(self.ix))
        B = l.groupby('order_id').agg(u=('customer_id', 'first'), d=('date', 'first'), h=('hour', 'first'),
                                      channel=('channel', 'first')).reset_index()
        B['items'] = B.order_id.map(l.groupby('order_id').ii.apply(lambda x: np.array(sorted(set(x)))))
        B = B.sort_values(['u', 'd', 'h', 'order_id']).reset_index(drop=True)
        self.users = sorted(B.u.unique()); self.ui = {u: i for i, u in enumerate(self.users)}
        B['ui'] = B.u.map(self.ui)
        B['k'] = B.groupby('ui').cumcount(); B['n'] = B.groupby('ui').ui.transform('size')
        self.B = B; self.U, self.I = len(self.users), len(self.items)

    def summary(self):
        B = self.B; n = B.groupby('ui').size()
        return dict(customers=self.U, products=self.I, baskets=len(B), lines=len(self.lines),
                    first=str(B.d.min().date()), last=str(B.d.max().date()),
                    customers_2plus=int((n >= 2).sum()), customers_3plus=int((n >= 3).sum()))
