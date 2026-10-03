"""Repeat-purchase discovery: who buys what again, how often, and who is due for a reorder."""
import numpy as np, pandas as pd


def repeat_cycles(c, min_repeats=2):
    """Per (customer, SKU) with >= min_repeats purchases: median inter-purchase days and next expected date."""
    l = c.lines.drop_duplicates(['order_id', 'sku'])[['customer_id', 'sku', 'date']].sort_values('date')
    rows = []
    for (u, s), g in l.groupby(['customer_id', 'sku']):
        d = g.date.drop_duplicates().sort_values()
        if len(d) < min_repeats: continue
        gaps = d.diff().dt.days.dropna().values
        med = float(np.median(gaps)); rows.append(dict(customer_id=u, sku=s, name=c.name[s], category=c.category[s],
            purchases=len(d), median_gap_days=med, last_purchase=d.iloc[-1], next_due=d.iloc[-1] + pd.Timedelta(days=med)))
    return pd.DataFrame(rows)


def due_now(cycles, today=None, window=14):
    """Customer-SKU pairs whose expected reorder date is within +-window days of today (overdue_days > 0 = late)."""
    if cycles.empty: return cycles
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    x = cycles.assign(overdue_days=(today - cycles.next_due).dt.days)
    return x[x.overdue_days.abs() <= window].sort_values('overdue_days', ascending=False)


def repeat_summary(c):
    """Repeat rate overall, by product, by category, by channel (repeat = bought by same customer before)."""
    B = c.B; seen = {}; rows = []
    for ui, d, its, ch in zip(B.ui.values, B.d.values, B['items'].values, B.channel.values):
        s = seen.setdefault(ui, set())
        for i in its: rows.append((ui, i, i in s, ch))
        s.update(its.tolist())
    R = pd.DataFrame(rows, columns=['ui', 'ii', 'repeat', 'channel'])
    R = R[R.ui.map(B.groupby('ui').size()) >= 2]
    first = B.groupby('ui').head(1).index                     # first basket of a customer cannot be a repeat
    prod = R.groupby('ii').agg(lines=('repeat', 'size'), repeat_rate=('repeat', 'mean')).reset_index()
    prod['sku'] = prod.ii.map(lambda i: c.items[i]); prod['name'] = prod.sku.map(c.name); prod['category'] = prod.sku.map(c.category)
    cat = prod.assign(r=prod.lines * prod.repeat_rate).groupby('category').agg(lines=('lines', 'sum'), r=('r', 'sum'))
    cat['repeat_rate'] = cat.r / cat.lines
    ch = R.groupby('channel').repeat.agg(lines='size', repeat_rate='mean') if R.channel.notna().any() else pd.DataFrame()
    return dict(overall=float(R['repeat'].mean()) if len(R) else float('nan'),
                by_product=prod.sort_values('lines', ascending=False)[['sku', 'name', 'category', 'lines', 'repeat_rate']],
                by_category=cat[['lines', 'repeat_rate']].reset_index().sort_values('lines', ascending=False),
                by_channel=ch.reset_index() if len(ch) else ch)
