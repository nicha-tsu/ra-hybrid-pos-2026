"""Production-side recommender: fit on all history with the selected method, serve top-K per customer."""
import numpy as np, pandas as pd
from .models import State, SCORERS
from .repeat import repeat_cycles


class Recommender:
    def __init__(self, corpus, method, params):
        self.c, self.method, self.params = corpus, method, params
        self.s = State(corpus.B, corpus.U, corpus.I)
        self.cycles = repeat_cycles(corpus).set_index(['customer_id', 'sku']) if len(corpus.B) else None
        self.last = corpus.lines.groupby(['customer_id', 'sku']).date.max()

    def _scores(self, ui, hour):
        h = [hour if hour is not None else -1]
        return SCORERS[self.method](self.s, np.array([ui]), np.array(h), **self.params)[0]

    def recommend(self, customer_id, k=5, explore_slots=0, hour=None):
        """Top-k SKUs. explore_slots reserves the last slots for items the customer has never bought
        (trades a little HR@k for new-product trial; the paper measured 1.4% -> 5.4% new-item hits at 1 of 3 slots)."""
        c = self.c; known = customer_id in c.ui
        if known:
            ui = c.ui[customer_id]; sc = self._scores(ui, hour); hist = self.s.X[ui] > 0
        else:                                                    # cold start: popularity
            sc = SCORERS['Popularity'](self.s, np.array([0]), None)[0]; hist = np.zeros(c.I, bool)
        order = [i for i in np.argsort(-sc) if sc[i] > 0 or not known][:max(k * 3, 30)]
        main = order[:k - explore_slots]
        new = [i for i in np.argsort(-np.where(hist, -np.inf, sc)) if not hist[i] and i not in main][:explore_slots]
        out = []
        for rank, i in enumerate(main + new, 1):
            sku = c.items[i]; r = dict(rank=rank, sku=sku, name=c.name[sku], category=c.category[sku], score=float(sc[i]))
            if hist[i]:
                r['type'] = 'ซื้อซ้ำ'; r['times_bought'] = int(self.s.X[ui, i]); r['last_bought'] = str(self.last[(customer_id, sku)].date())
                if self.cycles is not None and (customer_id, sku) in self.cycles.index:
                    cy = self.cycles.loc[(customer_id, sku)]
                    r['median_gap_days'] = float(cy.median_gap_days); r['next_due'] = str(cy.next_due.date())
            else:
                r['type'] = 'สินค้าใหม่สำหรับลูกค้า' if known else 'ยอดนิยม (ลูกค้าใหม่)'
            out.append(r)
        return out
