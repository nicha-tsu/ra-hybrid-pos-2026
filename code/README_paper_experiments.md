# RA-Hybrid: data and code (anonymised for double-blind review)

POS receipts from a beverage and snack shop at a university in southern Thailand, with **simulated** customer IDs, plus the code to reproduce the scale-up experiments in Section 3.4.

> **Customer IDs are simulated.** The POS export holds no customer identity. Every `customer_id` was assigned by `code/simulate.py` and **does not represent real customer behaviour**. Baskets, items, dates, times, quantities and prices are real.

## What was removed or replaced

| Field | Treatment |
|---|---|
| Basket/receipt number | Replaced with a sequential `order_id` (`B000001`…), in the original order. The POS receipt numbers are not released. |
| Product code | The shop-specific prefix was replaced with a neutral `P-<category>-<nnn>`. |
| Branch | Column removed. |
| Source file name | Column removed. |
| Customer name, phone, invoice no. | Not included (all empty in the POS export). |
| Cash-drawer code, table/queue no. | Not included. |
| Simulated segment label | Uses the neutral term `นักศึกษา` (students). |

The owning organisation permitted the data and code to be published.

## Files

`data/`
- `orders.csv`: 65,614 lines, 44,268 baskets, 347 products (Aug 2022 – Jan 2024). `row_seq` is the original row order, which the simulator needs for exact reproduction. `customer_id` is the main simulated set; `customer_id_simulated` = True.
- `products.csv`: product list with `sku`, normalised name, category and typical price.
- `customers_simulated.csv`: summary of each simulated customer under the main set.
- `simulated_customer_assignments.csv`: basket → simulated customer for the main set (`set=main`, `item_w=0.6`, `seed=2569`) and the 20 sensitivity sets (`item_w` ∈ {0, 0.3, 0.6, 1.0} × seeds 1–5).

`code/`
- `simulate.py`: customer-ID simulator. `SEED=2569 ITEM_W=0.6` reproduces the main set exactly.
- `exp2.py`: data loading, splits (global temporal split, leave-last-basket-out), models, metrics.
- `run2.py`: tunes hyperparameters on validation and runs every method → `v2_results.pkl` (Table 2).
- `stats2.py`: bootstrap CIs, Wilcoxon-Pratt, t-test, Holm correction.
- `extra2.py`: robustness to alternative cutoffs (Table 3) and the explore-weight curve.
- `ana3.py`: repeat/explore metrics and user-group analysis (Tables 4–5).
- `sens_eval.py`: simulator sensitivity analysis (first-round version).
- `exp3.py`, `run3.py`: revised experiments used in the current manuscript. Every method, including the Personal baseline, is tuned on validation. They add TIFU-KNN and UP-CF@r, RA-Hybrid ablations, NDCG@5/MRR, ex-ante user groups, the 2+1 slot test, cold-start users, timing, alternative cutoffs and the 20 simulator sets. They produce Tables 2–6 and Figs. 1–4 → `run3.pkl`.
- `simulate_alt.py`: alternative simulator added in round 3. Customers prefer product **categories** (2 favourite categories out of 15) instead of individual products; everything else matches `simulate.py`. IDs are again **simulated**.
- `run4.py`: round-3 additions → `run4_<mode>.pkl`. Modes: `tost` (equivalence test, δ = 0.01 NDCG@10), `abl` (RA-Hybrid ablations on the 20 sensitivity sets), `alt` (15 alternative-simulator sets), `perm` (customer IDs shuffled within each month, 20 permutations), `seedtune` (per-seed re-tuning at item_w = 0.6). Needs `run3.pkl` in the same folder.

## Reproduce

```bash
pip install pandas numpy scipy
cd code
SEED=2569 ITEM_W=0.6 OUT=assign_main.csv python simulate.py   # optional: regenerates set=main
python run3.py   # current manuscript (about 10 min); run2.py etc. reproduce the first-round tables
mkdir -p alt; for w in 0.6 1.0 2.0; do for s in 1 2 3 4 5; do SEED=$s ITEM_W=$w OUT=alt/alt_w${w}_s${s}.csv python simulate_alt.py; done; done
for m in tost abl alt perm seedtune; do python run4.py $m; done   # Table 7, TOST, permutation control, per-seed tuning
```

Verified before release: `simulate.py` reproduces all 44,268 main-set assignments exactly. `run3.py` on these files reproduces every number in the current Tables 2–6 and Figs. 1–4 (e.g., NDCG@10: RA-Hybrid 0.2989, UP-CF@r 0.2975, tuned Personal 0.2974, Popularity 0.2092; 1,392 test users). Running all of `run2.py` takes about 3–4 minutes on a laptop CPU.
