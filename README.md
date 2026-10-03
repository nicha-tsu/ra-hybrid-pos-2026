# RA-Hybrid on POS data: code, data and demo app (anonymised for double-blind review)

POS receipts from a beverage and snack shop, with **simulated** customer IDs, plus (1) the code that reproduces the experiments in the paper and (2) a small application that applies the selected method (recommendations, repeat-purchase discovery).

> **Customer IDs are simulated.** The POS export holds no customer identity. Every `customer_id` was assigned by `code/simulate.py` and does not represent real customer behaviour. Baskets, items, dates, times, quantities and prices are real. The owning organisation permitted the data and code to be published.

## Layout
| Path | Content |
|---|---|
| `data/` | `orders.csv` (65,614 lines, 44,268 baskets, 347 products, Aug 2022 – Jan 2024), `products.csv`, simulated customer files |
| `code/` | Scripts that reproduce the paper's tables and figures; see `code/README_paper_experiments.md` |
| `nbr/` | Library: data loading, models (Popularity, tuned Personal, TIFU-KNN, UP-CF@r, RA-Hybrid), validation-based selection, repeat-purchase discovery, recommender |
| `app/` | Local web app (English, responsive/mobile layout) |
| `run_main.py` | One-command pipeline on `data/orders.csv` |

## Reproduce
```bash
pip install -r requirements.txt
python run_main.py                      # about 2-3 minutes on a laptop CPU
```
Outputs go to `results/pos/`: `evaluation.xlsx` (NDCG@10/@5, MRR@10, HR@K, repeat/new-item recall, 95% CI, sample-size figures), `repeat_discovery.xlsx`, `recommendations.xlsx`, `dashboard.html`, `model.json`.

Expected NDCG@10 on the 1,392 test customers (cut 2023-08-01): Popularity 0.209, Personal 0.297, TIFU-KNN 0.289, UP-CF@r 0.297, RA-Hybrid 0.299; the pipeline keeps the tuned Personal baseline because no method beats it on validation with statistical support.

Paper experiments (round-3 tables, simulator sensitivity): see `code/README_paper_experiments.md`.

## Run the app
```bash
python app/server.py --port 8080        # then open http://127.0.0.1:8080
```
Listens on localhost only. `--lan` exposes it on the local network (no login; do not use on untrusted networks). A phone-sized layout is used automatically on narrow screens.

## Using your own data
`python -m nbr summary <file.csv|xlsx>` checks the columns (customer ID, date, order number, product code are required; Thai or English headers are detected, or use `--map sku=product_code,...`). Then `python -m nbr evaluate <file> --out out`.
