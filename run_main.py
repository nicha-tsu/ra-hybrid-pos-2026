"""Run the full pipeline on the released POS data (customer IDs are SIMULATED).
usage: python run_main.py     ->  results/pos/{evaluation.xlsx, repeat_discovery.xlsx, recommendations.xlsx, dashboard.html, model.json}"""
import os, sys, shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
OUT = os.path.join('results', 'pos'); os.makedirs(OUT, exist_ok=True)
csv = os.path.join(OUT, 'orders_main.csv'); shutil.copy(os.path.join('data', 'orders.csv'), csv)
model = os.path.join(OUT, 'model.json')
nbr = lambda *a: subprocess.run([sys.executable, '-m', 'nbr', *a], check=True, env=dict(os.environ, PYTHONIOENCODING='utf-8'))
nbr('evaluate', csv, '--cut', '2023-08-01', '--val-cut', '2023-05-01', '--out', OUT)
nbr('repeat', csv, '--out', OUT, '--today', '2024-02-01')
nbr('recommend', csv, '--model', model, '-k', '5', '--explore-slots', '1', '--out', OUT)
nbr('dashboard', csv, '--model', model, '--out', OUT, '--top', '300')
