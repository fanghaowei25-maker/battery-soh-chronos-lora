"""Recompute the exploratory initial-capacity SOH metrics from saved predictions.

This is a post-hoc sensitivity analysis, not a fresh 17-cell retraining run.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'sit_v2_discharge.npz'
PRED = ROOT / 'results' / 'ensemble_predictions'
EXCLUDED = {'001-3', '001-8', '101-3'}
INCOMPLETE = {('003-5', 1), ('003-5', 18), ('101-1', 699)}

def main():
    d = np.load(DATA, allow_pickle=True)
    init = dict(zip(d['c_init_cells'].astype(str), d['c_init'].astype(float)))
    rows = []
    for path in sorted(PRED.glob('ensemble_cell*.csv')):
        f = pd.read_csv(path)
        cell = str(f['cell_id'].iloc[0])
        if cell in EXCLUDED:
            continue
        for r in f.itertuples(index=False):
            cycle = int(r.cycle)
            if cycle <= 5 or (cell, cycle) in INCOMPLETE:
                continue
            idx = int(r.sample_index)
            if str(d['cell_id'][idx]) != cell or int(d['cycle'][idx]) != cycle:
                raise ValueError(f'prediction/data alignment failed: {cell} cycle {cycle}')
            truth = float(d['q_smooth'][idx]) / init[cell]
            if truth < 0.7:
                continue
            pred = float(r.soh_pred_ensemble) * 50.0 / init[cell]
            rows.append((cell, cycle, abs(truth-pred)*100))
    out = pd.DataFrame(rows, columns=['cell_id','cycle','abs_error_pp'])
    pooled = out['abs_error_pp'].mean()
    macro = out.groupby('cell_id')['abs_error_pp'].mean().mean()
    print(f'cells={out.cell_id.nunique()} cycles={len(out)} pooled_MAE_pp={pooled:.6f} macro_MAE_pp={macro:.6f}')
    assert out.cell_id.nunique() == 17 and len(out) == 11302
    assert abs(pooled - 2.974251) < 0.0001 and abs(macro - 2.999158) < 0.0001
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
