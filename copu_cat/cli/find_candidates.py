"""
Step 3: for each posterior, keep the injections whose frequency lies inside the posterior's
[min, max] frequency range.

Output: <data-dir>/injection_matches/<Name>_injections.feather (one per posterior; may be empty),
plus <data-dir>/candidate_counts.csv, the number of candidates per posterior.

Usage:
    copu-cat-find-candidates [--snr-cut 1]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from copu_cat import config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default=None, help='data directory (default: $COPU_CAT_DATA_DIR or ./data)')
    ap.add_argument('--snr-cut', type=float, default=1.0, help='drop injections with SNR below this')
    args = ap.parse_args(argv)
    data_dir = config.get_data_dir(args.data_dir)

    inj = pd.read_feather(data_dir / 'injection_set' / 'injections.feather')
    inj = inj[inj['SNR'] >= args.snr_cut].sort_values('Frequency').reset_index(drop=True)
    f_inj = inj['Frequency'].to_numpy()

    out_dir = data_dir / 'injection_matches'
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = []
    for fp in tqdm(sorted((data_dir / 'posterior_chains').glob('*_posterior.feather'))):
        name = fp.name.removesuffix('_posterior.feather')
        f = pd.read_feather(fp, columns=['Frequency'])['Frequency']
        lo = np.searchsorted(f_inj, f.min(), side='left')
        hi = np.searchsorted(f_inj, f.max(), side='right')
        cand = inj.iloc[lo:hi].reset_index(drop=True)
        cand.to_feather(out_dir / f"{name}_injections.feather")
        counts.append((name, len(cand)))
    counts = pd.DataFrame(counts, columns=['Name', 'n_candidates'])
    counts.to_csv(data_dir / 'candidate_counts.csv', index=False)
    print(counts['n_candidates'].value_counts().sort_index().rename('n_posteriors').to_string())


if __name__ == '__main__':
    main()
