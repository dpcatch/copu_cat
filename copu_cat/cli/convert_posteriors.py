"""
Step 2: split the global-fit samples (n_sources x n_samples x n_parameters .npy) into one
feather file per recovered source, as GalacticBinary.load_feather() expects.

Output:
    <data-dir>/posterior_chains/GF_<i>_posterior.feather   (one per source, columns BINARY_PARAMETERS)
    <data-dir>/posterior_index.csv                         (source index i -> Name)

The parameter order along the last axis is not stored in the .npy file. The default below was
inferred by comparing posterior medians with gb_catalog.h5, and the two angles
(Polarization vs Initial Phase) are the least certain. Check it and override it with
--param-order if it's wrong.

Usage:
    copu-cat-convert-posteriors all_gb_samples.npy
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from copu_cat import config
from copu_cat.config import BINARY_PARAMETERS
from copu_cat.inputs import load_samples, source_name, wrap_angles

DEFAULT_ORDER = ['Frequency', 'Frequency Derivative', 'Dec', 'RA',
                 'Amplitude', 'Inclination', 'Polarization', 'Initial Phase']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('samples', help='e.g. all_gb_samples.npy')
    ap.add_argument('--param-order', default=','.join(DEFAULT_ORDER),
                    help='comma-separated column names for the last axis')
    ap.add_argument('--data-dir', default=None, help='data directory (default: $COPU_CAT_DATA_DIR or ./data)')
    args = ap.parse_args(argv)

    order = [p.strip() for p in args.param_order.split(',')]
    if sorted(order) != sorted(BINARY_PARAMETERS):
        raise ValueError(f"--param-order must be a permutation of {BINARY_PARAMETERS}")

    samples = load_samples(args.samples)
    if samples.ndim != 3 or samples.shape[2] != len(order):
        raise ValueError(f"Expected (n_sources, n_samples, {len(order)}), got {samples.shape}")

    out_dir = config.get_data_dir(args.data_dir) / 'posterior_chains'
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i in tqdm(range(samples.shape[0])):
        df = pd.DataFrame(samples[i], columns=order)
        df = wrap_angles(df)[BINARY_PARAMETERS]
        name = source_name(i)
        df.to_feather(out_dir / f"{name}_posterior.feather")
        names.append(name)
    pd.DataFrame({'source_index': range(len(names)), 'Name': names}).to_csv(
        config.get_data_dir(args.data_dir) / 'posterior_index.csv', index=False)
    print(f"Wrote {len(names)} posteriors to {out_dir}")


if __name__ == '__main__':
    main()
