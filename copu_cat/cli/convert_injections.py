"""
Step 1: convert the injection catalog (pandas HDF, key 'cat') to the feather format that
copu_cat expects.

Output: <data-dir>/injection_set/injections.feather with columns
    BINARY_PARAMETERS + ['Name', 'SNR', 'catalog_index']

Usage:
    copu-cat-convert-injections gb_catalog.h5
"""
import argparse
from pathlib import Path
import pandas as pd
from copu_cat import config
from copu_cat.config import BINARY_PARAMETERS
from copu_cat.inputs import wrap_angles

COLUMN_MAP = {
    'Frequency': 'Frequency',
    'Amplitude': 'Amplitude',
    'Inclination': 'Inclination',
    'InitialPhase': 'Initial Phase',
    'Declination': 'Dec',
    'RightAscension': 'RA',
    'Polarization': 'Polarization',
    'FrequencyDerivative': 'Frequency Derivative',
    'ID': 'Name',
    'snr': 'SNR',
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('catalog', help='injection catalog, e.g. gb_catalog.h5')
    ap.add_argument('--key', default='cat', help='HDF key of the catalog table')
    ap.add_argument('--data-dir', default=None, help='data directory (default: $COPU_CAT_DATA_DIR or ./data)')
    args = ap.parse_args(argv)

    cat = pd.read_hdf(args.catalog, args.key)
    missing = set(COLUMN_MAP) - set(cat.columns)
    if missing:
        raise KeyError(f"Catalog is missing columns {missing}; edit COLUMN_MAP.")
    inj = cat[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
    inj['Name'] = inj['Name'].astype(str)
    if not inj['Name'].is_unique:
        raise ValueError("Injection IDs are not unique: matching needs a unique Name per injection.")
    inj['catalog_index'] = cat.index.to_numpy()
    inj = wrap_angles(inj)[BINARY_PARAMETERS + ['Name', 'SNR', 'catalog_index']].reset_index(drop=True)

    out = config.get_data_dir(args.data_dir) / 'injection_set' / 'injections.feather'
    out.parent.mkdir(parents=True, exist_ok=True)
    inj.to_feather(out)
    print(f"Wrote {len(inj)} injections to {out}")


if __name__ == '__main__':
    main()
