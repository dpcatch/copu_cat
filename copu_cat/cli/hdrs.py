"""
Step 4: compute the 1D and 6D HDRs (and optionally the 8D HDR) of every candidate injection of
every recovered source.

The 6D HDR (all parameters except Initial Phase and Polarization) is the one used for matching. The
8D HDR needs a second flow fit per source, roughly doubling the run time, and is not used downstream,
so it is off by default: `--with-8d` turns it on. Without it the '8D HDR' column is NaN, so the files
keep the same columns either way.

Output: <data-dir>/hdrs/hdrs_<Name>.feather, one per source (sources without candidates are
skipped). Re-running skips sources that already have a file, so an interrupted run can be resumed,
and disjoint ranges (--start/--stop) can run in parallel.

Usage:
    copu-cat-hdrs                      # all sources
    copu-cat-hdrs --start 0 --stop 100
    copu-cat-hdrs --with-8d            # also the 8D HDR
"""
import pandas as pd
from copu_cat.utils import get_galactic_binary_names
from copu_cat.galactic_binary import GalacticBinary
from copu_cat.highest_density_region import compute_1d_hdrs, compute_8d_hdr_all_injections
from copu_cat.highest_density_region import compute_6d_hdr_all_injections
import argparse
from pathlib import Path
from tqdm import tqdm

from copu_cat import config

def get_hdrs(gb_index: int, plot_dir: str|None = None, names: list|None = None,
             results_dir: Path|None = None, data_dir=None, with_8d: bool = False) -> pd.DataFrame:
    print(f'Computing HDRs for GB index {gb_index}')
    if names is None:
        names = get_galactic_binary_names(data_dir=data_dir)
    if results_dir is None:
        results_dir = config.hdr_dir(data_dir)
    hdr_data = {'Name': [], 'Candidate': [], 'SNR': [], '8D HDR': [],
                '6D HDR': [], 'Frequency HDR': [],
                'Amplitude HDR': [], 'Inclination HDR': [],
                'Initial Phase HDR': [], 'Dec HDR': [],
                'RA HDR': [], 'Polarization HDR': [],
                'Frequency Derivative HDR': []}
    gb = GalacticBinary.load_feather(names[gb_index], data_dir=data_dir)
    for i in range(gb.injections.shape[0]):
        hdrs = compute_1d_hdrs(gb, i)
        hdr_data['Frequency HDR'].append(hdrs['Frequency'])
        hdr_data['Amplitude HDR'].append(hdrs['Amplitude'])
        hdr_data['Inclination HDR'].append(hdrs['Inclination'])
        hdr_data['Initial Phase HDR'].append(hdrs['Initial Phase'])
        hdr_data['Dec HDR'].append(hdrs['Dec'])
        hdr_data['RA HDR'].append(hdrs['RA'])
        hdr_data['Polarization HDR'].append(hdrs['Polarization'])
        hdr_data['Frequency Derivative HDR'].append(hdrs['Frequency Derivative'])
        hdr_data['Name'].append(gb.name)
        hdr_data['Candidate'].append(gb.injections.iloc[i]['Name'])
        hdr_data['SNR'].append(gb.injections.iloc[i]['SNR'])
    if with_8d:
        hdr_8d = compute_8d_hdr_all_injections(gb, plot_dir=plot_dir)
    else:
        hdr_8d = [float('nan')] * gb.injections.shape[0]
    for i in range(gb.injections.shape[0]):
        hdr_data['8D HDR'].append(hdr_8d[i])
    hdr_6d = compute_6d_hdr_all_injections(gb, plot_dir=plot_dir)
    for i in range(gb.injections.shape[0]):
        hdr_data['6D HDR'].append(hdr_6d[i])
    hdrs = pd.DataFrame(hdr_data)
    out = Path(results_dir) / f'hdrs_{gb.name}.feather'
    print(f'Saving HDRs to {out}')
    hdrs.to_feather(out)
    return hdrs


def main(argv=None):
    ap = argparse.ArgumentParser(description='Compute 1D and 6D (optionally 8D) HDRs for every posterior and its candidate injections.')
    ap.add_argument('--start', type=int, default=0, help='first source index (inclusive)')
    ap.add_argument('--stop', type=int, default=None, help='last source index (exclusive); default: all')
    ap.add_argument('--data-dir', default=None, help='data directory (default: $COPU_CAT_DATA_DIR or ./data)')
    ap.add_argument('--results-dir', default=None, help='where to write the HDR files (default: <data-dir>/hdrs)')
    ap.add_argument('--plot-dir', default=None, help='save corner plots here (off by default)')
    ap.add_argument('--overwrite', action='store_true', help='recompute sources that already have an output file')
    ap.add_argument('--with-8d', action='store_true',
                    help='also compute the 8D HDR (a second flow fit per source; not used downstream)')
    args = ap.parse_args(argv)

    data_dir = config.get_data_dir(args.data_dir)
    results_dir = Path(args.results_dir) if args.results_dir else config.hdr_dir(data_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    names = get_galactic_binary_names(data_dir=data_dir)
    if not names:
        raise FileNotFoundError(f'No posteriors found in {config.posterior_dir(data_dir)}')
    indices = range(args.start, len(names) if args.stop is None else min(args.stop, len(names)))

    skipped_empty, skipped_done, failed = [], [], []
    for index in tqdm(indices):
        name = names[index]
        if not args.overwrite and (results_dir / f'hdrs_{name}.feather').exists():
            skipped_done.append(name)
            continue
        gb = GalacticBinary.load_feather(name, data_dir=data_dir)
        if gb.injections.empty:
            skipped_empty.append(name)
            continue
        try:
            get_hdrs(index, plot_dir=args.plot_dir, names=names, results_dir=results_dir, data_dir=data_dir,
                     with_8d=args.with_8d)
        except Exception as err:  # keep going; report at the end
            print(f'FAILED {name}: {err!r}')
            failed.append(name)

    print(f'Done. Skipped {len(skipped_done)} already computed, {len(skipped_empty)} with no candidates, '
          f'{len(failed)} failed.')
    if failed:
        (results_dir / 'failed.txt').write_text('\n'.join(failed) + '\n')
        print(f'Failed sources listed in {results_dir / "failed.txt"}')


if __name__ == "__main__":
    main()
