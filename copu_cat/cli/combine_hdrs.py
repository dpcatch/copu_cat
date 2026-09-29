"""
Step 5: combine the per-source HDR tables written by copu-cat-hdrs (<data-dir>/hdrs/hdrs_<Name>.feather)
into one table, <data-dir>/hdrs.feather, which the notebooks read.

Safe to run while copu-cat-hdrs is still going: it combines whatever is finished (a preview)
and can be re-run later for the full table. It also lists posteriors that have candidates but
no HDR file yet in <data-dir>/hdrs_missing.txt.

Usage:
    copu-cat-combine-hdrs [--data-dir data] [--results-dir <data-dir>/hdrs]
"""
import argparse
from pathlib import Path
import pandas as pd

from copu_cat import config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default=None, help='data directory (default: $COPU_CAT_DATA_DIR or ./data)')
    ap.add_argument('--results-dir', default=None, help='per-source HDR files (default: <data-dir>/hdrs)')
    args = ap.parse_args(argv)
    data_dir = config.get_data_dir(args.data_dir)
    results_dir = Path(args.results_dir) if args.results_dir else config.hdr_dir(data_dir)

    files = sorted(results_dir.glob('hdrs_*.feather'))
    if not files:
        raise FileNotFoundError(f"No hdrs_*.feather files in {results_dir}")
    # copu-cat-hdrs may still be writing some files. A half-written feather can't be read, so
    # skip it and pick it up in the next combine.
    tables, unreadable = {}, []
    for f in files:
        try:
            tables[f] = pd.read_feather(f)
        except Exception as err:
            unreadable.append(f.name)
    if unreadable:
        print(f"Skipped {len(unreadable)} unreadable (probably still being written) files: {unreadable[:5]}")
    files = list(tables)
    hdrs = pd.concat([t for t in tables.values() if not t.empty], ignore_index=True)

    dup = hdrs.duplicated(['Name', 'Candidate'])
    if dup.any():
        print(f"Warning: dropping {dup.sum()} duplicated (Name, Candidate) rows")
        hdrs = hdrs[~dup].reset_index(drop=True)

    # Which posteriors with at least one candidate still lack HDRs?
    counts_fp = data_dir / 'candidate_counts.csv'
    if counts_fp.exists():
        counts = pd.read_csv(counts_fp)
        expected = set(counts.loc[counts['n_candidates'] > 0, 'Name'])
        done = {f.name.removeprefix('hdrs_').removesuffix('.feather') for f in files}
        missing = sorted(expected - done)
        n_rows_expected = counts.loc[counts['Name'].isin(done), 'n_candidates'].sum()
        print(f"{len(done)} HDR files, {len(hdrs)} (posterior, candidate) rows "
              f"(expected {n_rows_expected} for these posteriors)")
        if missing:
            print(f"{len(missing)} posteriors with candidates have no HDR file yet, e.g. {missing[:5]}")
            (data_dir / 'hdrs_missing.txt').write_text('\n'.join(missing) + '\n')

    out = data_dir / 'hdrs.feather'
    out.parent.mkdir(parents=True, exist_ok=True)
    hdrs.to_feather(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
