"""
Purity and completeness of the recovered catalog (paper, Sec. III A and III D).

purity       : per recovered source (posterior), 1 if it was assigned an injection by the
               one-to-one assignment, else 0 (a false positive, or its injections were all
               taken by other sources). Binned in the RECOVERED median frequency and catalog SNR.
completeness : per injection, 1 if it was assigned to a recovered source, else 0.
               Binned in the INJECTED frequency and SNR.
In each bin the colour is the mean of that 0/1 flag, i.e. the fraction.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from copu_cat import config
from copu_cat.config import BINARY_PARAMETERS as binary_parameters


def posterior_medians(names=None, posterior_dir=None, cache: str | Path | None = None,
                      data_dir=None) -> pd.DataFrame:
    """
    Median of every binary parameter for each posterior: one row per Name.
    posterior_dir defaults to the posterior_chains folder of the data directory.
    With `cache`, the table is read from that feather file if it exists, else computed and saved.
    """
    if posterior_dir is None:
        posterior_dir = config.posterior_dir(data_dir)
    if cache is not None and Path(cache).exists():
        med = pd.read_feather(cache)
        return med if names is None else med[med['Name'].isin(names)].reset_index(drop=True)
    posterior_dir = Path(posterior_dir)
    if names is None:
        names = sorted(f.name.removesuffix('_posterior.feather') for f in posterior_dir.glob('*_posterior.feather'))
    rows = []
    for name in tqdm(names):
        df = pd.read_feather(posterior_dir / f'{name}_posterior.feather')
        rows.append({'Name': name, **df[binary_parameters].median().to_dict()})
    med = pd.DataFrame(rows)
    if cache is not None:
        med.to_feather(cache)
    return med


def fraction_map(values: pd.DataFrame, x: str, y: str, flag: str, xedges, yedges):
    """Counts and mean of the 0/1 column `flag` in each (x, y) bin. Returns (fraction, counts)."""
    counts, _, _ = np.histogram2d(values[x], values[y], bins=[xedges, yedges])
    hits, _, _ = np.histogram2d(values[x], values[y], bins=[xedges, yedges],
                                weights=values[flag].astype(float))
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = hits / counts
    return np.ma.masked_where(counts == 0, frac), counts


def purity(post_summary: pd.DataFrame) -> float:
    return float(post_summary['assigned_injection'].notna().mean())


def completeness(inj_summary: pd.DataFrame) -> float:
    return float(inj_summary['assigned_posterior'].notna().mean())
