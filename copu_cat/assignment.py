"""
Match table and one-to-one assignment between posteriors (catalog entries) and injections.

build_match_table   one row per (posterior, candidate) pair with a `matched` flag.
                      criterion='1d': every 1D HDR in MATCH_HDR_PARAMS below the Bonferroni-
                                      corrected threshold (same rule as utils.check_injection_match)
                      criterion='6d': 6D HDR below threshold, i.e. the injection lies inside the
                                      `threshold` credible region (the paper's rule, Sec. III A)
assign_one_to_one   picks matched pairs so that every posterior and every injection is used at
                    most once. Two methods:
                      method='min_hdr': as many pairs as possible and, among those, the lowest
                                        total 6D HDR (exact, per connected group).
                      method='max_snr': the paper's greedy algorithm (Sec. III A). Go through the
                                        posteriors from highest to lowest catalog SNR; give each
                                        one the highest-SNR injection still available among its
                                        matches; remove that injection from the pool.
                    All other matched pairs are kept with assigned=False.
injection_summary   one row per injection: status (assigned / matched, not assigned / not
                    matched) and how many posteriors it matches.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from copu_cat.utils import MATCH_HDR_PARAMS, multiple_trial_correction

STATUS_ORDER = ['assigned (one-to-one)', 'matched, not assigned', 'not matched']


def build_match_table(hdrs: pd.DataFrame, threshold: float = 0.99, criterion: str = '1d',
                      hdr_params: list = MATCH_HDR_PARAMS) -> pd.DataFrame:
    """Add a `matched` column to a copy of `hdrs`. See the module docstring for `criterion`."""
    table = hdrs.copy()
    if criterion == '1d':
        cut = multiple_trial_correction(1.0 - threshold, ntrials=len(hdr_params))
        table['matched'] = (table[hdr_params] < cut).all(axis=1)
    elif criterion == '6d':
        table['matched'] = table['6D HDR'] < threshold
    else:
        raise ValueError("criterion must be '1d' or '6d'")
    return table


def _assign_min_hdr(table: pd.DataFrame, cost: str) -> np.ndarray:
    """Boolean array over table rows: maximum-cardinality, minimum-total-cost assignment."""
    assigned = np.zeros(len(table), dtype=bool)
    m = np.flatnonzero(table['matched'].to_numpy())
    if not len(m):
        return assigned
    post_codes, post_names = pd.factorize(table['Name'].to_numpy()[m])
    inj_codes, inj_names = pd.factorize(table['Candidate'].to_numpy()[m])
    n_p, n_i = len(post_names), len(inj_names)
    # Posteriors and injections form a bipartite graph. Pairs only link nearby frequencies,
    # so it splits into many small connected pieces, each solved exactly on its own.
    graph = coo_matrix((np.ones(len(m)), (post_codes, n_p + inj_codes)), shape=(n_p + n_i,) * 2)
    _, labels = connected_components(graph, directed=False)
    comp = labels[post_codes]
    costs = table[cost].to_numpy(dtype=float)[m]
    order = np.argsort(comp, kind='stable')
    for edges in np.split(order, np.flatnonzero(np.diff(comp[order])) + 1):
        p_loc, p_idx = np.unique(post_codes[edges], return_inverse=True)
        i_loc, i_idx = np.unique(inj_codes[edges], return_inverse=True)
        # Non-pairs cost more than any possible sum of real costs (HDR <= 1), so the solver
        # first maximizes the number of real pairs, then minimizes the total cost.
        big = len(edges) + 1.0
        mat = np.full((len(p_loc), len(i_loc)), big)
        mat[p_idx, i_idx] = costs[edges]
        rows, cols = linear_sum_assignment(mat)
        keep = mat[rows, cols] < big
        chosen = set(zip(p_loc[rows[keep]], i_loc[cols[keep]]))
        for e in edges:
            if (post_codes[e], inj_codes[e]) in chosen:
                assigned[m[e]] = True
    return assigned


def _assign_max_snr(table: pd.DataFrame, posterior_snr: pd.Series) -> np.ndarray:
    """Boolean array over table rows: the paper's greedy highest-SNR-first assignment."""
    missing = set(table.loc[table['matched'], 'Name']) - set(posterior_snr.index)
    if missing:
        raise KeyError(f"posterior_snr has no value for {len(missing)} posteriors, e.g. {sorted(missing)[:3]}")
    assigned = np.zeros(len(table), dtype=bool)
    matched = table[table['matched']].assign(_row=np.flatnonzero(table['matched'].to_numpy()))
    # within each posterior, try candidates from highest to lowest injection SNR
    matched = matched.sort_values('SNR', ascending=False)
    groups = dict(tuple(matched.groupby('Name', sort=False)))
    used = set()
    for name in posterior_snr.loc[list(groups)].sort_values(ascending=False).index:
        for cand, row in zip(groups[name]['Candidate'], groups[name]['_row']):
            if cand not in used:
                used.add(cand)
                assigned[row] = True
                break
    return assigned


def assign_one_to_one(table: pd.DataFrame, method: str = 'min_hdr', cost: str = '6D HDR',
                      posterior_snr: pd.Series | None = None) -> pd.DataFrame:
    """
    Add columns to a copy of `table` (output of build_match_table):
      assigned               - pair is in the one-to-one assignment
      n_injections_matched   - number of matched injections of this pair's posterior
      n_posteriors_matched   - number of matched posteriors of this pair's injection

    method        : 'min_hdr' (lowest total `cost`) or 'max_snr' (paper's greedy algorithm).
    posterior_snr : for 'max_snr', catalog SNR of each posterior (Series indexed by posterior
                    Name). This sets the order in which posteriors pick their injection. The
                    injection SNR comes from the table's `SNR` column.
    """
    table = table.copy()
    if method == 'min_hdr':
        table['assigned'] = _assign_min_hdr(table, cost)
    elif method == 'max_snr':
        if posterior_snr is None:
            raise ValueError("method='max_snr' needs posterior_snr (catalog SNR per posterior Name)")
        table['assigned'] = _assign_max_snr(table, posterior_snr)
    else:
        raise ValueError("method must be 'min_hdr' or 'max_snr'")

    matched = table[table['matched']]
    table['n_injections_matched'] = table['Name'].map(matched.groupby('Name').size()).fillna(0).astype(int)
    table['n_posteriors_matched'] = table['Candidate'].map(matched.groupby('Candidate').size()).fillna(0).astype(int)
    return table


def injection_summary(table: pd.DataFrame, injections: pd.DataFrame) -> pd.DataFrame:
    """
    One row per injection in `injections` (columns Name, Frequency, Amplitude, SNR, ...), with
    `status` (see STATUS_ORDER), `n_posteriors_matched` and, if assigned, `assigned_posterior`.
    """
    inj = injections.copy()
    matched = table[table['matched']]
    assigned = table[table['assigned']].set_index('Candidate')['Name']
    inj['n_posteriors_matched'] = inj['Name'].map(matched.groupby('Candidate').size()).fillna(0).astype(int)
    inj['assigned_posterior'] = inj['Name'].map(assigned)
    inj['status'] = np.where(inj['assigned_posterior'].notna(), STATUS_ORDER[0],
                             np.where(inj['n_posteriors_matched'] > 0, STATUS_ORDER[1], STATUS_ORDER[2]))
    inj['status'] = pd.Categorical(inj['status'], categories=STATUS_ORDER, ordered=True)
    return inj


def posterior_summary(table: pd.DataFrame, posteriors: pd.DataFrame) -> pd.DataFrame:
    """
    One row per posterior in `posteriors` (column Name plus e.g. median Frequency, Amplitude, SNR),
    with `assigned_injection` (NaN if none) and `n_injections_matched`.
    """
    post = posteriors.copy()
    matched = table[table['matched']]
    assigned = table[table['assigned']].set_index('Name')['Candidate']
    post['n_injections_matched'] = post['Name'].map(matched.groupby('Name').size()).fillna(0).astype(int)
    post['assigned_injection'] = post['Name'].map(assigned)
    return post
