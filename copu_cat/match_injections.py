import pandas as pd
from tqdm import tqdm
from copu_cat.utils import check_injection_match, get_galactic_binary_names
from copu_cat.galactic_binary import GalacticBinary


def get_all_matches(hdrs: pd.DataFrame, threshold: float = 0.89, names: list | None = None,
                    data_dir=None):
    """
    Iterate through all galactic binaries and check which injections have matches to posteriors based on HDR values.

    Parameters
    ----------
    names : list of posterior names to check. Default: every posterior on disk. For a preview on a
        partial `hdrs` table, pass `hdrs['Name'].unique()`. Otherwise posteriors that have no HDRs
        yet are reported as "no match".
    data_dir : data directory (see copu_cat.config).

    Returns
    -------
    dict[str, list[str]]
        Maps each posterior name to the list of matched injection names (the `Name` column of
        the injections, which is the `Candidate` column of `hdrs`). An empty list means no match.
    """
    if names is None:
        names = get_galactic_binary_names(data_dir=data_dir)
    matches = {}

    for name in tqdm(names):
        matches[name] = []
        gb = GalacticBinary.load_feather(name, data_dir=data_dir)
        for injection_index in range(len(gb.injections)):
            if check_injection_match(gb, injection_index, hdrs, threshold=threshold):
                matches[name].append(gb.injections["Name"].iloc[injection_index])
    return matches
