from pathlib import Path
import numpy as np
import pandas as pd
from copu_cat import config
from copu_cat.galactic_binary import GalacticBinary


def get_galactic_binary_names(path=None, data_dir=None) -> list:
    """
    Names of all recovered sources: the <Name> of every <Name>_posterior.feather in `path`
    (default: the posterior_chains folder of the data directory).
    """
    path = Path(path) if path is not None else config.posterior_dir(data_dir)
    return sorted(f.name.removesuffix('_posterior.feather') for f in path.glob('*_posterior.feather'))

# 1D HDRs used to decide a match. Initial Phase and Polarization are left out for now: the
# waveform is unchanged under (Polarization + pi/2, Initial Phase + pi), so a correct injection
# can sit in the "other copy" of the posterior and get a 1D HDR close to 1 for those two angles.
# They can be added back once that symmetry is folded out.
MATCH_HDR_PARAMS = ['Frequency HDR', 'Amplitude HDR', 'Inclination HDR',
                    'Dec HDR', 'RA HDR', 'Frequency Derivative HDR']


def check_injection_match(galactic_binary: GalacticBinary, injection_index: int, hdr_dataframe: pd.DataFrame, threshold:float = 0.9):
    # threshold is the maximum allowed HDR (1 - alpha)%
    alpha = 1.0 - threshold
    # correct for multiple 1D trials (one per parameter in MATCH_HDR_PARAMS)
    hdr_params = MATCH_HDR_PARAMS
    threshold = multiple_trial_correction(alpha, ntrials=len(hdr_params))
    name = galactic_binary.name
    injection_name = galactic_binary.injections["Name"].iloc[injection_index]
    masked_df = hdr_dataframe[(hdr_dataframe["Name"] == name) & (hdr_dataframe["Candidate"] == injection_name)][hdr_params]
    if masked_df.empty is True:
        print(name, injection_name)
        return False
    if np.all(masked_df.iloc[0].to_numpy() < threshold):
        return True
    else:
        return False

def multiple_trial_correction(alpha: float, ntrials: int = 8):
    """
    Apply multiple trial (Bonferroni) correction to a given alpha value.

    Parameters:
    ----------
    alpha (float): The original alpha value.
    ndims (int): The number of dimensions (default is 8).

    Returns:
    -------
    float: The corrected alpha value.
    """
    return 1 - (alpha / ntrials)
