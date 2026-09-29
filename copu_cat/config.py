"""
Where copu_cat reads and writes its files.

Everything lives under one data directory. Its default is ``./data`` (relative to where you run
a command), or the ``COPU_CAT_DATA_DIR`` environment variable if set; every command takes
``--data-dir`` and every function that touches files takes ``data_dir``.

Layout of the data directory::

    injection_set/injections.feather        converted injection catalog   (copu-cat-convert-injections)
    posterior_chains/<Name>_posterior.feather   one file per recovered source (copu-cat-convert-posteriors)
    posterior_index.csv                     source index -> Name
    injection_matches/<Name>_injections.feather candidate injections per source (copu-cat-find-candidates)
    candidate_counts.csv
    hdrs/hdrs_<Name>.feather                HDRs per source                (copu-cat-hdrs)
    hdrs.feather                            all HDRs combined              (copu-cat-combine-hdrs)
    posterior_snr.csv                       catalog SNR per source          (copu-cat-snr)
    noise_samples_0.h5                      global-fit noise posterior (input to copu-cat-snr)
    matches_*.feather, *_summary_*.feather, posterior_medians.feather   written by the notebooks
"""
import os
from pathlib import Path

# Parameter names used throughout the package. The sky angles are RA/Dec, the frame of the
# injection catalog; the posterior samples are in the same frame.
BINARY_PARAMETERS = ['Frequency', 'Amplitude', 'Inclination', 'Initial Phase',
                     'Dec', 'RA', 'Polarization', 'Frequency Derivative']


def get_data_dir(data_dir=None) -> Path:
    """`data_dir` if given, else $COPU_CAT_DATA_DIR, else ./data."""
    if data_dir is not None:
        return Path(data_dir)
    return Path(os.environ.get('COPU_CAT_DATA_DIR', 'data'))


def posterior_dir(data_dir=None) -> Path:
    return get_data_dir(data_dir) / 'posterior_chains'


def injection_matches_dir(data_dir=None) -> Path:
    return get_data_dir(data_dir) / 'injection_matches'


def hdr_dir(data_dir=None) -> Path:
    return get_data_dir(data_dir) / 'hdrs'
