# copu_cat

Match simulated (injected) galactic binaries to the sources recovered by a LISA global fit, without
knowing in advance which is which, and measure the **purity** and **completeness** of the catalog.

For every recovered source (a posterior from the global fit) and every injection close to it in
frequency, copu_cat computes where the injection falls inside the posterior (highest-density-region
level, "HDR", in 1D and 6D/8D). An injection *matches* a posterior when it lies inside its credible
region. A one-to-one assignment then pairs each posterior with at most one injection, either
greedily by SNR (as in the paper draft) or by the lowest total 6D HDR.

## Installation

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/dpcatch/copu_cat.git
cd copu_cat
uv sync                 # creates .venv with all dependencies, including the global-fit packages
source .venv/bin/activate
```

The global-fit packages (`lisa-global-fit`, `backgrounds`, and `lisa-gf-noise`, which comes with
`lisa-global-fit`) are installed from GitLab.

## Inputs

1. **Injection catalog**: a pandas HDF5 table (key `cat`) in global-fit catalog format, with columns
   `Frequency, Amplitude, FrequencyDerivative, RightAscension, Declination, Inclination,
   Polarization, InitialPhase, snr, ID`, e.g. `gb_catalog.h5`.
2. **Posterior samples**: a `.npy` array of shape `(n_sources, n_samples, 8)`, e.g. `all_gb_samples.npy`.
   The order of the last axis is set with `--param-order` (default:
   `Frequency, Frequency Derivative, Dec, RA, Amplitude, Inclination, Polarization, Initial Phase`).
3. For the catalog SNR (optional): the global-fit noise posterior, `noise_samples_0.h5`, copied into the data directory.

All outputs go to one **data directory**: `./data` by default, or `$COPU_CAT_DATA_DIR`, or
`--data-dir` on any command. Its layout is described in `copu_cat/config.py`.

## Pipeline

Each step is an installed command; `python scripts/<name>.py` does the same.

| step | command | script | output (in the data directory) |
|---|---|---|---|
| 1 | `copu-cat-convert-injections gb_catalog.h5` | `scripts/convert_injections.py` | `injection_set/injections.feather` |
| 2 | `copu-cat-convert-posteriors all_gb_samples.npy` | `scripts/convert_posteriors.py` | `posterior_chains/GF_<i>_posterior.feather` |
| 3 | `copu-cat-find-candidates` | `scripts/find_candidates.py` | `injection_matches/`, `candidate_counts.csv` |
| 4 | `copu-cat-hdrs` | `scripts/hdrs.py` | `hdrs/hdrs_<Name>.feather` |
| 5 | `copu-cat-combine-hdrs` | `scripts/combine_hdrs.py` | `hdrs.feather` |
| 6 | `copu-cat-snr posteriors --catalog gb_catalog.h5 --n-samples 20` | `scripts/snr.py` | `posterior_snr.csv` (optional) |

Notes:

- **Step 4** is the slow one. It can be resumed (existing files are skipped) and split into
  independent ranges that run in parallel, e.g. `copu-cat-hdrs --start 0 --stop 1000`.
  Step 5 can be run at any time for a preview.
- **Step 6** computes the SNR of each recovered source with the global-fit waveform and noise model.
  It is needed for the greedy max-SNR assignment and for binning purity in catalog SNR.
  `copu-cat-snr injections --catalog gb_catalog.h5` recomputes the injection SNRs as a check
  against the catalog's own `snr`.

Then run the notebooks, in order:

1. `notebooks/match_injections_with_hdrs.ipynb`: matching, one-to-one assignment, match-status plots, purity and completeness.
2. `notebooks/ambiguous_matches.ipynb`: corner plots of injections matched to several posteriors and of posteriors matched to several injections.

Set `DATA_DIR` and `PLOTS_DIR` in the first code cell of each notebook.
