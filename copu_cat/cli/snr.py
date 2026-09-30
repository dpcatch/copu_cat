"""
Step 6 (optional): compute the SNR of injections and/or recovered sources, the same way the geemoo global fit
does it (geemoo-global-fit/scripts/cut_gbcat_by_snr.py -> globalfit.gb.tools.compute_snr):

    SNR^2 = 4 df sum_f (|A|^2 + |E|^2) / <S_A>

with the JaxGB waveform and <S_A> the A-channel noise PSD of the `Backgrounds` (GeeMoo) noise
model, averaged over the frequency band of the source's waveform.

Uses the global-fit packages (globalfit, lisa_gf_noise, jaxgb, backgrounds).

Noise parameters: posterior median of the 7 sampled parameters in <data-dir>/noise_samples_0.h5, plus the
fixed instrument levels (FIXED_NOISE_PARS below). Alternatively, pass --noise-pars <noise_pars.h5>,
a noise file written by the global fit (it names its model and has all 12 parameters).

Erebor runs (--noise-model erebor): <S_A> is the A-channel PSD of the Erebor parametric noise model
(lisa_gf_noise.erebor.NoiseModelErebor, needs the `erebor` extra of lisa-gf-noise), at the median
(or max-likelihood) of the run's noise posterior. The XYZ spectral density matrix is converted to A
with the same xyz2aet matrix as the waveforms. The observation window defaults to the one of the
run's GB waveform model.

Usage:
    # 1) check against the SNRs stored in the injection catalog (writes <data-dir>/injection_snr_check.csv)
    copu-cat-snr injections --catalog gb_catalog.h5
    # 2) catalog SNR of every recovered source (writes <data-dir>/posterior_snr.csv)
    copu-cat-snr posteriors --catalog gb_catalog.h5 --n-samples 20
    # 3) Erebor run
    copu-cat-snr posteriors --catalog gb_catalog.h5 --n-samples 20 --noise-model erebor \
        --erebor-run mojito/data/erebor/CDL1run1_erebor_v4_2026-06-26T104116 \
        --erebor-orbits ../gf-noise-models/tests/NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5
"""
import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from globalfit.noise.generator import NoiseGenerator, DEF_NOISE_GRID
from globalfit.noise.model_sgwb import Backgrounds
from globalfit.gb.generator import GBGenerator
import typed_lisa_toolkit as tlt

from copu_cat import config


def tobs2fgrid(tobs, dt):
    """Frequency grid, as globalfit.tools.make_tdi.tobs2fgrid. Copied here because importing
    globalfit.tools.make_tdi also imports the MBHB code (jaxmbhb), which the SNR doesn't need."""
    return np.fft.rfftfreq(int(tobs / dt), d=dt)

logger = logging.getLogger('copu_cat.snr')

# Fixed (not sampled) instrument-noise levels: NoiseSGWBBlock default config and
# lisa_gf_noise.gee_moo.DEFAULT_CONFIG agree on these.
FIXED_NOISE_PARS = {'oms_asd': 1.5e-11, 'oms_fknee': 0.002,
                    'tm_asd': 3e-15, 'tm_fknee': 0.0004, 'tm_fbreak': 8e-3}

# Observation window of the run (lisa_gf_noise.gee_moo.DEFAULT_CONFIG and the noise metadata in
# geemoo-global-fit/scripts/geemoo_to_l3c.py).
RUN_T0, RUN_END = 97729939.827664, 121057939.827664
RUN_DURATION = RUN_END - RUN_T0


def default_orbits():
    """orbits.h5 shipped with lisa_gf_noise (what gee_moo uses by default), or None."""
    try:
        import lisa_gf_noise
        fp = Path(lisa_gf_noise.__file__).parent / 'orbits.h5'
        return str(fp) if fp.exists() else None
    except ImportError:
        return None


# copu_cat column names -> global-fit catalog column names (as in gb_catalog.h5)
TO_GF_COLUMNS = {'Frequency': 'Frequency', 'Amplitude': 'Amplitude', 'Inclination': 'Inclination',
                 'Initial Phase': 'InitialPhase', 'Dec': 'Declination', 'RA': 'RightAscension',
                 'Polarization': 'Polarization', 'Frequency Derivative': 'FrequencyDerivative'}


def load_noise(noise_samples=None, noise_pars=None, statistic='median', orbits=None,
               tdi_generation=2.0, duration=None):
    """Backgrounds noise model, either from a global-fit noise file or from noise posterior samples."""
    gen = NoiseGenerator(fgrid=DEF_NOISE_GRID, orbits=orbits, tdi_generation=tdi_generation)
    if noise_pars is not None:
        logger.info('Noise model from %s', noise_pars)
        return gen.noise_model(fn=noise_pars, duration=duration)
    with h5py.File(noise_samples, 'r') as fh:
        p = fh['0/p'][:]
        ll = fh['0/ll'][:].ravel()
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in fh['0/p'].attrs['pnames']]
    values = np.median(p, axis=0) if statistic == 'median' else p[np.argmax(ll)]
    pars = dict(FIXED_NOISE_PARS, **dict(zip(names, values)))
    missing = set(Backgrounds.PARAM_NAMES) - set(pars)
    if missing:
        raise KeyError(f'noise parameters missing: {missing}')
    logger.info('Noise model Backgrounds with %s', {k: float(v) for k, v in pars.items()})
    return gen.noise_model(model=Backgrounds, pars=pars, duration=duration)


def make_psd_interpolator(noise):
    """Evaluate the A-channel PSD once on DEF_NOISE_GRID and interpolate in log-log afterwards.
    compute_snr evaluates the full noise model for every source, which is slow for thousands of
    sources. --exact switches back to that."""
    psd_a = np.asarray(noise.psd(grid=DEF_NOISE_GRID, channel='A'))
    logf, logs = np.log(DEF_NOISE_GRID), np.log(psd_a)
    return lambda f: np.exp(np.interp(np.log(f), logf, logs))


# Erebor noise posterior column -> NoiseModelErebor parameter (in the order set_sensitivity_matrix takes
# them: Soms, Sa, spline knot positions, spline knot amplitudes, Amp, alpha, f_1, kn, f_2).
EREBOR_NOISE_PARAMS = ['S_oms', 'S_tm', None, None, 'galactic_amplitude', 'galactic_spectral_index',
                       'galactic_freq_1', 'galactic_knee_frequency', 'galactic_freq_2']


def erebor_files(run):
    """global_metadata.json, noise posterior and GB summary file of an Erebor submission folder."""
    run = Path(run)
    noise_post = sorted(run.glob('*_NOISE_posteriordir/*.h5'))
    gb = sorted(run.glob('*_GB.h5'))
    if len(noise_post) != 1 or len(gb) != 1:
        raise FileNotFoundError(f'Expected one noise posterior and one *_GB.h5 in {run}, '
                                f'found {noise_post} and {gb}')
    return run / 'global_metadata.json', noise_post[0], gb[0]


def erebor_time_window(run):
    """(t0, duration) of the GB waveform model of an Erebor run."""
    _, _, gb = erebor_files(run)
    with h5py.File(gb, 'r') as fh:
        cfg = json.loads(fh.attrs['waveform_model_config'])
    return cfg['init_kwargs']['t0'], cfg['runtime_kwargs']['T']


def load_erebor_psd(run, orbits_file, statistic='median', n_grid=2000):
    """
    A-channel PSD of the Erebor noise model of a run, as a function of frequency (log-log
    interpolation on a grid inside the model's frequency range).
    """
    from lisa_gf_noise.erebor import NoiseModelErebor
    from typed_lisa_toolkit.shop.conversions import get_xyz2aet_matrix
    metadata, noise_post, _ = erebor_files(run)
    with h5py.File(noise_post, 'r') as fh:
        post = fh['noise/posterior'][:]
    if statistic == 'median':
        point = {n: float(np.median(post[n])) for n in post.dtype.names}
    else:
        point = {n: float(post[n][np.argmax(post['loglikelihood'])]) for n in post.dtype.names}
    params = [None if n is None else point[n] for n in EREBOR_NOISE_PARAMS]
    logger.info('Erebor noise model (%s of %s): %s', statistic, noise_post.name,
                {n: point[n] for n in EREBOR_NOISE_PARAMS if n is not None})

    with open(metadata) as fh:
        dom = json.load(fh)['domain_metadata']['kwargs']
    fmin, fmax = dom['min_freq'], dom['max_freq']
    grid = np.logspace(np.log10(fmin), np.log10(fmax), n_grid)[1:-1]   # strictly inside the domain
    model = NoiseModelErebor(params, freqs=grid, times=np.array([0.0]),
                             metadata_file=str(metadata), orbits_file=str(orbits_file))
    sdm_xyz = np.asarray(model.sdm[:, 0])                      # (n_freq, 3, 3)
    m = np.asarray(get_xyz2aet_matrix(np))
    psd_a = np.einsum('i,fij,j->f', m[0], sdm_xyz, m[0]).real  # S_AA = m_A S m_A^T (m real)
    if not np.all(psd_a > 0):
        raise ValueError('Erebor A-channel PSD is not positive everywhere on the grid')
    logf, logs = np.log(grid), np.log(psd_a)

    def interp(f):
        f = np.asarray(f)
        if f.min() < grid[0] or f.max() > grid[-1]:
            raise ValueError(f'frequencies outside the Erebor noise grid [{grid[0]:.3g}, {grid[-1]:.3g}] Hz')
        return np.exp(np.interp(np.log(f), logf, logs))
    return interp


def snr_of(src, fgb, psd_a=None, noise=None):
    """Same formula as globalfit.gb.tools.compute_snr, optionally with an interpolated PSD."""
    aet = tlt.shop.xyz2aet(fgb.get_tdi_repr(src))
    freqs = np.asarray(aet.frequencies)
    if psd_a is not None:
        noise_fact = np.mean(psd_a(freqs))
    else:
        noise_fact = np.mean(noise.psd(channel='A', grid=freqs))
    A, E = aet['A'], aet['E']
    return float(np.sqrt(4 * aet.df * np.sum((A.abs() * A.abs() + E.abs() * E.abs()).entries) / noise_fact))


def circular_mean(x):
    return np.mod(np.angle(np.mean(np.exp(1j * x))), 2 * np.pi)


def posterior_catalog(template_catalog, posterior_dir, n_samples=0, seed=0):
    """
    One global-fit style catalog row per recovered source, at its posterior point estimate
    (median of each parameter, circular mean of RA). With n_samples > 0, also n_samples random
    posterior draws per source (to get the median SNR over the posterior, as in the paper).
    Rows are tagged with Name and draw (-1 = point estimate).
    """
    template = pd.read_hdf(template_catalog, 'cat').iloc[:0]
    rng = np.random.default_rng(seed)
    rows = []
    files = sorted(Path(posterior_dir).glob('*_posterior.feather'))
    for fp in tqdm(files, desc='posteriors'):
        name = fp.name.removesuffix('_posterior.feather')
        df = pd.read_feather(fp)
        point = df.median()
        point['RA'] = circular_mean(df['RA'].to_numpy())
        draws = [(-1, point)]
        if n_samples:
            idx = rng.choice(len(df), min(n_samples, len(df)), replace=False)
            draws += [(k, df.iloc[i]) for k, i in enumerate(idx)]
        for k, d in draws:
            row = {TO_GF_COLUMNS[c]: float(d[c]) for c in TO_GF_COLUMNS}
            row.update(Name=name, draw=k)
            rows.append(row)
    cat = pd.DataFrame(rows)
    # fill the remaining catalog columns so GBGenerator sees the usual format
    for col in template.columns:
        if col not in cat:
            cat[col] = {'snr': 0.0, 'bayes_factor': 1.0, 'confidence': 0.0}.get(col, '')
    cat['ID'] = cat['Name'] + '_' + cat['draw'].astype(str)
    return cat, list(template.columns)


def to_records(cat, columns):
    """Record array with exactly the catalog columns, as GBGenerator.load_catalog returns it."""
    c = cat[columns].to_records(index=True)
    _dt = [d if d[0] != 'origin' else ('origin', '<U200') for d in c.dtype.descr]
    _dt = [d if d[0] != 'ID' else ('ID', '<U64') for d in _dt]
    return c.astype(_dt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('which', choices=['injections', 'posteriors'])
    ap.add_argument('--catalog', required=True,
                    help='injection catalog in global-fit format (e.g. gb_catalog.h5); '
                         'also the column template for posteriors')
    ap.add_argument('--data-dir', default=None, help='data directory (default: $COPU_CAT_DATA_DIR or ./data)')
    ap.add_argument('--noise-samples', default=None, help='default: <data-dir>/noise_samples_0.h5')
    ap.add_argument('--noise-pars', default=None, help='global-fit noise_pars.h5 (overrides --noise-samples)')
    ap.add_argument('--noise-statistic', choices=['median', 'maxll'], default='median')
    ap.add_argument('--noise-model', choices=['geemoo', 'erebor'], default='geemoo',
                    help='geemoo: Backgrounds model from --noise-samples/--noise-pars; '
                         'erebor: Erebor parametric model of --erebor-run')
    ap.add_argument('--erebor-run', default=None,
                    help='Erebor submission folder (global_metadata.json, *_NOISE_posteriordir/, *_GB.h5)')
    ap.add_argument('--erebor-orbits', default=None,
                    help='orbits file for the Erebor noise model (any Mojito L1 data file)')
    ap.add_argument('--t0', type=float, default=None,
                    help='start of the observation (s); default: gee_moo DEFAULT_CONFIG, or the '
                         'GB waveform model of --erebor-run')
    ap.add_argument('--duration', type=float, default=None,
                    help='observation time in s (default: the run window of gee_moo DEFAULT_CONFIG, '
                         '~270 d, or of the GB waveform model of --erebor-run; '
                         'cut_gbcat_by_snr.py uses 1 year = 31535995)')
    ap.add_argument('--tdi-generation', type=float, default=2.0)
    ap.add_argument('--orbits', default='default',
                    help="orbits file; 'default' = the orbits.h5 shipped with lisa_gf_noise, "
                         "'equal' = equal-arm orbits")
    ap.add_argument('--tbin', type=float, default=5, help='time bin in s (cut_gbcat_by_snr.py default: 5)')
    ap.add_argument('--bsize', type=int, default=512, help='waveform buffer size (cut_gbcat_by_snr.py: 512)')
    ap.add_argument('--n-samples', type=int, default=0,
                    help='posteriors only: also compute the SNR of this many random draws per source')
    ap.add_argument('--exact', action='store_true', help='evaluate the noise PSD per source (slow)')
    ap.add_argument('--limit', type=int, default=None, help='only the first N sources (for testing)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)
    data_dir = config.get_data_dir(args.data_dir)
    if args.noise_samples is None:
        args.noise_samples = data_dir / 'noise_samples_0.h5'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    if args.noise_model == 'erebor':
        if args.erebor_run is None or args.erebor_orbits is None:
            ap.error('--noise-model erebor needs --erebor-run and --erebor-orbits')
        if args.exact:
            ap.error('--exact is only available for the geemoo noise model')
        t0, duration = erebor_time_window(args.erebor_run)
    else:
        t0, duration = RUN_T0, RUN_DURATION
    args.t0 = t0 if args.t0 is None else args.t0
    args.duration = duration if args.duration is None else args.duration

    orbits = {'default': default_orbits(), 'equal': None}.get(args.orbits, args.orbits)
    logger.info('orbits: %s, TDI generation %s, t0 %s, duration %s s',
                orbits or 'equal-arm', args.tdi_generation, args.t0, args.duration)
    if args.noise_model == 'erebor':
        noise = None
        psd_a = load_erebor_psd(args.erebor_run, args.erebor_orbits, args.noise_statistic)
    else:
        noise = load_noise(args.noise_samples, args.noise_pars, args.noise_statistic, orbits=orbits,
                           tdi_generation=args.tdi_generation, duration=[args.t0, args.t0 + args.duration])
        psd_a = None if args.exact else make_psd_interpolator(noise)
    # like cut_gbcat_by_snr.py, but that script omits tdi_generation, which the current
    # globalfit Generator requires
    fgb = GBGenerator(dt=args.tbin, fgrid=tobs2fgrid(args.duration, args.tbin), itime=args.t0,
                      orbits=orbits, tdi_generation=args.tdi_generation).get_fast_waveform(args.bsize)

    if args.which == 'injections':
        cat = GBGenerator.load_catalog(args.catalog)
        if args.limit:
            cat = cat[:args.limit]
    else:
        table, columns = posterior_catalog(args.catalog, config.posterior_dir(data_dir), args.n_samples)
        if args.limit:
            table = table[table['Name'].isin(table['Name'].unique()[:args.limit])]
        cat = to_records(table, columns)

    snr = np.array([snr_of(src, fgb, psd_a=psd_a, noise=noise) for src in tqdm(cat, desc='SNR')])

    if args.which == 'injections':
        out = pd.DataFrame({'Name': cat['ID'].astype(str), 'SNR_catalog': cat['snr'], 'SNR': snr})
        ratio = out['SNR'] / out['SNR_catalog']
        print(f'recomputed / catalog SNR: median {ratio.median():.4f}, '
              f'16-84% [{ratio.quantile(.16):.4f}, {ratio.quantile(.84):.4f}]')
        out_fp = args.out or data_dir / 'injection_snr_check.csv'
    else:
        full = pd.DataFrame({'Name': table['Name'].to_numpy(), 'draw': table['draw'].to_numpy(), 'SNR': snr})
        out = full[full['draw'] == -1][['Name', 'SNR']].reset_index(drop=True)
        if args.n_samples:
            med = full[full['draw'] >= 0].groupby('Name')['SNR'].median().rename('SNR_median')
            out = out.join(med, on='Name')
            full.to_csv(data_dir / 'posterior_snr_draws.csv', index=False)
        out_fp = args.out or data_dir / 'posterior_snr.csv'
    out.to_csv(out_fp, index=False)
    print(f'Wrote {len(out)} rows to {out_fp}')


if __name__ == '__main__':
    main()
