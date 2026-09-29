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

Usage:
    # 1) check against the SNRs stored in the injection catalog (writes <data-dir>/injection_snr_check.csv)
    copu-cat-snr injections --catalog gb_catalog.h5
    # 2) catalog SNR of every recovered source (writes <data-dir>/posterior_snr.csv)
    copu-cat-snr posteriors --catalog gb_catalog.h5 --n-samples 20
"""
import argparse
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
    ap.add_argument('--t0', type=float, default=RUN_T0, help='start of the observation (s)')
    ap.add_argument('--duration', type=float, default=RUN_DURATION,
                    help='observation time in s (default: the run window of gee_moo DEFAULT_CONFIG, '
                         '~270 d; cut_gbcat_by_snr.py uses 1 year = 31535995)')
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

    orbits = {'default': default_orbits(), 'equal': None}.get(args.orbits, args.orbits)
    logger.info('orbits: %s, TDI generation %s, t0 %s, duration %s s',
                orbits or 'equal-arm', args.tdi_generation, args.t0, args.duration)
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
