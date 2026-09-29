"""
Corner plots for ambiguous matches:

corner_injection_vs_posteriors  one injection and every posterior it is matched to
corner_posterior_vs_injections  one posterior and every injection it is matched to

In both, the pair chosen by the one-to-one assignment is highlighted.
`match_table` is the output of assignment.assign_one_to_one (columns Name, Candidate,
matched, assigned, 6D HDR, ...); `injections` has one row per injection (Name + parameters).
Posterior samples are read from the posterior_chains folder of `data_dir` (see copu_cat.config).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import corner

from copu_cat import config
# the six parameters used for matching (phase and polarization are left out)
MATCH_PARAMS = ['Frequency', 'Amplitude', 'Inclination', 'Dec', 'RA', 'Frequency Derivative']
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7', '#008300', '#e34948']


def _display(values, params, f_ref):
    """Readable axes: frequency as offset from f_ref in microHz, log10 amplitude.
    Returns (array, labels). values: (n, len(params))."""
    v = np.array(values, dtype=float, ndmin=2).copy()
    labels = list(params)
    for j, p in enumerate(params):
        if p == 'Frequency':
            v[:, j] = (v[:, j] - f_ref) * 1e6
            labels[j] = f'f - {f_ref * 1e3:.5f} mHz [µHz]'
        elif p == 'Amplitude':
            v[:, j] = np.log10(v[:, j])
            labels[j] = 'log10 Amplitude'
        elif p == 'Frequency Derivative':
            labels[j] = 'Frequency Derivative [Hz/s]'
    return v, labels


def _load_posterior(name, data_dir):
    return pd.read_feather(config.posterior_dir(data_dir) / f'{name}_posterior.feather')


def _ranges(arrays, points, pad=0.05):
    """Common axis range covering all sample arrays and all marked points."""
    data = np.vstack([np.vstack(arrays)] + ([np.atleast_2d(points)] if len(points) else []))
    lo, hi = np.nanmin(data, axis=0), np.nanmax(data, axis=0)
    width = np.where(hi > lo, hi - lo, np.abs(hi) * 1e-3 + 1e-30)
    return list(zip(lo - pad * width, hi + pad * width))


def _corner_kwargs():
    return dict(plot_datapoints=False, plot_density=False, fill_contours=False, smooth=1.0,
                levels=(0.68, 0.95), bins=30, hist_kwargs={'density': True, 'linewidth': 1.2},
                labelpad=0.15, max_n_ticks=4)


def corner_injection_vs_posteriors(injection_name, match_table, injections, params=MATCH_PARAMS,
                                   data_dir=None, posterior_snr=None,
                                   max_posteriors=len(COLORS)):
    """
    One injection (black lines) against every posterior it is matched to (one colour each).
    The posterior it is assigned to is drawn with solid thick lines; the others dashed.
    posterior_snr : optional Series (index posterior Name) shown in the legend.
    """
    rows = match_table[(match_table['Candidate'] == injection_name) & match_table['matched']]
    rows = rows.sort_values('6D HDR').head(max_posteriors)
    if rows.empty:
        raise ValueError(f'injection {injection_name} is not matched to any posterior')
    inj = injections.set_index('Name').loc[injection_name]
    f_ref = float(inj['Frequency'])
    truth, labels = _display(inj[params].to_numpy(dtype=float), params, f_ref)
    truth = truth[0]
    chains = {n: _display(_load_posterior(n, data_dir)[params].to_numpy(), params, f_ref)[0]
              for n in rows['Name']}
    rng = _ranges(list(chains.values()), truth)

    fig, handles = None, []
    for color, (_, r) in zip(COLORS, rows.iterrows()):
        assigned = bool(r['assigned'])
        kw = _corner_kwargs()
        kw['contour_kwargs'] = {'linestyles': 'solid' if assigned else 'dashed',
                                'linewidths': 2.0 if assigned else 1.0}
        kw['hist_kwargs'] = dict(kw['hist_kwargs'], linestyle='solid' if assigned else 'dashed',
                                 linewidth=2.0 if assigned else 1.0)
        fig = corner.corner(chains[r['Name']], fig=fig, color=color, range=rng, labels=labels, **kw)
        snr = f", SNR {posterior_snr[r['Name']]:.1f}" if posterior_snr is not None else ''
        label = f"{r['Name']}: 6D HDR {r['6D HDR']:.3f}{snr}" + ('  <- assigned' if assigned else '')
        handles.append(Line2D([], [], color=color, lw=2.0 if assigned else 1.0,
                              ls='-' if assigned else '--', label=label))
    corner.overplot_lines(fig, truth, color='k', lw=1.2)
    corner.overplot_points(fig, truth[None, :], marker='*', color='k', ms=10)
    handles.append(Line2D([], [], color='k', marker='*', label=f'injection {injection_name} (SNR {inj["SNR"]:.1f})'))
    fig.legend(handles=handles, loc='upper right', frameon=False, fontsize=10)
    n_total = int(((match_table['Candidate'] == injection_name) & match_table['matched']).sum())
    fig.suptitle(f'Injection {injection_name}: matched to {n_total} posteriors', y=1.01)
    return fig


def corner_posterior_vs_injections(posterior_name, match_table, injections, params=MATCH_PARAMS,
                                   data_dir=None, max_injections=len(COLORS)):
    """
    One posterior (grey contours) against every injection it is matched to (one coloured star
    each). The injection assigned to it is also drawn as solid lines across the panels.
    """
    rows = match_table[(match_table['Name'] == posterior_name) & match_table['matched']]
    rows = rows.sort_values('6D HDR').head(max_injections)
    if rows.empty:
        raise ValueError(f'posterior {posterior_name} is not matched to any injection')
    inj = injections.set_index('Name')
    raw_chain = _load_posterior(posterior_name, data_dir)[params].to_numpy()
    f_ref = float(np.median(raw_chain[:, params.index('Frequency')])) if 'Frequency' in params else 0.0
    points, labels = _display(inj.loc[rows['Candidate'], params].to_numpy(dtype=float), params, f_ref)
    chain = _display(raw_chain, params, f_ref)[0]
    rng = _ranges([chain], points)

    fig = corner.corner(chain, color='0.35', range=rng, labels=labels, **_corner_kwargs())
    handles = [Line2D([], [], color='0.35', label=f'posterior {posterior_name}')]
    for color, (_, r), pt in zip(COLORS, rows.iterrows(), points):
        assigned = bool(r['assigned'])
        if assigned:
            corner.overplot_lines(fig, pt, color=color, lw=1.5)
        corner.overplot_points(fig, pt[None, :], marker='*' if assigned else 'o', color=color,
                               ms=12 if assigned else 7)
        label = (f"injection {r['Candidate']}: 6D HDR {r['6D HDR']:.3f}, SNR {r['SNR']:.1f}"
                 + ('  <- assigned' if assigned else ''))
        handles.append(Line2D([], [], color=color, ls='none', marker='*' if assigned else 'o', label=label))
    n_total = int(((match_table['Name'] == posterior_name) & match_table['matched']).sum())
    fig.legend(handles=handles, loc='upper right', frameon=False, fontsize=10)
    fig.suptitle(f'Posterior {posterior_name}: matched to {n_total} injections', y=1.01)
    return fig
