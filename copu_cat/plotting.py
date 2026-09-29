"""Plots for the injection / posterior matching results."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copu_cat.assignment import STATUS_ORDER

# Categorical slots 1-3 of the reference palette (validated for colour-vision deficiency).
STATUS_COLORS = dict(zip(STATUS_ORDER, ['#2a78d6', '#eb6834', '#1baf7a']))
_INK, _MUTED, _GRID, _SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb'


def _style(ax):
    ax.set_facecolor(_SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#c3c2b7')
    ax.tick_params(colors=_MUTED, labelcolor=_INK, labelsize=9)
    ax.grid(axis='y', color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def match_counts_by_bin(summary: pd.DataFrame, column: str, bins: np.ndarray) -> pd.DataFrame:
    """Number of injections per status in each bin of `column` (rows: bins, columns: status)."""
    cut = pd.cut(summary[column], bins, include_lowest=True)
    return summary.groupby([cut, 'status'], observed=False).size().unstack('status')[STATUS_ORDER]


def plot_matches_vs_params(summary: pd.DataFrame, n_bins: int = 30, title: str | None = None):
    """
    Stacked histograms of injections by match status, against frequency (mHz), amplitude and
    log10(SNR). `summary` is the output of assignment.injection_summary.
    Returns (fig, {panel: counts table}). The tables hold the numbers behind each panel.
    """
    df = summary.assign(**{'Frequency (mHz)': summary['Frequency'] * 1e3,
                           'log10(SNR)': np.log10(summary['SNR'])})
    panels = [('Frequency (mHz)', 'log'), ('Amplitude', 'log'), ('log10(SNR)', 'linear')]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True, facecolor=_SURFACE)
    tables = {}
    for ax, (col, scale) in zip(axes, panels):
        x = df[col].to_numpy()
        lo, hi = np.nanmin(x), np.nanmax(x)
        bins = np.geomspace(lo, hi, n_bins + 1) if scale == 'log' else np.linspace(lo, hi, n_bins + 1)
        counts = match_counts_by_bin(df, col, bins)
        tables[col] = counts
        bottom = np.zeros(n_bins)
        for status in STATUS_ORDER:
            h = counts[status].to_numpy()
            # thin surface-coloured edge = 2px-style gap between stacked segments and bars
            ax.bar(bins[:-1], h, width=np.diff(bins), bottom=bottom, align='edge',
                   color=STATUS_COLORS[status], edgecolor=_SURFACE, linewidth=0.8, label=status)
            bottom += h
        ax.set_xscale(scale)
        ax.set_xlabel(col, color=_INK)
        _style(ax)
    axes[0].set_ylabel('Number of injections', color=_INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.99))
    if title:
        fig.suptitle(title, y=1.04, fontsize=11, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig, tables


def _with_derived_columns(summary: pd.DataFrame) -> pd.DataFrame:
    derived = {'Frequency (mHz)': ('Frequency', lambda v: v * 1e3),
               'log10(SNR)': ('SNR', np.log10),
               'log10(Amplitude)': ('Amplitude', np.log10)}
    return summary.assign(**{k: f(summary[col]) for k, (col, f) in derived.items() if col in summary})


def plot_match_hist2d(summary: pd.DataFrame, x: str = 'Frequency (mHz)', y: str = 'log10(SNR)',
                      statuses: list | None = None, bins=(60, 60), xlim=None, ylim=None,
                      cmap: str = 'viridis'):
    """
    2D histograms of injection counts, one panel per match status, on a shared log colour scale.

    x, y     : any of 'Frequency (mHz)', 'log10(SNR)', 'log10(Amplitude)', or a column of `summary`.
    statuses : which panels to draw (default: all of STATUS_ORDER). Use e.g. [None] for one panel
               with every injection.
    bins     : number of bins (nx, ny). The bin edges are shared by all panels.
    xlim, ylim : axis ranges (default: data range).
    Returns (fig, {status: (counts, xedges, yedges)}).
    """
    from matplotlib.colors import LogNorm

    df = _with_derived_columns(summary)
    statuses = list(STATUS_ORDER) if statuses is None else statuses
    xr = xlim or (df[x].min(), df[x].max())
    yr = ylim or (df[y].min(), df[y].max())
    xedges = np.linspace(*xr, bins[0] + 1)
    yedges = np.linspace(*yr, bins[1] + 1)

    hists = {}
    for s in statuses:
        sel = df if s is None else df[df['status'] == s]
        hists[s] = np.histogram2d(sel[x], sel[y], bins=[xedges, yedges])[0]
    vmax = max(max(h.max() for h in hists.values()), 1)
    norm = LogNorm(vmin=1, vmax=vmax)

    fig, axes = plt.subplots(1, len(statuses), figsize=(6.2 * len(statuses) + 1.2, 5),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    for ax, s in zip(axes, statuses):
        h = np.ma.masked_less(hists[s], 1)  # empty bins stay white
        mesh = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=norm)
        ax.set_xlim(xr); ax.set_ylim(yr)
        ax.set_xlabel(x)
        ax.set_title(('all injections' if s is None else s) + f'  (N = {int(hists[s].sum())})', fontsize=11)
    axes[0].set_ylabel(y)
    fig.colorbar(mesh, ax=list(axes), label='Count', fraction=0.03 if len(axes) > 1 else 0.05, pad=0.02)
    return fig, {s: (hists[s], xedges, yedges) for s in statuses}


def plot_fraction_map(values: pd.DataFrame, flag: str, label: str,
                      x: str = 'Frequency (mHz)', y: str = 'log10(SNR)',
                      bins=(60, 60), xlim=None, ylim=None, hist: dict | None = None,
                      cmap: str = 'viridis', ax_title: str | None = None):
    """
    Map of the fraction of `flag` (0/1 column) per (x, y) bin, as in the paper's purity (Fig. 1)
    and completeness (Fig. 6) plots. Colour runs from 0 to 1; empty bins stay white.

    values : table with Frequency, SNR (and Amplitude) columns plus `flag`. Derived columns
             'Frequency (mHz)', 'log10(SNR)', 'log10(Amplitude)' are added automatically.
    hist   : optional {legend label: array of frequencies in mHz} drawn as step histograms in a
             panel under the map (the paper's "Catalog" / "Injections" panel).
    Returns (fig, (fraction, counts, xedges, yedges)).
    """
    from copu_cat.catalog_metrics import fraction_map

    df = _with_derived_columns(values)
    xr = xlim or (df[x].min(), df[x].max())
    yr = ylim or (df[y].min(), df[y].max())
    xedges = np.linspace(*xr, bins[0] + 1)
    yedges = np.linspace(*yr, bins[1] + 1)
    frac, counts = fraction_map(df, x, y, flag, xedges, yedges)

    if hist:
        fig, (ax, axh) = plt.subplots(2, 1, figsize=(6.5, 6.2), sharex=True,
                                      gridspec_kw={'height_ratios': [3.2, 1], 'hspace': 0.05})
    else:
        fig, ax = plt.subplots(figsize=(6.5, 5))
    mesh = ax.pcolormesh(xedges, yedges, frac.T, cmap=cmap, vmin=0, vmax=1)
    ax.set_xlim(xr); ax.set_ylim(yr)
    ax.set_ylabel(y)
    if ax_title:
        ax.set_title(ax_title, fontsize=11)
    cbar_axes = [ax, axh] if hist else [ax]
    fig.colorbar(mesh, ax=cbar_axes, label=label, pad=0.02, fraction=0.05)
    if hist:
        for (name, freqs), color in zip(hist.items(), ['#2a78d6', '#eb6834', '#1baf7a']):
            axh.hist(np.asarray(freqs), bins=xedges, histtype='step', linewidth=1.5, color=color, label=name)
        axh.set_ylabel('N')
        axh.set_xlabel(x)
        axh.legend(frameon=False, fontsize=9)
    else:
        ax.set_xlabel(x)
    return fig, (frac, counts, xedges, yedges)
