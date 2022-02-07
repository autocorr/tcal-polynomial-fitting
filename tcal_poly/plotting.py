#!/usr/bin/env python3

import warnings
from copy import copy

import numpy as np
from pandas import IndexSlice as idx
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from astropy import units as u

from . import PATHS
from .core import SedPoly


# Matplotlib configuration settings
plt.rc("text", usetex=True)
plt.rc("font", size=10, family="serif")
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.ioff()

CMAP = copy(plt.cm.get_cmap("magma"))
CMAP.set_bad("0.5", 1.0)

FSCALE_FIELDS = ["0521+166=3C138", "0137+331=3C48", "0542+498=3C147"]
WEATHER_COLS = ["api", "tsurf", "tdew", "pwv", "wind"]
WEATHER_YLABELS = [
        r"API [deg]",
        r"$T_\mathrm{surf}$ [$^\circ\mathrm{C}$]",
        r"$T_\mathrm{dew}$ [$^\circ\mathrm{C}$]",
        r"PWV [mm]",
        r"Wind Speed [m/s]",
]


def savefig(outname, dpi=300, relative=True, overwrite=True):
    outpath = PATHS.plot / outname if relative else Path(outname)
    if outpath.exists() and not overwrite:
        print(f"Figure exists, continuing: {outpath}")
    else:
        for ext in ("pdf", "png"):
            plt.savefig(str(outpath.with_suffix(f".{ext}")), dpi=dpi)
        print(f"Figure saved to: {outpath}")
    plt.close("all")


def plot_sed(date_df, field, date, outname=None):
    df = date_df
    assert len(df) > 0
    freq = df.freq.values * u.Hz.to("GHz")
    log_freq = np.log10(freq)
    if outname is None:
        outname = f"{field}_{date}_sed"
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(4, 6))
    axes[0].set_title(f"{field}; {date}")
    for stokes, ax in zip(STOKES, axes):
        poly = SedPoly.from_stokes(df, stokes, use_weights=True)
        poly.clip_and_refit(sigma=5.0)
        poly.plot_data(ax)
        poly.plot_poly(ax)
        ax.set_ylabel(rf"$I_\nu(\mathcal{{ {stokes} }}) \ [\mathrm{{Jy}}]$")
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(r"$\log_{10}(f \ [\mathrm{GHz}])$")
    plt.tight_layout()
    savefig(f"{outname}")


def plot_relative_i_sed(date_df, field, date, scaled_data=True, outname=None):
    df = date_df
    assert len(df) > 0
    pb = PerleyButler17Poly(field) if not scaled_data else None
    poly = SedPoly.from_stokes(date_df, "I", pb_poly=pb, use_weights=True)
    poly.clip_and_refit(sigma=5.0)
    if outname is None:
        outname = f"{field}_{date}_relative_sed"
    fig, axes = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(4, 4.0))
    axes[0].set_title(f"{field.replace('_', '')}; {date}")
    # With all/bad points
    ax = axes[0]
    poly.plot_poly(ax)
    poly.plot_data(ax, show_clipped=True)
    ax.hlines(1, poly.log_freq.min(), poly.log_freq.max(), color="cyan",
            linestyle="dashed")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylabel(r"$I_\nu(\mathcal{I}) / I_{\nu;\mathrm{PB17}}$")
    # Just good points
    ax = axes[1]
    poly.plot_poly(ax)
    poly.plot_data(ax, show_clipped=False)
    ax.hlines(1, poly.log_freq.min(), poly.log_freq.max(), color="cyan",
            linestyle="dashed")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylabel(r"$I_\nu(\mathcal{I}) / I_{\nu;\mathrm{PB17}}$")
    ax.set_xlabel(r"$\log_{10}(f \ [\mathrm{GHz}])$")
    plt.tight_layout()
    savefig(f"{outname}")


def plot_all_seds(full_df, field_names=None, do_abs=True, do_rel=True,
        abs_kwargs=None, rel_kwargs=None):
    rel_kwargs = {} if rel_kwargs is None else rel_kwargs
    abs_kwargs = {} if abs_kwargs is None else abs_kwargs
    if field_names is None:
        field_names = full_df.index.get_level_values("field").unique()
    for field in field_names:
        field_df = full_df.xs(field, level="field")
        all_dates = field_df.index.get_level_values("date").unique()
        for date in all_dates:
            date_df = field_df.xs(date, level="date")
            try:
                if do_abs:
                    plot_sed(date_df, field, date, **abs_kwargs)
                if do_rel:
                    plot_relative_i_sed(date_df, field, date, **rel_kwargs)
            except np.linalg.LinAlgError:
                continue


def plot_all_seds_rel(full_df):
    fields = ["0137+331=3C48", "0521+166=3C138", "0542+498=3C147"]
    plot_all_seds(full_df, field_names=fields, do_abs=False,
            rel_kwargs=dict(scaled_data=True))


def get_light_curve_data(field_df, band, pb_poly=None, filter_dates=None):
    if filter_dates is None:
        filter_dates = []
    try:
        dates = field_df.xs(band, level="band").index.get_level_values("date").unique()
    except KeyError:
        raise ValueError(f"Field does not contain {band} Band.")
    band_center = {
            "L": 1.5,
            "S": 3.0,
            "C": 6.0,
            "X": 10.0,
            "U": 15.0,
            "K": 22.3,
            "A": 33.3,
            "Q": 45.0,
    }
    band_freqs = np.log10(np.array([band_center[band]]))
    dates.sort_values()
    field_df = field_df.loc[idx[dates, :, :]]
    dates = [d for d in dates if d not in filter_dates]
    light_curve = []
    for date in dates:
        date_df = field_df.xs(date, level="date")
        poly = SedPoly.from_stokes(date_df, "I", pb_poly=pb_poly, use_weights=True)
        poly.clip_and_refit(sigma=5.0)
        fluxes = np.polyval(poly.poly_coef, band_freqs)
        light_curve.append(fluxes)
    return dates, light_curve


def overplot_light_curve(ax, dates, light_curve):
    ax.plot(dates, light_curve, color="black", drawstyle="steps-mid")
    ax.hlines(1.0, dates[0], dates[-1], color="0.5", linestyle="dashed",
            zorder=-1)
    return ax


def plot_light_curve(full_df, field, band="A", scaled_data=True, outname=None):
    if outname is None:
        outname = f"{field}_{band}_light_curve"
    field_df = full_df.xs(field, level="field")
    pb = PerleyButler17Poly(field) if not scaled_data else None
    dates, light_curve = get_light_curve_data(field_df, band, pb_poly=pb)
    fig, ax = plt.subplots(figsize=(4, 3))
    overplot_light_curve(ax, dates, light_curve)
    ax.set_title(f"{field.replace('_', '')}; {band} Band")
    ax.set_xticklabels(dates, rotation=90)
    ax.set_xlabel(r"Execution Block")
    ax.set_ylabel(r"$I_\nu(\mathcal{I}) / I_{\nu;\mathrm{PB17}}$")
    plt.tight_layout()
    savefig(f"{outname}")


def plot_all_light_curves(full_df, bands=("U", "A")):
    fields = full_df.index.get_level_values("field").unique()
    for field in fields:
        for band in bands:
            try:
                plot_light_curve(full_df, field, band=band)
            except ValueError:
                warnings.warn(f"Invalid values in {field} Band {band}.")


def plot_weather_series(df):
    fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(4, 7))
    dates = df.index
    for ax, ylabel, col in zip(axes, WEATHER_YLABELS, WEATHER_COLS):
        weather_data = df[col]
        ax.plot(dates, weather_data, color="black", drawstyle="steps-mid")
        ax.xaxis.set_ticklabels([])
        ax.set_ylabel(ylabel)
    ax.set_xlabel("Execution Block")
    ax.xaxis.set_ticklabels(list(dates), rotation=90)
    plt.tight_layout()
    savefig(f"weather_series")


def plot_weather_light_curve(field_df, w_df, field, band, outname=None):
    if outname is None:
        outname = f"{field}_{band}_light_curve_weather"
    dates, light_curve = get_light_curve_data(field_df, band)
    fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=True,
            figsize=(4, 7))
    axes[0].set_title(f"{field}; {band} Band")
    w_color = "red"
    for ax, ylabel, col in zip(axes, WEATHER_YLABELS, WEATHER_COLS):
        # Relative intensity to PB17
        overplot_light_curve(ax, dates, light_curve)
        ax.xaxis.set_ticklabels([])
        ax.set_ylabel(r"$I_\nu(\mathcal{I}) / I_{\nu;\mathrm{PB17}}$")
        # Weather data
        weather_data = w_df.loc[dates, col]
        ax_w = ax.twinx()
        ax_w.plot(dates, weather_data, color=w_color, linestyle="dashed",
                linewidth=1, marker="o", markersize=2)
        ax_w.xaxis.set_ticklabels([])
        ax_w.tick_params(axis="y", labelcolor=w_color)
        ax_w.set_ylabel(ylabel, color=w_color)
    ax.set_xlabel("Execution Block")
    ax.xaxis.set_ticklabels(list(dates), rotation=90)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.tight_layout()
    savefig(f"{outname}")


def plot_all_weather_light_curves(full_df, w_df, fields=None, bands=("U", "A")):
    if fields is None:
        fields = full_df.index.get_level_values("field").unique()
    for field in fields:
        field_df = full_df.xs(field, level="field")
        for band in bands:
            try:
                plot_weather_light_curve(field_df, w_df, field, band)
            except ValueError:
                warnings.warn(f"Invalid values in {field} Band {band}.")


