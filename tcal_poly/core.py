#!/usr/bin/env python3

import copy
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pandas import IndexSlice as idx
from scipy import optimize
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from astropy import units as u
from astropy.time import Time
from casatools import msmetadata

from tcal_poly import PATHS


# Matplotlib configuration settings
plt.rc("text", usetex=True)
plt.rc("font", size=10, family="serif")
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.ioff()

CMAP = copy.copy(plt.cm.get_cmap("magma"))
CMAP.set_bad("0.5", 1.0)


STOKES = list("IQUV")
BANDS = list("LSCXUKAQ")


def savefig(outname, dpi=300, relative=True, overwrite=True):
    outpath = PATHS.plot / outname if relative else Path(outname)
    if outpath.exists() and not overwrite:
        print(f"Figure exists, continuing: {outpath}")
    else:
        plt.savefig(str(outpath), dpi=dpi)
        print(f"Figure saved to: {outpath}")
        plt.close("all")


def get_all_execution_dirs():
    return sorted(PATHS.extern.glob("20??-??-*"))


def get_all_field_files():
    return sorted(PATHS.extern.glob("20??-??-*/?/images/field_*.dat"))


def mjs_to_date(mjs):
    mjd = mjs * u.s.to("day")
    time = Time(mjd, format="mjd")
    return str(time.datetime.date())


def weighted_mean(vals, errs):
    if len(errs) == 0:
        return np.nan
    else:
        weights = 1 / errs**2
        return np.nansum(vals * weights) / np.nansum(weights)


def weighted_mean_error(errs):
    if len(errs) == 0:
        return np.nan
    else:
        weights = 1 / errs**2
        return np.sqrt(1 / np.nansum(weights))


def get_mjs_from_ms_path(path, field=None):
    ms_path = list(path.parent.parent.glob("*.ms"))[0]
    try:
        tool = msmetadata()
        tool.open(str(ms_path))
        fields = tool.fieldnames()
        if field is None:
            scan_id = tool.scannumbers()[0]
        else:
            scan_id = tool.scansforfield(field)
        time = tool.timesforscans(scan_id)
        return time.min()  # seconds
    except RuntimeError:
        raise
    finally:
        tool.done()
        tool.close()
    return times.mean()


class DataFile:
    columns = [
            "freq",
            "flux_I",
            "flux_I_err",
            "flux_I_peak",
            "flux_I_peak_err",
            "flux_Q",
            "flux_Q_err",
            "flux_Q_peak",
            "flux_Q_peak_err",
            "flux_U",
            "flux_U_err",
            "flux_U_peak",
            "flux_U_peak_err",
            "flux_V",
            "flux_V_err",
            "flux_V_peak",
            "flux_V_peak_err",
    ]
    stokes = ("I", "Q", "U", "V")

    def __init__(self, filen):
        filen = Path(filen)
        assert filen.exists()
        with filen.open("r") as f:
            line = f.readline().split()
            self.field = line[2].rstrip(";")
            self.band = line[4]
        self.date = filen.parent.parent.parent.name
        mjs = get_mjs_from_ms_path(filen, field=self.field)
        df = pd.read_csv(filen, names=self.columns, header=0,
                index_col=False, skiprows=2, sep=" ", na_values=0,
                comment="#")
        df.index.name = "channel"
        df.reset_index(inplace=True)
        df["field"] = self.field
        df["date"] = self.date
        df["band"] = self.band
        df["mjs"] = mjs
        df.set_index(["field", "date", "band", "channel"], inplace=True)
        self.df = df


def aggregate_dynamic_seds(parallel=True, nproc=30):
    if parallel:
        assert nproc > 0
        with Pool(nproc) as pool:
            data_files = pool.map(DataFile, get_all_field_files())
    else:
        data_files = [
                DataFile(file_path)
                for file_path in get_all_field_files()
        ]
    df = pd.concat([d.df for d in data_files])
    return df


class SedPoly:
    log_freq_delta = 0.02  # log10(GHz)
    min_order = 2
    max_order = 3

    def __init__(self, freq, flux, flux_err, nbands, fit_order=None):
        assert nbands > 0
        if freq.min() > 1e3:
            freq *= u.Hz.to("GHz")
        l_freq = np.log10(freq)
        weights = 1 / np.abs(flux_err)  # Gaussian weights
        if fit_order is None:
            fit_order = max(min(nbands, self.max_order), self.min_order)
        # FIXME do some error checking and cleaning
        # Fit polynomial
        p_coef = np.polyfit(
                l_freq, flux, fit_order,
                #w=weights,
        )
        p_freq = np.arange(
                l_freq.min()-self.log_freq_delta,
                l_freq.max()+self.log_freq_delta,
                self.log_freq_delta,
        )
        p_flux = np.polyval(p_coef, p_freq)
        # Assign attributes
        self.d_freq = freq
        self.l_freq = l_freq
        self.d_flux = flux
        self.d_flux_err = flux_err
        self.weights = weights
        self.nbands = nbands
        self.fit_order = fit_order
        self.p_coef = p_coef
        self.p_freq = p_freq
        self.p_flux = p_flux

    @classmethod
    def from_stokes(cls, df, stokes, **kwargs):
        df = df.sort_values("freq")
        freq = df.freq
        flux = df[f"flux_{stokes}"].values
        flux_err = df[f"flux_{stokes}_err"].values
        nbands = get_num_bands(df)
        return cls(freq, flux, flux_err, nbands, **kwargs)

    def clip_and_refit(self, sigma=10, n_clips=3):
        assert sigma > 0
        assert n_clips > 0
        d_flux = self.d_flux.copy()
        l_freq = self.l_freq.copy()
        weights = self.weights.copy()
        p_coef = self.p_coef
        clipped_flux = np.array([])
        clipped_freq = np.array([])
        for _ in range(n_clips):
            p_flux = np.polyval(p_coef, l_freq)
            residuals = p_flux - d_flux
            mad = np.nanmedian(np.abs(residuals - np.nanmedian(residuals)))
            std = 1.4826 * mad
            mask = np.abs(residuals) < sigma * std
            clipped_flux = np.concatenate([clipped_flux, d_flux[~mask]])
            clipped_freq = np.concatenate([clipped_freq, l_freq[~mask]])
            d_flux = d_flux[mask]
            l_freq = l_freq[mask]
            weights = weights[mask]
            p_coef = np.polyfit(
                    l_freq, d_flux, self.fit_order,
                    #w=self.weights,
            )
        self.p_coef = p_coef
        self.p_flux = np.polyval(p_coef, self.p_freq)
        return clipped_freq, clipped_flux


def fit_sed(df, fit_order=3):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame sub-selected for field, band, and time.

    Returns
    -------
    poly
    freq_min
    freq_max
    """
    assert fit_order > 0
    df = df.copy()
    for stokes in STOKES:
        poly = SedPoly.from_stokes(df, stokes)
    # do some cleaning, all data is good, etc.
    # get array of all frequencies, fluxes, and uncertainties
    # put fit polynomials into new DF
    # return coefficients
    return


def get_num_bands(df):
    return df.index.get_level_values("band").unique().size


def plot_sed(date_df, outname=None):
    df = date_df
    assert len(df) > 0
    freq = df.freq.values * u.Hz.to("GHz")
    log_freq = np.log10(freq)
    target, date, _, _ = df.index[0]
    if outname is None:
        outname = f"{target}_{date}_sed"
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(4, 6))
    for stokes, ax in zip(STOKES, axes):
        flux = df[f"flux_{stokes}"]
        flux_err = df[f"flux_{stokes}_err"].abs()
        ax.errorbar(log_freq, flux, yerr=flux_err, color="black",
                marker="o", markersize=2, linestyle="none",
                ecolor="black", elinewidth=0.5)
        poly = SedPoly.from_stokes(df, stokes)
        c_freq, c_flux = poly.clip_and_refit()
        ax.plot(poly.p_freq, poly.p_flux, "r-")
        ax.plot(c_freq, c_flux, color="0.5", marker="o", markersize=2.1,
                linestyle="none", zorder=20)
        ax.set_ylabel(rf"$I_\nu(\mathcal{{ {stokes} }}) \ [\mathrm{{Jy}}]$")
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(r"$\log_{10}(f \ [\mathrm{GHz}])$")
    plt.tight_layout()
    savefig(f"{outname}.pdf")


def plot_all_seds(df):
    field_names = df.index.get_level_values("field").unique()
    for field_name in field_names:
        field_df = df.loc[idx[field_name, :, :, :]]
        dates = field_df.index.get_level_values("date").unique()
        for date in dates:
            date_df = df.loc[idx[field_name, date, :, :]]
            try:
                plot_sed(date_df)
            except np.linalg.LinAlgError:
                continue


def test_plot_light_curve(full_df, outname=None):
    if outname is None:
        outname = "3C48_light_curve"
    field_df = full_df.loc[idx["0137+331=3C48", :, "A", :]]
    dates = field_df.index.get_level_values("date").unique().sort_values()
    dates = [
            d for d in dates if d not in
            # Bad dates for 3C138
            #("2021-05-09", "2021-06-15", "2020-02-03", "2018-01-30A",
            #    "2021-03-08", "2019-07-27")
            # Bad dates for 3C48
            ["2020-02-03"]
    ]
    light_curve = []
    #band_freqs = np.log10(np.array([1.5, 3.0, 6.0, 10.0, 15.0, 22.3, 33.3, 45.0]))
    band_freqs = np.log10(np.array([33.3]))
    for date in dates:
        date_df = field_df.loc[idx[:, date, :, :]]
        poly = SedPoly.from_stokes(date_df, "I")
        poly.clip_and_refit()
        fluxes = np.polyval(poly.p_coef, band_freqs)
        light_curve.append(fluxes)
    fluxes = np.array(fluxes).T
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(dates, light_curve, drawstyle="steps-mid")
    ax.set_xticklabels(dates, rotation=90)
    ax.set_xlabel(r"Execution Block")
    ax.set_ylabel(r"$I_\nu(\mathcal{I}) \ [\mathrm{Jy}]$")
    plt.tight_layout()
    savefig(f"{outname}.pdf")


