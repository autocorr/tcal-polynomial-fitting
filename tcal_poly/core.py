#!/usr/bin/env python3

import re
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
MAD_TO_STD = 1.4826
TEST_FSCALE_LINE = "# Flux density for J2355+4950 in SpW=0 (freq=3.2072e+10 Hz) is: 0.289629 +/- 0.0493206 (SNR = 5.87238, N = 42)"


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


def get_all_fscale_files():
    return sorted(PATHS.extern.glob("20??-??-*/?/*.ms.fscale.dat"))


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
        df.index.name = "spw"
        df.reset_index(inplace=True)
        df["field"] = self.field
        df["date"] = self.date
        df["band"] = self.band
        df["mjs"] = mjs
        df.set_index(["field", "date", "band", "spw"], inplace=True)
        self.df = df


def aggregate_dynamic_seds(parallel=True, nproc=30):
    field_files = get_all_field_files()
    if parallel:
        assert nproc > 0
        with Pool(nproc) as pool:
            data_files = pool.map(DataFile, field_files)
    else:
        data_files = [
                DataFile(file_path)
                for file_path in field_files
        ]
    df = pd.concat([d.df for d in data_files])
    return df


class FluxFile:
    expression = (
            R"# Flux density for (?P<field>.+?) "
            R"in SpW=(?P<spw>\d+) "
            R"\(freq=(?P<freq>\S+) Hz\) "
            R"is: (?P<flux_I>\S+) \+/\- (?P<flux_I_err>\S+) "
            R"\(SNR = (?P<snr>\S+?), N = (?P<nchan>\S+?)\)"
    )
    columns = "field,spw,freq,flux_I,flux_I_err,snr,nchan".split(",")
    dtypes = (str, int, float, float, float, float, int)

    def __init__(self, filen):
        """
        Parameters
        ----------
        filen : str
            Full path to fluxscale data file, e.g. "*_A.ms.fscale.dat".
        """
        filen = Path(filen)
        assert filen.exists()
        date = filen.parent.parent.name
        p = re.compile(self.expression)
        df = pd.DataFrame(columns=self.columns)
        with filen.open("r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                try:
                    group = p.match(line).groupdict()
                    for col in self.columns:
                        df.loc[i, col] = group[col]
                except AttributeError:
                    continue
        df = df.astype({c: d for c, d in zip(self.columns, self.dtypes)})
        df["band"] = filen.parent.name
        df["date"] = filen.parent.parent.name
        df.set_index(["field", "date", "band", "spw"], inplace=True)
        self.df = df


def aggregate_flux_files(parallel=True, nproc=30):
    flux_files = get_all_fscale_files()
    if parallel:
        assert nproc > 0
        with Pool(nproc) as pool:
            flux_files = pool.map(FluxFile, flux_files)
    else:
        flux_files = [
                FluxFile(file_path)
                for file_path in flux_files
        ]
    df = pd.concat([f.df for f in flux_files])
    return df


def read_df(filen="survey"):
    if not filen.endswith(".csv"):
        filen = f"{filen}.csv"
    index_col = ["field", "date", "band", "spw"]
    df = pd.read_csv(PATHS.data/filen, index_col=index_col)
    return df


class SedPoly:
    log_freq_delta = 0.02  # log10(GHz)
    min_order = 2
    max_order = 4

    def __init__(self, freq, flux, flux_err, nbands, use_weights=False,
            fit_order=None, pb_poly=None):
        assert nbands > 0
        if freq.min() > 1e3:
            freq *= u.Hz.to("GHz")
        log_freq = np.log10(freq)
        if pb_poly is not None:
            pb_flux = pb_poly.eval_flux(log_freq)
            flux /= pb_flux
            flux_err /= pb_flux
        self.log_freq = log_freq
        self.data_freq = freq
        self.data_flux = flux
        self.data_flux_err = flux_err
        self.good_data_mask = np.ones_like(freq, dtype=bool)
        self.nbands = nbands
        if fit_order is None:
            fit_order = max(min(nbands, self.max_order), self.min_order)
        self.use_weights = use_weights
        self.fit_order = fit_order
        self.pb_poly = pb_poly
        self.mask_by_pb_deviation()
        # Fit polynomial
        self.weights = 1 / np.abs(flux_err)  # Gaussian weights
        self.poly_coef = None
        self.poly_freq = None
        self.poly_flux = None
        self.fit()

    @classmethod
    def from_stokes(cls, df, stokes, **kwargs):
        df = df.sort_values("freq")
        freq = df.freq
        flux = df[f"flux_{stokes}"].values
        flux_err = df[f"flux_{stokes}_err"].values
        nbands = get_num_bands(df)
        return cls(freq, flux, flux_err, nbands, **kwargs)

    @property
    def bad_data_mask(self):
        return ~self.good_data_mask

    @property
    def clipped_freq(self):
        return self.log_freq[self.bad_data_mask]

    @property
    def clipped_flux(self):
        return self.data_flux[self.bad_data_mask]

    @property
    def clipped_weights(self):
        return self.weights[self.bad_data_mask]

    @property
    def residuals(self):
        mask = self.good_data_mask
        poly_flux = np.polyval(self.poly_coef, self.log_freq[mask])
        data_flux = self.data_flux[mask]
        return poly_flux - data_flux

    def mask_by_pb_deviation(self, fact_hi=5.0, fact_lo=0.5):
        if self.pb_poly is not None:
            keep_mask = (fact_lo < self.data_flux) & (self.data_flux < fact_hi)
            self.good_data_mask &= keep_mask

    def fit(self):
        mask = self.good_data_mask
        log_freq = self.log_freq[mask]
        data_flux = self.data_flux[mask]
        weights = self.weights[mask] if self.use_weights else None
        poly_coef = np.polyfit(
                log_freq,
                data_flux,
                self.fit_order,
                w=weights,
        )
        poly_freq = np.arange(
                log_freq.min()-self.log_freq_delta,
                log_freq.max()+self.log_freq_delta,
                self.log_freq_delta,
        )
        poly_flux = np.polyval(poly_coef, poly_freq)
        self.poly_coef = poly_coef
        self.poly_freq = poly_freq
        self.poly_flux = poly_flux

    def clip_and_refit(self, sigma=10, n_clips=3):
        assert sigma > 0
        assert n_clips > 0
        for _ in range(n_clips):
            residuals = self.residuals
            med = np.nanmedian(residuals)
            mad = np.nanmedian(np.abs(residuals - med))
            std = MAD_TO_STD * mad
            self.good_data_mask[self.good_data_mask] &= (
                    np.abs(residuals) < sigma * std
            )
            self.fit()

    def plot_data(self, ax, show_clipped=True):
        # Plot good data
        g_freq = self.log_freq[self.good_data_mask]
        g_flux = self.data_flux[self.good_data_mask]
        g_errs = self.data_flux_err[self.good_data_mask]
        ax.errorbar(g_freq, g_flux, yerr=g_errs, color="black",
                marker="o", markersize=2, linestyle="none",
                ecolor="black", elinewidth=0.5)
        # Plot bad/clipped data
        if show_clipped:
            b_freq = self.clipped_freq
            b_flux = self.clipped_flux
            b_errs = self.data_flux_err[self.bad_data_mask]
            ax.errorbar(b_freq, b_flux, yerr=b_errs, color="0.5",
                    marker="o", markersize=2, linestyle="none",
                    ecolor="0.5", elinewidth=0.5)

    def plot_poly(self, ax):
        ax.plot(self.poly_freq, self.poly_flux, "r-")


class PerleyButler17Poly:
    # Polynomial coefficients listed in reverse order, e.g. [a2, a1, a0]
    coef_by_field = {
            "J0133-3629":   [1.0440,  -0.6619, -0.2252],
            "3C48":         [1.3253,  -0.7553, -0.1914,  0.0498],
            "Fornax A":     [2.2175,  -0.6606],
            "3C123":        [1.8017,  -0.7884, -0.1035, -0.0248,  0.0090],
            "J0444-2809":   [0.9710,  -0.8938, -0.1176],
            "3C138":        [1.0088,  -0.4981, -0.1552, -0.0102,  0.0223],
            "Pictor A":     [1.9380,  -0.7470, -0.0739],
            "Taurus A":     [2.9516,  -0.2173, -0.0473, -0.0674],
            "3C147":        [1.4516,  -0.6961, -0.2007,  0.0640, -0.0464,  0.0289],
            "3C196":        [1.2872,  -0.8530, -0.1534, -0.0200,  0.0201],
            "Hydra A":      [1.7795,  -0.9176, -0.0843, -0.0139,  0.0295],
            "Virgo A":      [2.4466,  -0.8116, -0.0483],
            "3C286":        [1.2481,  -0.4507, -0.1798,  0.0357],
            "3C295":        [1.4701,  -0.7658, -0.2780, -0.0347,  0.0399],
            "Hercules A":   [1.8298,  -1.0247, -0.0951],
            "3C353":        [1.8627,  -0.6938, -0.0998, -0.0732],
            "3C380":        [1.2320,  -0.7909,  0.0947,  0.0976, -0.1794, -0.1566],
            "Cygnus A":     [3.3498,  -1.0022, -0.2246,  0.0227,  0.0425],
            "3C444":        [1.1064,  -1.0052, -0.0750, -0.0767],
            "Cassiopeia A": [3.3584,  -0.7518, -0.0347, -0.0705],
    }

    def __init__(self, field):
        field = field.split("=")[-1] if "=" in field else field
        self.field = field
        try:
            self.p_coef = list(reversed(self.coef_by_field[field]))
        except KeyError:
            raise ValueError(f"Invalid field for PB17: {field}")

    def eval_flux(self, logfreq):
        assert logfreq.min() > np.log10( 0.05)
        assert logfreq.max() < np.log10(50.0)
        return 10**np.polyval(self.p_coef, logfreq)

    def eval_from_poly(self, poly):
        d_flux_ratio = self.eval_flux(poly.l_freq)
        pb_d_flux = self.eval_flux(poly.l_freq)
        pb_p_flux = self.eval_flux(poly.p_freq)
        d_ratio = poly.d_flux / pb_d_flux
        p_ratio = poly.p_flux / pb_p_flux
        return d_ratio, p_ratio


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
        poly = SedPoly.from_stokes(df, stokes)
        poly.clip_and_refit(sigma=5.0)
        poly.plot_data(ax)
        poly.plot_poly(ax)
        ax.set_ylabel(rf"$I_\nu(\mathcal{{ {stokes} }}) \ [\mathrm{{Jy}}]$")
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(r"$\log_{10}(f \ [\mathrm{GHz}])$")
    plt.tight_layout()
    savefig(f"{outname}.pdf")


def plot_relative_i_sed(date_df, field, date, scaled_data=True, outname=None):
    df = date_df
    assert len(df) > 0
    pb = PerleyButler17Poly(field) if not scaled_data else None
    poly = SedPoly.from_stokes(date_df, "I", pb_poly=pb)
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
    savefig(f"{outname}.pdf")


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


def plot_light_curve(full_df, field, band="A", scaled_data=True, outname=None):
    if outname is None:
        outname = f"{field}_{band}_light_curve"
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
    pb = PerleyButler17Poly(field) if not scaled_data else None
    field_df = full_df.xs(field, level="field")
    try:
        dates = field_df.xs("A", level="band").index.get_level_values("date").unique()
    except KeyError:
        raise ValueError(f"Field does not contain A Band: {field}")
    dates.sort_values()
    print(dates)
    field_df = field_df.loc[idx[dates, :, :]]
    dates = [
            d for d in dates if d not in
            []
    ]
    light_curve = []
    for date in dates:
        date_df = field_df.xs(date, level="date")
        poly = SedPoly.from_stokes(date_df, "I", pb_poly=pb)
        poly.clip_and_refit(sigma=5.0)
        fluxes = np.polyval(poly.poly_coef, band_freqs)
        light_curve.append(fluxes)
    fluxes = np.array(fluxes).T
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(dates, light_curve, color="black", drawstyle="steps-mid")
    ax.hlines(1.0, dates[0], dates[-1], color="cyan", linestyle="dashed",
            zorder=-1)
    ax.set_title(f"{field.replace('_', '')}; {band} Band")
    ax.set_xticklabels(dates, rotation=90)
    ax.set_xlabel(r"Execution Block")
    ax.set_ylabel(r"$I_\nu(\mathcal{I}) / I_{\nu;\mathrm{PB17}}$")
    plt.tight_layout()
    savefig(f"{outname}.pdf")


def plot_all_light_curves(full_df, bands=["U", "A"]):
    fields = full_df.index.get_level_values("field").unique()
    for field in fields:
        for band in bands:
            try:
                plot_light_curve(full_df, field, band=band)
            except ValueError:
                pass


