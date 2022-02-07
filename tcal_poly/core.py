#!/usr/bin/env python3

import re
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

from astropy import units as u
from astropy.time import Time
from casatools import msmetadata

from tcal_poly import PATHS


STOKES = list("IQUV")
BANDS = list("LSCXUKAQ")
MAD_TO_STD = 1.4826
TEST_FSCALE_LINE = "# Flux density for J2355+4950 in SpW=0 (freq=3.2072e+10 Hz) is: 0.289629 +/- 0.0493206 (SNR = 5.87238, N = 42)"


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


def regularize_name(name):
    if name == "3C138":
        return "0521+166=3C138"
    else:
        return name


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
            self.field = regularize_name(line[2].rstrip(";"))
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
        df["field"] = df.field.apply(regularize_name)
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


def with_csv_ext(name):
    if not name.endswith(".csv"):
        return f"{name}.csv"
    else:
        return name


def read_df(filen="survey"):
    filen = with_csv_ext(filen)
    index_col = ["field", "date", "band", "spw"]
    df = pd.read_csv(PATHS.data/filen, index_col=index_col)
    return df


def read_weather(filen="weather"):
    filen = with_csv_ext(filen)
    return pd.read_csv(PATHS.data/filen, index_col="date")


def get_num_bands(df):
    return df.index.get_level_values("band").unique().size


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


