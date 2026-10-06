"""Exactly 15 compact features from the fixed 25th DAS channel."""
from __future__ import annotations

import numpy as np
from scipy import signal, stats


FEATURE_NAMES = ["rms", "standard_deviation", "mean_absolute_value", "peak_absolute_amplitude",
                 "peak_to_peak_amplitude", "skewness", "kurtosis", "zero_crossing_rate",
                 "crest_factor", "dominant_frequency_hz", "spectral_centroid_hz",
                 "spectral_spread_hz", "spectral_entropy", "relative_energy_0_500_hz",
                 "relative_energy_500_1000_hz"]


def extract_features(waveform: np.ndarray, fs: float = 2000, nperseg: int = 256,
                     noverlap: int = 128, bands: list[list[float]] | None = None,
                     eps: float = 1e-12, window: str = "hann") -> np.ndarray:
    """Feature statistics use all 2000 input points, including fixed prefix zeros.

    std uses population ddof=0; skew and Pearson kurtosis use bias=True.
    A crossing is a sign change between consecutive *non-zero* values, so zero
    padding does not create artificial crossings. ZCR denominator is T-1.
    Welch: Hann, constant detrend, one-sided density. Spectral entropy is the
    Shannon entropy of normalized PSD bins, divided by log(number of bins).
    Both trapezoidal band integrals share their 500 Hz endpoint; together their
    sub-intervals exactly partition the 0--1000 Hz integral.
    """
    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("Feature waveform must be finite 1D array")
    absolute = np.abs(x)
    rms = np.sqrt(np.mean(x ** 2))
    std = x.std(ddof=0)
    peak = absolute.max()
    nonzero = x[x != 0]
    crossings = np.sum(np.signbit(nonzero[1:]) != np.signbit(nonzero[:-1]))
    skew = stats.skew(x, bias=True) if std > eps else 0.0
    kurt = stats.kurtosis(x, fisher=False, bias=True) if std > eps else 0.0
    frequencies, psd = signal.welch(x, fs=fs, window=window, nperseg=nperseg,
                                    noverlap=noverlap, detrend="constant", scaling="density")
    total = np.trapezoid(psd, frequencies)
    weights = psd / (psd.sum() + eps)
    centroid = np.sum(weights * frequencies)
    spread = np.sqrt(np.sum(weights * (frequencies - centroid) ** 2))
    entropy = -np.sum(weights * np.log(weights + eps)) / np.log(len(weights))
    selected_bands = bands or [[0, 500], [500, 1000]]
    if len(selected_bands) != 2:
        raise ValueError("Exactly two handcrafted frequency bands are required")
    band_values = []
    for lo, hi in selected_bands:
        # Interpolate boundaries as needed rather than silently dropping intervals.
        band_f = np.concatenate(([lo], frequencies[(frequencies > lo) & (frequencies < hi)], [hi]))
        band_p = np.interp(band_f, frequencies, psd)
        band_values.append(np.trapezoid(band_p, band_f) / total if total > eps else 0.0)
    result = np.array([rms, std, absolute.mean(), peak, np.ptp(x), skew, kurt,
                       crossings / (len(x) - 1), peak / (rms + eps),
                       frequencies[np.argmax(psd)], centroid, spread, entropy, *band_values], dtype=np.float64)
    if result.shape != (15,) or not np.isfinite(result).all():
        raise ValueError("Handcrafted feature contains NaN/Inf or incorrect dimension")
    return result
