#!/usr/bin/env python3
"""Compute and plot the Power Spectral Density (PSD) of a signal stored in a .mat file.

Reads a MATLAB .mat file containing a 1-D signal (plus optional metadata such as
sample rate, and, for the MARS hydrophone dataset format, fields like fs, device,
passage, track, sog, rpm_babord, rpm_tribord) and estimates its PSD using Welch's
method at a user-specified frequency resolution.

Example:
    python3 psd.py raw_2_MARS12_3_1.mat --resolution 1.0 --output psd.png
    python3 psd.py raw_2_MARS12_3_1.mat --resolution 0.5 --fmax 20000 --xlog
"""

import argparse
import sys

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

# Candidate variable names to look for the signal / sample rate inside the .mat file.
SIGNAL_KEYS = ("signal", "x", "data", "y")
FS_KEYS = ("fs", "Fs", "FS", "fsamp", "sample_rate")


def _first_present(d, keys):
    for k in keys:
        if k in d:
            return k
    return None


def load_signal(mat_path):
    """Load the signal and sample rate from a .mat file, returning (signal, fs, meta)."""
    d = loadmat(mat_path, squeeze_me=True)
    d = {k: v for k, v in d.items() if not k.startswith("__")}

    sig_key = _first_present(d, SIGNAL_KEYS)
    if sig_key is None:
        raise ValueError(
            f"Could not find a signal array in {mat_path}. "
            f"Available variables: {sorted(d.keys())}"
        )
    signal = np.asarray(d[sig_key]).squeeze().astype(float)
    if signal.ndim != 1:
        raise ValueError(f"Expected a 1-D signal, got shape {signal.shape} for '{sig_key}'")

    fs_key = _first_present(d, FS_KEYS)
    if fs_key is None:
        raise ValueError(
            f"Could not find a sample rate in {mat_path}. "
            f"Available variables: {sorted(d.keys())}"
        )
    fs = float(np.asarray(d[fs_key]).squeeze())

    meta = {k: v for k, v in d.items() if k not in (sig_key, fs_key)}
    return signal, fs, meta


def compute_psd(signal, fs, resolution, window="hann", overlap=0.5):
    """Estimate the PSD via Welch's method at the requested frequency resolution (Hz).

    resolution is the target frequency bin spacing df; nperseg = fs / df, rounded to
    the nearest power of two capped at the signal length for an efficient/exact FFT.
    """
    if resolution <= 0:
        raise ValueError("resolution must be > 0 Hz")

    nperseg = fs / resolution
    nperseg = int(2 ** round(np.log2(nperseg)))
    nperseg = max(8, min(nperseg, len(signal)))
    noverlap = int(nperseg * overlap)

    freqs, psd = welch(
        signal,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
    )
    actual_resolution = fs / nperseg
    return freqs, psd, actual_resolution, nperseg


def plot_psd(freqs, psd, meta_text, resolution, output=None, fmin=None, fmax=None,
             xlog=False, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    psd_db = 10 * np.log10(np.maximum(psd, np.finfo(float).tiny))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs, psd_db, linewidth=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB / Hz)")
    ax.set_title(f"Power Spectral Density (resolution ≈ {resolution:.4g} Hz)")
    ax.grid(True, which="both", alpha=0.3)

    lo = fmin if fmin is not None else freqs[0]
    hi = fmax if fmax is not None else freqs[-1]
    ax.set_xlim(lo, hi)

    if xlog:
        ax.set_xscale("log")
        if lo <= 0:
            ax.set_xlim(max(freqs[1], 1e-3), hi)

    if meta_text:
        ax.text(
            0.99, 0.99, meta_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"Saved plot to {output}")
    if show:
        plt.show()
    plt.close(fig)


def format_meta(meta, fs, duration):
    lines = [f"fs = {fs:g} Hz", f"duration = {duration:.3g} s"]
    for k, v in meta.items():
        try:
            arr = np.asarray(v)
            if arr.size == 1:
                v = arr.item()
            lines.append(f"{k} = {v}")
        except Exception:
            continue
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mat_file", help="Path to the .mat file containing the signal")
    parser.add_argument(
        "--resolution", "-r", type=float, default=1.0,
        help="Target frequency resolution in Hz (df) for the PSD estimate (default: 1.0)",
    )
    parser.add_argument("--fmin", type=float, default=None, help="Minimum frequency to plot (Hz)")
    parser.add_argument("--fmax", type=float, default=None, help="Maximum frequency to plot (Hz)")
    parser.add_argument("--xlog", action="store_true", help="Use a logarithmic frequency axis")
    parser.add_argument(
        "--window", default="hann",
        help="Window function passed to scipy.signal.welch (default: hann)",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.5,
        help="Fractional segment overlap for Welch's method (default: 0.5)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path to save the plot image (e.g. psd.png). If omitted, defaults to "
             "'<mat_file stem>_psd.png' unless --show is used.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot interactively")
    parser.add_argument(
        "--csv", default=None,
        help="Optional path to also save the PSD (freq, psd, psd_db) as CSV",
    )
    args = parser.parse_args(argv)

    signal, fs, meta = load_signal(args.mat_file)
    duration = len(signal) / fs
    freqs, psd, actual_resolution, nperseg = compute_psd(
        signal, fs, args.resolution, window=args.window, overlap=args.overlap
    )

    print(f"Loaded signal: {len(signal)} samples, fs = {fs:g} Hz, duration = {duration:.3g} s")
    print(f"Requested resolution: {args.resolution:g} Hz -> nperseg = {nperseg} "
          f"(actual resolution = {actual_resolution:.4g} Hz)")

    if args.csv:
        np.savetxt(
            args.csv,
            np.column_stack([freqs, psd, 10 * np.log10(np.maximum(psd, np.finfo(float).tiny))]),
            delimiter=",",
            header="frequency_hz,psd,psd_db",
            comments="",
        )
        print(f"Saved PSD data to {args.csv}")

    output = args.output
    if output is None and not args.show:
        import os
        stem = os.path.splitext(os.path.basename(args.mat_file))[0]
        output = f"{stem}_psd.png"

    meta_text = format_meta(meta, fs, duration)
    plot_psd(
        freqs, psd, meta_text, actual_resolution,
        output=output, fmin=args.fmin, fmax=args.fmax, xlog=args.xlog, show=args.show,
    )


if __name__ == "__main__":
    sys.exit(main())
