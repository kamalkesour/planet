#!/usr/bin/env python3
"""Locate the Engine Firing Frequency (EFF) in the PSD of a ship-noise .mat recording.

For a 4-stroke engine, each cylinder fires once every 2 crankshaft
revolutions, so with Ncyl cylinders the firing-pulse train repeats at:

    EFF (Hz) = (RPM / 60) * (2 * Ncyl / strokes)

For a 4-stroke, 6-cylinder engine this simplifies to EFF = RPM / 20.
Given a plausible engine speed range [rpm_min, rpm_max], the frequency
band to search for EFF in the PSD is therefore:

    [eff(rpm_min), eff(rpm_max)]

e.g. 6 cylinders, 4-stroke, 500-1800 RPM -> search 25.0-90.0 Hz.

Combustion impulses are broadband/impulsive, so EFF and its harmonics
usually stand out as narrow tonals riding on top of the broader
machinery/flow-noise floor. This script computes the PSD of the raw
signal, whitens it against a local (median-filtered) baseline to expose
those narrow tonals even where the floor itself is sloped or humped, and
reports the strongest candidates within the computed EFF band.

Caveat: other rotating machinery (an auxiliary generator running at a
constant, propulsion-independent RPM, for example) can produce its own
tonal inside the same band and be *louder* than the true, speed-dependent
EFF line, especially at higher engine speeds where cavitation/flow noise
also picks up. Treat the ranked list as candidates to confirm against
what you know about the vessel (idle/cruise speed, generator RPM), not as
an infallible single answer -- always check the saved plot.

Example:
    python3 eff_estimate.py raw_6_MARS12_3_5.mat --cylinders 6 --strokes 4 \\
        --rpm-min 500 --rpm-max 1800
"""

import argparse
import sys

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, find_peaks, medfilt

SIGNAL_KEYS = ("signal", "x", "data", "y")
FS_KEYS = ("fs", "Fs", "FS", "fsamp", "sample_rate")


def _first_present(d, keys):
    for k in keys:
        if k in d:
            return k
    return None


def load_signal(mat_path):
    d = loadmat(mat_path, squeeze_me=True)
    d = {k: v for k, v in d.items() if not k.startswith("__")}

    sig_key = _first_present(d, SIGNAL_KEYS)
    if sig_key is None:
        raise ValueError(f"Could not find a signal array in {mat_path}. "
                          f"Available variables: {sorted(d.keys())}")
    signal = np.asarray(d[sig_key]).squeeze().astype(float)

    fs_key = _first_present(d, FS_KEYS)
    if fs_key is None:
        raise ValueError(f"Could not find a sample rate in {mat_path}. "
                          f"Available variables: {sorted(d.keys())}")
    fs = float(np.asarray(d[fs_key]).squeeze())

    return signal, fs


def eff_of_rpm(rpm, n_cyl, strokes):
    return rpm / 60.0 * (2.0 * n_cyl / strokes)


def rpm_of_eff(eff, n_cyl, strokes):
    return eff * 60.0 / (2.0 * n_cyl / strokes)


def whitened_psd(signal, fs, resolution, fmin, fmax, context_factor_lo=0.5, context_factor_hi=1.3,
                  baseline_hz=3.0):
    nperseg = int(fs / resolution)
    nperseg = int(2 ** round(np.log2(nperseg)))
    nperseg = max(64, min(nperseg, len(signal)))
    freqs, psd = welch(signal, fs=fs, window="hann", nperseg=nperseg,
                        noverlap=nperseg // 2, scaling="density")
    df = fs / nperseg

    ctx_mask = (freqs >= fmin * context_factor_lo) & (freqs <= fmax * context_factor_hi)
    fb, p_db = freqs[ctx_mask], 10 * np.log10(psd[ctx_mask] + 1e-30)

    ksize = int(baseline_hz / df)
    ksize += (ksize % 2 == 0)
    ksize = max(3, ksize)
    baseline = medfilt(p_db, kernel_size=ksize)
    prominence_db = p_db - baseline
    return fb, p_db, prominence_db, df


def rank_eff_candidates(fb, p_db, prom_db, fmin, fmax, top_n, min_prominence=1.0):
    band_mask = (fb >= fmin) & (fb <= fmax)
    fbb, promb, pdbb = fb[band_mask], prom_db[band_mask], p_db[band_mask]
    pk_idx, _ = find_peaks(promb, prominence=min_prominence)
    order = np.argsort(promb[pk_idx])[::-1][:top_n]
    candidates = [(fbb[pk_idx][i], promb[pk_idx][i], pdbb[pk_idx][i]) for i in order]
    return candidates


def plot_eff(fb, p_db, fmin, fmax, candidates, n_cyl, strokes, output=None, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fb, p_db, linewidth=0.8, color="steelblue")
    ax.axvspan(fmin, fmax, color="orange", alpha=0.12, label=f"EFF search band ({fmin:.1f}-{fmax:.1f} Hz)")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for rank, (f0, prom, pdb) in enumerate(candidates):
        c = colors[rank % len(colors)]
        ax.axvline(f0, color=c, linewidth=1.6,
                    label=f"#{rank+1}: {f0:.2f} Hz -> {rpm_of_eff(f0, n_cyl, strokes):.0f} RPM "
                          f"(prom={prom:.1f} dB)")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB / Hz)")
    ax.set_title(f"EFF search — {n_cyl}-cyl {strokes}-stroke engine")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"Saved plot to {output}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mat_file", help="Path to the .mat file containing the signal")
    parser.add_argument("--cylinders", type=int, default=6, help="Number of engine cylinders (default: 6)")
    parser.add_argument("--strokes", type=int, choices=(2, 4), default=4,
                         help="Engine cycle: 2-stroke or 4-stroke (default: 4)")
    parser.add_argument("--rpm-min", type=float, default=500, help="Minimum plausible engine RPM (default: 500)")
    parser.add_argument("--rpm-max", type=float, default=1800, help="Maximum plausible engine RPM (default: 1800)")
    parser.add_argument("--resolution", type=float, default=0.5,
                         help="PSD frequency resolution in Hz (default: 0.5)")
    parser.add_argument("--top-n", type=int, default=5, help="Number of ranked candidates to report")
    parser.add_argument("--output", "-o", default=None,
                         help="Path to save the plot (default: '<file stem>_eff.png')")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively")
    args = parser.parse_args(argv)

    signal, fs = load_signal(args.mat_file)

    fmin = eff_of_rpm(args.rpm_min, args.cylinders, args.strokes)
    fmax = eff_of_rpm(args.rpm_max, args.cylinders, args.strokes)

    print(f"Loaded '{args.mat_file}': {len(signal)} samples, fs = {fs:g} Hz")
    print(f"Engine: {args.cylinders} cylinders, {args.strokes}-stroke, "
          f"RPM range {args.rpm_min:g}-{args.rpm_max:g}")
    print(f"EFF formula: EFF = RPM/60 * (2*{args.cylinders}/{args.strokes}) = "
          f"RPM * {2*args.cylinders/args.strokes/60:.4f}")
    print(f"==> EFF search band in the PSD: {fmin:.2f} - {fmax:.2f} Hz\n")

    fb, p_db, prom_db, df = whitened_psd(signal, fs, args.resolution, fmin, fmax)
    candidates = rank_eff_candidates(fb, p_db, prom_db, fmin, fmax, args.top_n)

    print(f"PSD resolution: {df:.4g} Hz\n")
    print(f"{'#':<3}{'freq (Hz)':>12}{'RPM':>10}{'prominence (dB)':>18}")
    for i, (f0, prom, pdb) in enumerate(candidates):
        rpm = rpm_of_eff(f0, args.cylinders, args.strokes)
        print(f"{i+1:<3}{f0:>12.3f}{rpm:>10.1f}{prom:>18.2f}")

    if candidates:
        print(f"\nStrongest candidate: {candidates[0][0]:.3f} Hz -> "
              f"{rpm_of_eff(candidates[0][0], args.cylinders, args.strokes):.1f} RPM")
    print("Note: broadband cavitation noise or other constant-speed machinery (e.g. an "
          "auxiliary generator) can produce peaks in this same band that are louder than "
          "the true, speed-dependent EFF line. Check the saved plot and the runner-up "
          "candidates, and cross-check against any known plausible RPM for this passage.")

    output = args.output
    if output is None and not args.show:
        import os
        stem = os.path.splitext(os.path.basename(args.mat_file))[0]
        output = f"{stem}_eff.png"

    plot_eff(fb, p_db, fmin, fmax, candidates, args.cylinders, args.strokes,
              output=output, show=args.show)


if __name__ == "__main__":
    sys.exit(main())
