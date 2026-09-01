#!/usr/bin/env python3
"""Estimate shaft/engine RPM from a ship-radiated-noise .mat recording.

Method: DEMON analysis (DEtection of Modulation ON Noise). This is the
standard passive-acoustic technique for recovering a rotating machine's
rotation rate from broadband cavitation/flow noise:

  1. Bandpass-filter the raw signal into a broadband cavitation region
     (default 2-20 kHz), since propeller cavitation modulates broadband
     noise once per blade passage.
  2. Extract the envelope of that band (magnitude of the analytic signal).
  3. Low-pass filter + decimate the envelope down to a low sample rate,
     since the modulation of interest lives at a few Hz to a few tens of Hz.
  4. Compute the PSD of this envelope ("DEMON spectrum"): tonal peaks here
     correspond to the shaft rotation rate and/or blade-passage rate and
     their harmonics.
  5. Search over candidate fundamental frequencies for the one whose
     harmonic series best explains the strongest peaks in the DEMON
     spectrum, and report the top candidates ranked by that score.

Blind, fully-automatic RPM detection from noisy underwater recordings is
inherently ambiguous (harmonic/sub-harmonic confusion, masking by
broadband cavitation noise, multiple shafts running at close but
different speeds). This tool therefore reports a *ranked list* of
candidate fundamentals rather than a single trusted number, and always
saves the DEMON spectrum plot so a candidate can be visually confirmed
and cross-checked against what is known about the vessel (expected RPM
range, number of propeller blades, gearbox ratio).

Example:
    python3 rpm_estimate.py raw_6_MARS12_3_5.mat --rpm-min 300 --rpm-max 3000
"""

import argparse
import sys

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, find_peaks, butter, sosfiltfilt, hilbert, decimate

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

    meta = {k: v for k, v in d.items() if k not in (sig_key, fs_key)}
    return signal, fs, meta


def demon_envelope(signal, fs, band, env_lowpass, target_fs):
    """Bandpass -> envelope (Hilbert) -> lowpass -> decimate down to target_fs."""
    nyq = fs / 2
    hi = min(band[1], nyq * 0.98)
    lo = min(band[0], hi * 0.5)
    sos = butter(4, (lo, hi), btype="bandpass", fs=fs, output="sos")
    xb = sosfiltfilt(sos, signal)
    env = np.abs(hilbert(xb))

    sos_lp = butter(4, env_lowpass, btype="lowpass", fs=fs, output="sos")
    env = sosfiltfilt(sos_lp, env)

    fs_d = fs
    while fs_d / target_fs > 1.5:
        factor = int(min(10, round(fs_d / target_fs)))
        if factor < 2:
            break
        env = decimate(env, factor, ftype="iir", zero_phase=True)
        fs_d /= factor

    return env - env.mean(), fs_d


def demon_spectrum(env, fs_env, resolution):
    nperseg = int(fs_env / resolution)
    nperseg = int(2 ** round(np.log2(nperseg)))
    nperseg = max(64, min(nperseg, len(env)))
    noverlap = nperseg // 2
    freqs, psd = welch(env, fs=fs_env, window="hann", nperseg=nperseg,
                        noverlap=noverlap, scaling="density")
    return freqs, psd, fs_env / nperseg


def _comb_score(peak_f, peak_w, f0, n_harmonics, tol):
    score, matched = 0.0, 0
    for k in range(1, n_harmonics + 1):
        target = k * f0
        j = np.argmin(np.abs(peak_f - target))
        if abs(peak_f[j] - target) <= tol:
            score += peak_w[j] / np.sqrt(k)
            matched += 1
    return score, matched


def rank_candidates(freqs, psd, fmin, fmax, resolution, n_harmonics=6,
                     top_peaks=30, n_candidates_out=5, min_matches=3):
    """Rank candidate fundamentals by how well their harmonic series matches
    the strongest peaks of the DEMON spectrum."""
    mask = (freqs > fmin * 0.5) & (freqs < fmax * n_harmonics)
    f, p = freqs[mask], psd[mask]
    p_db = 10 * np.log10(p + 1e-30)

    pk_idx, _ = find_peaks(p_db, prominence=1.0)
    pf_all, pw_all = f[pk_idx], p_db[pk_idx] - np.median(p_db)
    order = np.argsort(pw_all)[::-1][:top_peaks]
    peak_f, peak_w = pf_all[order], np.clip(pw_all[order], 0.1, None)

    df = resolution
    cands = np.arange(fmin, fmax, df / 2)
    tol = max(df * 1.5, 0.05)

    scores = np.zeros_like(cands)
    matches = np.zeros_like(cands)
    for i, f0 in enumerate(cands):
        s, m = _comb_score(peak_f, peak_w, f0, n_harmonics, tol)
        scores[i], matches[i] = s, m

    order2 = np.argsort(scores)[::-1]
    ranked = []
    for i in order2:
        f0 = cands[i]
        if any(abs(f0 - r[0]) / r[0] < 0.05 for r in ranked):
            continue
        ranked.append((f0, scores[i], int(matches[i])))
        if len(ranked) >= n_candidates_out:
            break

    ranked.sort(key=lambda r: (-(r[2] >= min_matches), -r[1]))
    return ranked, peak_f, peak_w


def plot_demon(freqs, psd, candidates, gear_ratio, output=None, fmax_plot=None, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    psd_db = 10 * np.log10(np.maximum(psd, np.finfo(float).tiny))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs, psd_db, linewidth=0.8, color="steelblue")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if fmax_plot is None:
        fmax_plot = candidates[0][0] * 7 if candidates else (freqs[-1] if len(freqs) else 50)

    for rank, (f0, score, matches) in enumerate(candidates):
        c = colors[rank % len(colors)]
        k = 1
        while k * f0 <= fmax_plot:
            ax.axvline(k * f0, color=c, linestyle="--", linewidth=0.8, alpha=0.6)
            k += 1
        ax.axvline(f0, color=c, linewidth=1.6,
                    label=f"#{rank+1}: f0={f0:.3f} Hz -> {f0*60*gear_ratio:.1f} RPM "
                          f"(matches={matches})")

    ax.set_xlim(0, fmax_plot)
    plot_mask = freqs <= fmax_plot
    if plot_mask.any():
        lo, hi = np.percentile(psd_db[plot_mask], [1, 100])
        ax.set_ylim(lo - 3, hi + 5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("DEMON PSD (dB / Hz)")
    ax.set_title("DEMON (envelope) spectrum — candidate rotation rates")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"Saved DEMON spectrum plot to {output}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mat_file", help="Path to the .mat file containing the signal")
    parser.add_argument("--rpm-min", type=float, default=100,
                         help="Minimum plausible RPM to search for (default: 100)")
    parser.add_argument("--rpm-max", type=float, default=3000,
                         help="Maximum plausible RPM to search for (default: 3000)")
    parser.add_argument("--gear-ratio", type=float, default=1.0,
                         help="Multiply the estimated rotation frequency by this factor "
                              "before converting to RPM, to account for a reduction "
                              "gearbox between the detected modulation and the reported "
                              "engine RPM (default: 1.0)")
    parser.add_argument("--band-low", type=float, default=2000,
                         help="Lower edge of the cavitation/broadband band (Hz, default 2000)")
    parser.add_argument("--band-high", type=float, default=20000,
                         help="Upper edge of the cavitation/broadband band (Hz, default 20000)")
    parser.add_argument("--env-lowpass", type=float, default=200,
                         help="Envelope lowpass cutoff (Hz, default 200)")
    parser.add_argument("--resolution", type=float, default=0.15,
                         help="Target frequency resolution of the DEMON spectrum (Hz, default 0.15)")
    parser.add_argument("--n-harmonics", type=int, default=6,
                         help="Number of harmonics used when scoring a candidate fundamental")
    parser.add_argument("--top-n", type=int, default=5,
                         help="Number of ranked candidates to report")
    parser.add_argument("--output", "-o", default=None,
                         help="Path to save the DEMON spectrum plot (default: '<file stem>_demon.png')")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively")
    args = parser.parse_args(argv)

    signal, fs, meta = load_signal(args.mat_file)
    print(f"Loaded '{args.mat_file}': {len(signal)} samples, fs = {fs:g} Hz, "
          f"duration = {len(signal)/fs:.3g} s")

    env, fs_env = demon_envelope(signal, fs, (args.band_low, args.band_high),
                                  args.env_lowpass, target_fs=2.5 * args.env_lowpass * 5)
    freqs, psd, actual_res = demon_spectrum(env, fs_env, args.resolution)

    fmin = args.rpm_min / 60.0 / args.gear_ratio
    fmax = args.rpm_max / 60.0 / args.gear_ratio
    ranked, peak_f, peak_w = rank_candidates(freqs, psd, fmin, fmax, actual_res,
                                              n_harmonics=args.n_harmonics,
                                              n_candidates_out=args.top_n)

    print(f"\nSearch range: {args.rpm_min:g}-{args.rpm_max:g} RPM "
          f"(fundamental {fmin:.3f}-{fmax:.3f} Hz, gear ratio {args.gear_ratio:g})")
    print(f"DEMON spectrum resolution: {actual_res:.4g} Hz\n")
    print(f"{'#':<3}{'f0 (Hz)':>10}{'RPM':>10}{'score':>10}{'harmonics matched':>20}")
    for i, (f0, score, matches) in enumerate(ranked):
        rpm = f0 * 60 * args.gear_ratio
        print(f"{i+1:<3}{f0:>10.3f}{rpm:>10.1f}{score:>10.2f}{matches:>20d}")

    if ranked:
        best_f0, best_score, best_matches = ranked[0]
        print(f"\nBest candidate: {best_f0:.3f} Hz -> {best_f0*60*args.gear_ratio:.1f} RPM "
              f"({best_matches} harmonics matched)")
        print("Note: verify against the plotted DEMON spectrum -- with close-running "
              "twin shafts or heavy cavitation masking, the top-ranked candidate is not "
              "always the correct one; check the runner-up candidates too.")

    output = args.output
    if output is None and not args.show:
        import os
        stem = os.path.splitext(os.path.basename(args.mat_file))[0]
        output = f"{stem}_demon.png"

    plot_demon(freqs, psd, ranked, args.gear_ratio, output=output,
               fmax_plot=fmax * (args.n_harmonics + 1), show=args.show)


if __name__ == "__main__":
    sys.exit(main())
