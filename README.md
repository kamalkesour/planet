# PSD Viewer

Compute and plot the Power Spectral Density (PSD) of a signal stored in a
`.mat` file, at a user-specified frequency resolution.

The script auto-detects a 1-D signal array (`signal`, `x`, `data`, or `y`)
and a sample rate (`fs`, `Fs`, `FS`, `fsamp`, or `sample_rate`) in the
`.mat` file. Any other scalar variables present (e.g. `device`, `passage`,
`track`, `sog`, `rpm_babord`, `rpm_tribord`) are shown as metadata on the
plot.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 psd.py raw_2_MARS12_3_1.mat --resolution 1.0 --output psd.png
```

Key options:

- `--resolution` / `-r`: target frequency resolution (df, in Hz) for the
  Welch PSD estimate. Internally this sets the FFT segment length
  `nperseg = fs / resolution` (rounded to the nearest power of two). The
  actual achieved resolution is printed and used in the plot title.
- `--fmin`, `--fmax`: restrict the plotted frequency range.
- `--xlog`: use a logarithmic frequency axis.
- `--window`: window function name (default `hann`).
- `--overlap`: fractional segment overlap (default `0.5`).
- `--output` / `-o`: output image path (default `<file>_psd.png`).
- `--show`: display the plot interactively instead of/in addition to saving.
- `--csv`: also write the PSD (frequency, psd, psd_db) to a CSV file.

## Example

For a 51.2 kHz hydrophone recording, a 1 Hz resolution PSD:

```bash
python3 psd.py raw_2_MARS12_3_1.mat -r 1.0 --fmax 20000 -o psd_1hz.png
```

A finer, 0.1 Hz resolution view of the low-frequency band with a log axis:

```bash
python3 psd.py raw_2_MARS12_3_1.mat -r 0.1 --fmax 2000 --xlog -o psd_0.1hz.png
```
