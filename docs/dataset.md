# Dataset guide

This release contains exactly the five classes and 4,000 samples used in the associated paper. Every representation shares the metadata row order and `sample_id`. The machine-readable class order is **background, digging, raining, vehicle, walking** (IDs 0–4); `vehicle` denotes bicycle.

## Acquisition and spatial grouping

The manuscript describes a single-pulse phase-sensitive OTDR system with a kilometer-scale G.652 single-mode fiber deployed on campus and buried approximately 5 cm below the ground surface. Spatial sampling is 1 m and gauge length is 10 m. Neighboring channels have overlapping gauge intervals.

![Experimental system and event collection, reproduced from the associated manuscript](../assets/system.png)

Each source recording supplies four simultaneous 50-channel patches: acquisition channels **80–129, 130–179, 180–229, and 230–279**. These are four spatial views of one recording, not four verified independent physical events. The fixed relative 25th channel corresponds to acquisition channels 104, 154, 204 and 254.

Within each class, 200 chronologically ordered source recordings give 800 instances. Instance indices 1–4 share the first source recording, 5–8 the second, and so on. The first 120 recordings form training, the next 40 validation, and the final 40 testing. These partitions have no overlapping source groups or source-recording hashes and no additional time gap. Nearby recordings may remain correlated; trial, subject and session independence has not been established.

Acquisition times are included in the released metadata. Sample ordering does not imply uninterrupted one-second acquisition.

## Sampling, padding and amplitude

The released MAT files are ready to use at **2 kHz**. All reading, preprocessing and benchmark commands start from the public files listed below.

Each sample contains **1,959 measured points** after a **41-point zero prefix**, giving a fixed 2,000-point input. The tensor length corresponds to 1 s; valid point count divided by sample rate is 0.9795 s. The leading zeros are not measurements and must be excluded when computing waveform or patch normalization statistics. Filtering during acquisition processing may affect the final approximately 34 points; no additional removal is applied by the benchmark.

For storage, each valid 1959×50 patch was min–max scaled to [0,1] in double precision and saved as float32. Its original minimum and maximum are retained. Restoring with `stored*(maximum-minimum)+minimum` approximately recovers the filtered ADC amplitude, not calibrated strain, displacement or acceleration. The prefix must be reset to zero after restoration.

## Files and array conventions

All large arrays are distributed in the [v1.0.2 Release](https://github.com/StepsEight/DAS-Perimeter-Benchmark/releases/tag/v1.0.2). [`data/manifest.json`](../data/manifest.json) records file sizes and SHA-256 values. The repository's commands use these public files through relative paths.

### Spatiotemporal: five class MAT files

`data/spatiotemporal/background.mat`, `digging.mat`, `raining.mat`, `vehicle.mat`, `walking.mat` each contain exactly 800 instances.

| Variable | MATLAB shape | Meaning |
|---|---|---|
| `instances` | 2000×50×800, single | Time × space × instance; min–max storage values |
| `sample_ids` | 800×1 cell | Stable ID matching metadata |
| `instance_index` | 800×1 | 1-based chronological index within this class |
| `group_id` | 800×1 | Source-recording group, shared by four patches |
| `source_bin_index` | 800×1 | 1-based source-recording index within the class |
| `channel_start`, `channel_end` | 800×1 | Original inclusive spatial-channel indices |
| `normalization_min`, `normalization_max` | 800×1 | Inverse min–max metadata |
| `valid_data_mask` | 2000×1 | False for the first 41 rows; true for the remaining rows |
| `split`, `split_code` | 800×1 | train/val/test and codes 1/2/3 |
| `train_indices`, `val_indices`, `test_indices` | 480/160/160×1 | MATLAB indices within this file |
| `bin_filenames`, `bin_sha256`, `bin_acquisition_datetime` | 200×1 | Source-recording provenance |

MATLAB v7.3 reverses dimension order in the HDF5 view: `h5py` sees `instances` as **800×50×2000**. The CNN receives **[batch,1,50,2000]** after amplitude restoration and valid-region global standardization. MATLAB cell strings use HDF5 references; use the provided readers rather than treating them as numeric arrays.

### Temporal: one self-contained MAT file

`data/temporal/DAS_1D_2kHz_4000.mat` stores the fixed 25th-channel waveform, already z-score standardized over its 1,959 valid points. The first 41 points remain zero.

- MATLAB `waveforms`: **4000×2000**, single; Python `h5py` reads this variable as 2000×4000, so transpose it.
- MATLAB `instances`: **2000×1×4000**, single, an equivalent view following the class-MAT convention.
- `sample_ids`, `class_names`, `labels`, `group_id`, `split` and source/channel metadata support independent use.
- `labels`/`class_index` are **0-based**; `label_id` and MATLAB split indices are **1-based**.
- Do not invert this file's waveforms using min/max alone: its waveforms have also been z-score standardized. Min/max describes source storage; `standardization_mean` and `standardization_std` describe the waveform transformation.

### Time-frequency: one NPZ file

`data/time_frequency/stft.npz` contains `stft` (float32 **4000×129×33**), `sample_ids`, `labels`, `frequencies_hz` and `times_seconds`. It is loaded using `numpy.load(..., allow_pickle=False)`.

STFT uses the standardized temporal waveform, a periodic Hann window of 256 points, 192-point overlap, 64-point hop, FFT length 256, no detrending, and SciPy `boundary='zeros', padded=True`. The one-sided frequency axis contains 129 bins from 0 to 1000 Hz at 7.8125 Hz intervals. There are 33 frame centers from 0 to 1.024 s because of boundary extension and final-frame padding.

The input is `ln(abs(Z)**2+1e-12)`, standardized with one mean/std over the complete matrix and denominator epsilon 1e-8. It uses the SciPy STFT spectrum scaling convention, without additional interior-bin power doubling. No image resizing, RGB encoding or display color clipping is applied to model inputs.

### Handcrafted features: one CSV file

[`data/handcrafted_features.csv`](../data/handcrafted_features.csv) contains `sample_id`, `class_index`, `split` and these 15 feature columns:

`rms`, `standard_deviation`, `mean_absolute_value`, `peak_absolute_amplitude`, `peak_to_peak_amplitude`, `skewness`, `kurtosis`, `zero_crossing_rate`, `crest_factor`, `dominant_frequency_hz`, `spectral_centroid_hz`, `spectral_spread_hz`, `spectral_entropy`, `relative_energy_0_500_hz`, `relative_energy_500_1000_hz`.

Features use the amplitude-restored 25th-channel waveform before per-waveform z-score, including the 41 prefix zeros. Welch uses a 256-point Hann window, **128-point overlap**, constant detrending and a one-sided density PSD. Standard deviation is population std; kurtosis is Pearson kurtosis; skewness/kurtosis use `bias=True`. Zero crossings are counted between successive nonzero values and divided by 1999. Spectral entropy uses PSD-bin weights and is divided by ln(129); relative band energies use trapezoidal integration with interpolated band edges. Feature scaling for the SVM is fitted on training data only.

## Metadata and fixed split files

[`data/metadata.csv`](../data/metadata.csv) is the primary index. `row_index` is 0-based and aligns the combined waveform, STFT and feature arrays; `instance_index` is 1-based within each class MAT. Rows are grouped by the fixed class ID, then by acquisition order. `spatiotemporal_file` is relative to the repository root. `group_id`, `source_bin_index`, `bin_sha256` and `channel_start/end` identify related spatial views. Fields with a `bin_` prefix are retained provenance identifiers in the public metadata, not paths to files users need to obtain or process.

[`data/splits/train.csv`](../data/splits/train.csv), [`val.csv`](../data/splits/val.csv) and [`test.csv`](../data/splits/test.csv) contain the official `sample_id` and `row_index` memberships. Use these files across all representations. Training-batch shuffling is allowed within the training subset; randomly reassigning individual patches changes the benchmark and can leak same-recording information.
