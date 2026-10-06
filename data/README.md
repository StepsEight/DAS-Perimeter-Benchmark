# Data files

The release has **4,000 samples**, five classes and four aligned representations. Download the large arrays with:

```bash
python scripts/download_data.py
```

Run this command from the repository root. The download is a versioned GitHub Release asset; the ordinary Git checkout keeps metadata and handcrafted features small and accessible.

| Path | Content |
|---|---|
| `metadata.csv` | All sample IDs, labels, source groups, spatial channels and split memberships |
| `class_mapping.json` | Fixed numeric labels and paper display names |
| `splits/{train,val,test}.csv` | Fixed sample IDs and 0-based global row indices |
| `handcrafted_features.csv` | Aligned 15-dimensional feature vectors |
| `spatiotemporal/*.mat` | Five class files, MATLAB single [2000,50,800] each |
| `temporal/DAS_1D_2kHz_4000.mat` | Self-contained fixed-channel waveforms and metadata |
| `time_frequency/stft.npz` | Float32 [4000,129,33] log-power STFT maps and axes |
| `manifest.json` | Exact sizes, SHA-256 checksums and the versioned download URL |

Data are licensed under [CC BY-NC 4.0](../LICENSE-DATA). See the [dataset guide](../docs/dataset.md) for MATLAB/HDF5 axis conventions, inverse min–max handling, zero padding and normalization.
