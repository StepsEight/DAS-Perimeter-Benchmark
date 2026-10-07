# Data files

The dataset has **4,000 samples across five classes**. All four representations share sample IDs, labels and fixed train/validation/test splits.

## Included in the repository

These small files are ready to use without downloading the MAT/NPZ arrays:

| Path | Content |
|---|---|
| `handcrafted_features.csv` | Fifteen features per sample for the RBF-SVM |
| `metadata.csv` | Sample IDs, labels, spatial channels, recording groups and split membership |
| `class_mapping.json` | Numeric labels and event names |
| `splits/{train,val,test}.csv` | Fixed sample IDs and array row indices |

[Train the SVM in Colab](https://colab.research.google.com/github/StepsEight/DAS-Perimeter-Benchmark/blob/main/notebooks/quickstart.ipynb), or follow the [local quick start](../docs/quickstart.md).

## Available in the dataset download

| Path | Content |
|---|---|
| `spatiotemporal/*.mat` | Five class files, MATLAB single [2000,50,800] each |
| `temporal/DAS_1D_2kHz_4000.mat` | Combined fixed-channel waveforms and metadata |
| `time_frequency/stft.npz` | Float32 [4000,129,33] STFT maps, axes and IDs |

Install the arrays from the repository root:

```bash
python scripts/download_data.py
```

The [dataset ZIP](https://github.com/StepsEight/DAS-Perimeter-Benchmark/releases/tag/v1.0.2) also contains the small files listed above, a root `LICENSE` and `README.txt` for standalone use. It does not contain code or model checkpoints. [`manifest.json`](manifest.json) records sizes, SHA-256 checksums and the download URL.

See the [dataset guide](../docs/dataset.md) for array axes, normalization, padding and metadata fields. Data are licensed under [CC BY-NC 4.0](../LICENSE-DATA).
