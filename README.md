# DAS Perimeter Benchmark

**Five event classes · 4,000 aligned samples · Four representations · One fixed evaluation protocol**

A shared distributed acoustic sensing (DAS) dataset and reproducible benchmark for perimeter-security event recognition.

- **Share data:** provide an integrated online resource for the development of DAS perimeter-security applications.
- **Compare algorithms:** provide aligned inputs, fixed partitions and common metrics to support the development and optimization of event-recognition methods.

[Download data](https://github.com/StepsEight/DAS-Perimeter-Benchmark/releases/tag/v1.0.0) · [Dataset guide](docs/dataset.md) · [Benchmark protocol](docs/benchmark.md) · [Citation](#citation-and-contact)

![Space-time, temporal and time-frequency examples for the five event classes](assets/representations.png)

*The paper's illustrative samples. Rows show space-time, temporal and STFT representations; all four representations, including handcrafted features, share the same sample IDs.*

## Dataset at a glance

| Event | Description | Samples | Train / validation / test |
|---|---|---:|---:|
| Sunny background | Background without artificial disturbance in sunny weather | 800 | 480 / 160 / 160 |
| Rainy background | Background without artificial disturbance in rainy weather | 800 | 480 / 160 / 160 |
| Walking | One person walking beside the sensing fiber | 800 | 480 / 160 / 160 |
| Digging | One person digging near the fiber with a shovel | 800 | 480 / 160 / 160 |
| Bicycle | One person riding a bicycle around the sensing fiber | 800 | 480 / 160 / 160 |
| **Total** | | **4,000** | **2,400 / 800 / 800** |

The released sampling rate is **2 kHz**. Each space-time sample has **2,000 time points × 50 spatial channels**, with 1 m channel spacing and a 10 m gauge length. The fixed 1 s input includes **41 leading zeros and 1,959 measured points**. Single-channel representations use the fixed 25th channel of each patch.

The split is chronological within each class. All four spatial patches from a source recording stay in the same subset. There are 1,000 source-recording groups, split 600 / 200 / 200. This prevents patches from the same recording from crossing partitions; independent physical-event or session boundaries are not available.

## Four aligned representations

Sizes below describe one sample, before adding a CNN feature-plane dimension.

| Representation | Shape | Public file | Baseline |
|---|---|---|---|
| Handcrafted features | 15 | [`data/handcrafted_features.csv`](data/handcrafted_features.csv) | RBF-SVM |
| Temporal waveform | 2,000 | `data/temporal/DAS_1D_2kHz_4000.mat` | 1D-CNN |
| Spatiotemporal patch | 2,000 × 50 | `data/spatiotemporal/{class}.mat` | 2D-CNN |
| Time-frequency map | 129 × 33 | `data/time_frequency/stft.npz` | STFT-CNN |

[`data/metadata.csv`](data/metadata.csv) aligns every representation by `sample_id` and includes class labels, source-recording groups, spatial-channel ranges and fixed split membership. Internal labels are `background`, `digging`, `raining`, `vehicle`, `walking`; **`vehicle` means bicycle** in this dataset.

## Quick start

Use Python 3.12. Create and activate a virtual environment, then:

```bash
git clone https://github.com/StepsEight/DAS-Perimeter-Benchmark.git
cd DAS-Perimeter-Benchmark
python -m pip install -r requirements-data.txt
python scripts/download_data.py
python scripts/verify_data.py
python examples/read_sample.py --sample-id walking_000241
```

The downloader retrieves the versioned [Release ZIP](https://github.com/StepsEight/DAS-Perimeter-Benchmark/releases/download/v1.0.0/das-perimeter-v1.0.0.zip), verifies SHA-256 and installs the data into the paths above. Alternatively, download the ZIP manually and run `python scripts/download_data.py --archive /path/to/das-perimeter-v1.0.0.zip`. GitHub's automatic **Source code** ZIP contains code and metadata; the **dataset ZIP** is a separate Release asset.

For MATLAB, run [`examples/read_sample.m`](examples/read_sample.m) from the repository root. Both temporal and spatiotemporal files are native MATLAB v7.3 MAT files. Full shapes, axis conventions and normalization are documented in the [dataset guide](docs/dataset.md).

## Benchmark

All methods use the same sample IDs and split. Results below are the paper's saved **test** results, in percent. Model selection uses **validation Macro-F1**, with each class weighted equally and both precision and recall considered.

| Method | Accuracy | Macro-P | Macro-R | Macro-F1 |
|---|---:|---:|---:|---:|
| RBF-SVM | 75.875 | 80.517 | 75.875 | 76.363 |
| 1D-CNN | 84.875 | 88.585 | 84.875 | 84.097 |
| 2D-CNN | 88.625 | 90.518 | 88.625 | 88.093 |
| STFT-CNN | **95.625** | **95.715** | **95.625** | **95.633** |

Train the four baselines or evaluate a released checkpoint:

```bash
python -m pip install -r requirements.txt
python scripts/train.py --model all
python scripts/evaluate.py --model stft_cnn --split test
```

Training settings are in [`config.yaml`](config.yaml). Runs are written to `runs/`. The released [best checkpoints](checkpoints/README.md), [per-sample predictions](results/predictions) and [machine-readable metrics](results/summary.csv) are provided. To regenerate and check the single-channel representations from the five MAT files, run `python scripts/preprocess.py --check`.

The CNNs were trained for 100 epochs with seed 42. The published results use one fixed chronological split and one seed; they compare representation–model combinations, not a controlled representation-only ablation. See the [benchmark protocol](docs/benchmark.md) for architectures, normalization, hyperparameters and evaluation details.

## Repository layout

```text
data/           Sample metadata, fixed splits, features and downloaded arrays
src/            Four baseline implementations and preprocessing functions
scripts/        Download, verify, preprocess, train and evaluate
examples/       Python and MATLAB data readers
checkpoints/    Best validation-selected models
results/        Paper metrics, predictions and training histories
docs/           Dataset and benchmark specifications
assets/         The paper's system and representation figures
```

## Citation and contact

This repository accompanies **“An Open-Source Distributed Acoustic Sensing Dataset and Benchmark for Perimeter-Security Event Recognition”**, by **Yinghuan Li, Jingming Zhang, Changyuan Yu, and Alan Pak Tao Lau**. Publication details will be added when available. Please cite the dataset version using [`CITATION.cff`](CITATION.cff), and identify the protocol and representation used in your work.

Questions, corrections and benchmark contributions are welcome through [GitHub Issues](https://github.com/StepsEight/DAS-Perimeter-Benchmark/issues). Contact: **Yinghuan Li**, The Hong Kong Polytechnic University, [ying-huan.li@connect.polyu.hk](mailto:ying-huan.li@connect.polyu.hk). See [contribution guidance](CONTRIBUTING.md) for reporting a comparable result.

## License

**Code:** [MIT](LICENSE). **Dataset, released trained weights, results and accompanying figures/documentation:** [CC BY-NC 4.0](LICENSE-DATA), requiring attribution and noncommercial use. The MIT code license does not override the data license. Commercial data-use requests should be directed to the contact above.

The organization of this resource was informed by [Bristol's multi-domain monitoring and sensing data platform](https://github.com/hpn-bristol/Open-Source-Data-for-Multi-Domain-Network-Monitoring-and-Sensing-in-Optical-and-Wireless-Networks) and [TrafficMonitoringDAS](https://github.com/StepsEight/TrafficMonitoringDAS).
