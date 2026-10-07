# An Open-Source Distributed Acoustic Sensing Dataset and Benchmark for Perimeter-Security Event Recognition

**Five event classes · 4,000 aligned samples · Four representations · One fixed evaluation protocol**

Public DAS event-recognition datasets often focus on a particular data representation, with limited shared support for multiple common input forms. This makes it difficult for researchers to compare methods under a unified benchmark. We share this dataset to support the sensing community with **four aligned representations of the same measurements and a fixed evaluation protocol**. Together with baseline code and trained models, this resource provides a common starting point for developing, reproducing and comparing DAS perimeter-security event-recognition methods.

**Important Notes:**

- **Start small:** [train the feature-based SVM in Colab](https://colab.research.google.com/github/StepsEight/DAS-Perimeter-Benchmark/blob/main/notebooks/quickstart.ipynb) using the included CSV. No full dataset download or GPU is needed.
- **Get all representations:** the [dataset download (about 1.55 GB)](https://github.com/StepsEight/DAS-Perimeter-Benchmark/releases/tag/v1.0.2) contains the MAT files, STFT arrays, feature vectors, labels and fixed splits. Code and model checkpoints are in this repository.
- **Compare consistently:** use the provided train/validation/test splits and select models on the validation set. See the [benchmark protocol](docs/benchmark.md).
- **Access and licensing:** the complete dataset is publicly downloadable without registration, an access request, or author approval. Data and trained weights are licensed under [CC BY-NC 4.0](LICENSE-DATA), requiring attribution and noncommercial use. Supporting code is open-source under [MIT](LICENSE).

## 🚀 Overview

- [Experimental setup](#experimental-setup)
- [Dataset at a glance](#dataset)
- [Four aligned representations](#representations)
- [Get started](#quick-start)
- [Benchmark](#benchmark)
- [Citation and contact](#citation-and-contact)
- [License](#license)

<a id="experimental-setup"></a>
## 🔬 Experimental setup

The measurements were collected with a single-pulse phase-sensitive optical time-domain reflectometry (Φ-OTDR) DAS system and a kilometre-scale G.652 sensing fiber deployed on campus. The sensing fiber was buried approximately 5 cm below ground. The system uses a 10 m gauge length and 1 m spatial sampling interval. Background conditions and walking, digging and bicycle activities were recorded near the fiber. The released signals have a sampling rate of 2 kHz.

![DAS experimental system and field deployment from Figure 1 of the accompanying paper](assets/system.png)

*Figure 1. Experimental system and field deployment used to acquire the perimeter-security dataset.*

<a id="dataset"></a>
## 📊 Dataset at a glance

| Event | Description | Samples | Train / validation / test samples |
|---|---|---:|---:|
| Sunny background | Background without artificial disturbance in sunny weather | 800 | 480 / 160 / 160 |
| Rainy background | Background without artificial disturbance in rainy weather | 800 | 480 / 160 / 160 |
| Walking | One person walking beside the sensing fiber | 800 | 480 / 160 / 160 |
| Digging | One person digging near the fiber with a shovel | 800 | 480 / 160 / 160 |
| Bicycle | One person riding a bicycle around the sensing fiber | 800 | 480 / 160 / 160 |
| **Total** | | **4,000** | **2,400 / 800 / 800** |

Each space-time sample contains **2,000 time points × 50 spatial channels**, sampled at **2 kHz**. The five classes have **800 samples each**.

Each recording contributes **four samples from different fiber positions**. These four samples always stay together when recordings are split chronologically within each class: **60% training, 20% validation and 20% testing**. This prevents samples from the same recording from appearing in different sets. File formats and processing details are in the [dataset guide](docs/dataset.md).

<a id="representations"></a>
## 🧩 Four aligned representations

All representations describe the same 4,000 samples and use the same labels and splits.

| Representation | Shape | Public file | Baseline |
|---|---|---|---|
| Handcrafted features | 15 | [`data/handcrafted_features.csv`](data/handcrafted_features.csv) | RBF-SVM |
| Temporal waveform | 2,000 | `data/temporal/DAS_1D_2kHz_4000.mat` | 1D-CNN |
| Spatiotemporal patch | 2,000 × 50 | `data/spatiotemporal/{class}.mat` | 2D-CNN |
| Time-frequency map | 129 × 33 | `data/time_frequency/stft.npz` | STFT-CNN |

Match samples across files using `sample_id` in [`data/metadata.csv`](data/metadata.csv). In filenames and labels, **`vehicle` means bicycle**.

![Space-time, temporal and STFT examples for the five event classes](assets/representations.png)

*Examples of sunny background, rainy background, walking, digging and bicycle events. Rows show space-time, temporal and STFT views; the fourth representation is a 15-dimensional feature vector.*

<a id="quick-start"></a>
## 📥 Get started

### ☁️ Train an SVM in your browser

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StepsEight/DAS-Perimeter-Benchmark/blob/main/notebooks/quickstart.ipynb)

Run the notebook to inspect the feature CSV, train an RBF-SVM, select its parameters on the validation set, and evaluate the selected model on the test set. You can also download the full arrays to visualize the same sample in three signal representations.

### 💻 Work locally

Clone the repository or [download the code ZIP](https://github.com/StepsEight/DAS-Perimeter-Benchmark/archive/refs/heads/main.zip). For **SVM only**, no large data download is needed. Use Python 3.12 in a virtual environment:

```bash
git clone https://github.com/StepsEight/DAS-Perimeter-Benchmark.git
cd DAS-Perimeter-Benchmark
python -m pip install -r requirements-svm.txt
python scripts/train.py --model svm
```

For **MATLAB, signal visualization or the CNN baselines**, also [download the dataset](https://github.com/StepsEight/DAS-Perimeter-Benchmark/releases/download/v1.0.2/das-perimeter-v1.0.2.zip). Follow the [quick-start guide](docs/quickstart.md) for setup, automatic data installation, MATLAB/Python examples and checkpoint evaluation.

<a id="benchmark"></a>
## 🏁 Benchmark

All methods use the same samples and fixed splits. Models are selected by **validation Macro-F1**. The paper's **test results (%)** are:

| Method | Accuracy | Macro-P | Macro-R | Macro-F1 |
|---|---:|---:|---:|---:|
| RBF-SVM | 75.875 | 80.517 | 75.875 | 76.363 |
| 1D-CNN | 84.875 | 88.585 | 84.875 | 84.097 |
| 2D-CNN | 88.625 | 90.518 | 88.625 | 88.093 |
| STFT-CNN | **95.625** | **95.715** | **95.625** | **95.633** |

The [trained checkpoints](checkpoints/README.md), [per-sample predictions](results/predictions) and [metrics](results/summary.csv) are available for comparison. See the [benchmark guide](docs/benchmark.md) for architectures, hyperparameters, preprocessing and evaluation rules, or the [quick-start guide](docs/quickstart.md) to run them.

<a id="citation-and-contact"></a>
## 📚 Citation and contact

This repository accompanies **“An Open-Source Distributed Acoustic Sensing Dataset and Benchmark for Perimeter-Security Event Recognition”**, by **Yinghuan Li, Changyuan Yu, and Alan Pak Tao Lau**. Publication details will be added when available. Please cite the dataset version using [`CITATION.cff`](CITATION.cff), and identify the protocol and representation used in your work.

Questions, corrections and benchmark contributions are welcome through [GitHub Issues](https://github.com/StepsEight/DAS-Perimeter-Benchmark/issues).  
Contact: **Yinghuan Li**, The Hong Kong Polytechnic University, [ying-huan.li@connect.polyu.hk](mailto:ying-huan.li@connect.polyu.hk).  
See [contribution guidance](CONTRIBUTING.md) for reporting a comparable result.

<a id="license"></a>
## 📄 License

- **Code:** Licensed under [MIT](LICENSE).
- **Dataset, Model Checkpoints, and Documentation:** Licensed under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE-DATA). This also covers the released results and accompanying figures.

> **Note:** No separate permission is required for uses permitted by CC BY-NC 4.0. Commercial use is not permitted under this license, and the MIT code license does not grant commercial rights to the underlying data or trained weights. For separate commercial licensing inquiries, please contact: `ying-huan.li@connect.polyu.hk`.
