# Quick start

Choose the route that matches your task:

| Task | What you need |
|---|---|
| Explore features and train an SVM | Repository only; the feature CSV and fixed splits are included |
| Read MAT files, visualize signals or run CNNs | Repository plus the approximately 1.55 GB dataset download |

## Browser: Colab

[Open the notebook](https://colab.research.google.com/github/StepsEight/DAS-Perimeter-Benchmark/blob/main/notebooks/quickstart.ipynb) and run the cells in order. It trains the SVM on a CPU, selects parameters using validation Macro-F1, and evaluates the selected model on the test set. It saves the model, scaler, metrics and predictions in `runs/colab_svm/svm/` and offers a ZIP download of these outputs.

The full-array download is optional: set `DOWNLOAD_FULL_DATA = True` to plot the selected sample as space-time, waveform and STFT views. Allow about 3.2 GB free runtime storage for the archive and extracted files. Colab storage is temporary; download any results you want to keep. The notebook uses available runtime libraries; use the pinned local environment below when comparing exact numerical results.

## Local setup

Use Python 3.12. Clone the repository, then create and activate a virtual environment:

```bash
git clone https://github.com/StepsEight/DAS-Perimeter-Benchmark.git
cd DAS-Perimeter-Benchmark
python -m venv .venv
```

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

### Feature-based SVM: no large download

```bash
python -m pip install -r requirements-svm.txt
python scripts/train.py --model svm
```

This reads `data/handcrafted_features.csv`, verifies its alignment with the metadata and fixed splits, fits feature scaling on training data only, and compares the 20 configured SVM settings on validation data. The selected training-only fit is evaluated once on the test set. Outputs are saved under `runs/svm/`; rerunning a completed configuration reuses its saved result.

To evaluate the provided SVM checkpoint instead of training:

```bash
python scripts/evaluate.py --model svm --split test
```

This uses `checkpoints/svm.joblib` and saves outputs under `runs/evaluation/svm_test/`. For a separate evaluation, supply a new empty directory with `--output`. PyTorch is not required for either SVM command.

### Full arrays: automatic installation

```bash
python -m pip install -r requirements-data.txt
python scripts/download_data.py
python scripts/verify_data.py
python examples/read_sample.py --sample-id walking_000241
```

The downloader verifies SHA-256 and installs the arrays in the expected paths. If you already have the [dataset ZIP](https://github.com/StepsEight/DAS-Perimeter-Benchmark/releases/download/v1.0.1/das-perimeter-v1.0.1.zip), use:

```bash
python scripts/download_data.py --archive /path/to/das-perimeter-v1.0.1.zip
```

For standalone MATLAB/data use, extract the ZIP into a **separate folder** and read its root `README.txt` and `LICENSE`. Avoid manually extracting over the code repository: the archive's `LICENSE` covers data under CC BY-NC 4.0, while the repository's `LICENSE` covers code under MIT. The official downloader handles this distinction automatically.

The dataset ZIP contains the five 2D MAT files, one combined 1D MAT, STFT arrays, feature CSV, metadata, labels, fixed splits and license/readme files. Code, checkpoints and benchmark results are in the repository.

### MATLAB

After installing the data into the repository, run [`examples/read_sample.m`](../examples/read_sample.m). It loads a space-time sample and its matching waveform. Both MAT formats use MATLAB v7.3. Array axes and normalization are specified in the [dataset guide](dataset.md).

### CNN training and checkpoint evaluation

Install the full dependencies after downloading the arrays:

```bash
python -m pip install -r requirements.txt
python scripts/train.py --model all
```

Use `--model cnn1d`, `--model cnn2d` or `--model stft_cnn` to train just one CNN. Each run saves the checkpoint with the highest validation Macro-F1 under `runs/`. The settings are in [`config.yaml`](../config.yaml).

To evaluate a released checkpoint without training:

```bash
python scripts/evaluate.py --model stft_cnn --split test
```

The recorded software/device environment is in [`results/environment.json`](../results/environment.json); hardware and library differences can affect exact results. For detailed model selection and normalization rules, see the [benchmark protocol](benchmark.md).

## Check or regenerate the representations

To compare features, waveforms and STFT maps regenerated from the five public MAT files with the released versions:

```bash
python scripts/preprocess.py --check
```

To verify saved benchmark scores without running models:

```bash
python scripts/verify_results.py
```

Keep the provided splits when reporting comparable results. See [contribution guidance](../CONTRIBUTING.md) for what to report.
