# Best validation-selected models

| File | Model | Best epoch | Validation Macro-F1 |
|---|---|---:|---:|
| `svm.joblib` | RBF-SVM, C=100 and gamma=scale | — | 0.8840618504 |
| `cnn1d.pt` | 1D-CNN | 100 | 0.9862176527 |
| `cnn2d.pt` | 2D-CNN | 10 | 0.9987499878 |
| `stft_cnn.pt` | STFT-CNN | 79 | 0.9685727428 |

CNN files contain `model_state`, `method`, `class_names`, `epoch`, `parameter_count` and validation metrics. They support `torch.load(path, map_location='cpu', weights_only=True)`. They are inference checkpoints; newly trained runs save full resumable state in `runs/`.

The SVM file contains `model`, `scaler`, `class_names` and selected parameters. Its scaler was fitted on training data only. Use joblib files from this trusted release, since the format uses Python pickle.

All files share label order `background, digging, raining, vehicle, walking`. Model hashes and source-weight equivalence are recorded in [`manifest.json`](manifest.json). Example:

```bash
python scripts/evaluate.py --model stft_cnn --split test
```

Released trained weights are covered by [CC BY-NC 4.0](../LICENSE-DATA).
