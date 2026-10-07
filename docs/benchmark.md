# Benchmark protocol

The benchmark uses the five classes, four aligned representations and fixed chronological source-group split described in the [dataset guide](dataset.md). The reference results use seed 42. For installation and training commands, see the [quick-start guide](quickstart.md).

## Feature-only SVM workflow

The feature CSV, labels and fixed splits are included in the repository. SVM training and evaluation need only `requirements-svm.txt`; no MAT/NPZ download or PyTorch installation is required. The [Colab notebook](https://colab.research.google.com/github/StepsEight/DAS-Perimeter-Benchmark/blob/main/notebooks/quickstart.ipynb) runs this same training implementation and saves its selected model and results separately from the reference checkpoints.

## Models

Every CNN has convolutional widths 32/64/128, BatchNorm and ReLU after each convolution, global average pooling, Dropout(0.3), and a linear 128→5 logits layer. Models are initialized from scratch using the PyTorch layer defaults. There are no residual connections or pretrained backbones.

The dimensions below exclude batch; k, s and p denote kernel, stride and padding. Block outputs include their listed pooling.

| Model / block | Convolution (in→out; k; s; p) | MaxPool | Output |
|---|---|---|---|
| 1D-CNN input | — | — | 1×2000 |
| Block 1 | 1→32; 7; 2; 3 | 2 | 32×500 |
| Block 2 | 32→64; 5; 1; 2 | 2 | 64×250 |
| Block 3 | 64→128; 3; 1; 1 | 2 | 128×125 |
| 2D-CNN input | — | — | 1×50×2000 |
| Block 1 | 1→32; (3,9); (1,2); (1,4) | (2,2) | 32×25×500 |
| Block 2 | 32→64; (3,7); (1,1); (1,3) | (2,2) | 64×12×250 |
| Block 3 | 64→128; (3,5); (1,1); (1,2) | (2,2) | 128×6×125 |
| STFT-CNN input | — | — | 1×129×33 |
| Block 1 | 1→32; (3,3); (1,1); (1,1) | (2,2) | 32×64×16 |
| Block 2 | 32→64; (3,3); (1,1); (1,1) | (2,2) | 64×32×8 |
| Block 3 | 64→128; (3,3); (1,1); (1,1) | None | 128×32×8 |

Trainable parameter counts, including convolution/linear biases and BatchNorm affine parameters, are **36,357**, **168,069** and **93,765**, respectively. BatchNorm running statistics are buffers, not trainable parameters. The two 2D networks have different kernels, strides and pooling schedules despite sharing their widths and classifier head.

The RBF-SVM uses the 15 released handcrafted features after a training-only StandardScaler fit. The 20-candidate validation grid is C={0.1,1,10,100} × gamma={scale,0.001,0.01,0.1,1}. Validation Macro-F1 selected **C=100, gamma=scale**, with effective gamma approximately 1/15 and 711 support vectors. No class weights or probability calibration are used. Ties retain the first grid candidate. The final SVM is fitted only on training data.

## Input normalization

- **1D-CNN:** one mean/std over each waveform's valid 1,959 points; 41 prefix zeros remain zero.
- **2D-CNN:** one mean/std over each valid 1959×50 patch, followed by transposition to space×time. The original implementation restores amplitude to float32 before computing mean/std in float64; the released loader preserves this round trip.
- **STFT-CNN:** the same standardized waveform as 1D-CNN, then natural-log power STFT and one mean/std over the entire 129×33 map.
- **RBF-SVM:** restored amplitude statistics, without waveform z-score; feature-wise StandardScaler fitted on the 2,400 training instances.

All z-score denominators add 1e-8. None of the sample-wise transformations requires population statistics from validation or test data.

## Training and selection

| Setting | Value |
|---|---|
| Epochs | 100 for each CNN |
| Batch size | 32 |
| Optimizer | AdamW, betas=(0.9,0.999), eps=1e-8 |
| Initial learning rate | 1e-3 |
| Weight decay | 1e-4, one group containing all parameters |
| Learning-rate schedule | CosineAnnealingLR, T_max=100, eta_min=0 |
| Loss | Unweighted cross-entropy, no label smoothing |
| Dropout | 0.3, classifier head only |
| Augmentation / early stopping / gradient clipping | None |
| Seed | 42 |
| Precision | float16 CUDA AMP and GradScaler; float32 weights |
| Reproducibility settings | Deterministic algorithms, cuDNN benchmark disabled, TF32 disabled |
| Loader | Training shuffle only; num_workers=0, drop_last=False, pin_memory=True |
| CPU threads | 4 |
| Checkpoint rule | Highest validation Macro-F1; earliest epoch wins ties |

The selected epochs are **100 (1D-CNN), 10 (2D-CNN), and 79 (STFT-CNN)**. Every model still completed all 100 training epochs. After selection, the best checkpoint was loaded for the final test evaluation; validation data were not added to training. Training-history accuracy accumulates predictions during weight updates in training mode, rather than measuring a frozen checkpoint on all training data.

Original environment: Python 3.12.13, PyTorch 2.11.0+cu128, CUDA 12.8, scikit-learn 1.7.2, NumPy 2.3.5, SciPy 1.16.3, Windows 11, NVIDIA GeForce RTX 5070 Ti Laptop GPU. See [`results/environment.json`](../results/environment.json). Numerical behavior can vary across devices and library versions; use the recorded environment when comparing exact reproduction.

## Metrics and saved evidence

Accuracy is the proportion of correct predictions. Macro-P and Macro-R average per-class precision and recall across the fixed five classes. Macro-F1 averages the five per-class F1 values, where F1=2TP/(2TP+FP+FN). It is not the harmonic mean of Macro-P and Macro-R. Undefined per-class metrics use zero, consistently with the supplied implementation. Since validation/test are balanced, Macro-R equals Accuracy here.

[`results/summary.csv`](../results/summary.csv) stores metrics as fractions in [0,1]; the README table displays percentages. [`results/metrics.json`](../results/metrics.json) includes per-class metrics. [`results/predictions`](../results/predictions) contains all validation/test predictions, and [`results/history`](../results/history) records the training histories and SVM validation grid. To audit the published scores without running the models, use:

```bash
python scripts/verify_results.py
```

The release was checked for exact representation regeneration, MATLAB/Python readability, unchanged best-model weights, and matching sampled validation predictions. See [`results/release_verification.json`](../results/release_verification.json). Small protocol tests can be run without the large data download:

```bash
python -m unittest discover -s tests -v
```

The benchmark compares complete representation–model pipelines with different capacities and preprocessing. It does not establish a causal advantage of one representation or cross-session generalization. New algorithms should use the fixed partitions and validation-only selection, report their seed and input representation, and keep test data out of development decisions.
