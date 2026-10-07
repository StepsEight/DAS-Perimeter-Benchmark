DAS Perimeter Benchmark - standalone dataset package v1.0.2

An Open-Source Distributed Acoustic Sensing Dataset and Benchmark for
Perimeter-Security Event Recognition
Authors: Yinghuan Li, Changyuan Yu, Alan Pak Tao Lau
Contact: ying-huan.li@connect.polyu.hk
Repository: https://github.com/StepsEight/DAS-Perimeter-Benchmark
Version: https://github.com/StepsEight/DAS-Perimeter-Benchmark/releases/tag/v1.0.2

LICENSE AND USE
All contents of this archive are licensed under Creative Commons Attribution-
NonCommercial 4.0 International (CC BY-NC 4.0). See the full LICENSE at the
archive root, or https://creativecommons.org/licenses/by-nc/4.0/ .
Give appropriate credit, link the license, and indicate any changes.
Commercial use is not permitted under this license. For commercial licensing
inquiries, contact the author above. The repository's MIT CODE license does
not grant commercial rights to this dataset or the trained model weights.

CONTENTS
data/spatiotemporal/{background,digging,raining,vehicle,walking}.mat
  Five MATLAB v7.3 files; instances: [2000 time, 50 channels, 800 samples]
  in MATLAB, [800 samples, 50 channels, 2000 time] via Python h5py.
data/temporal/DAS_1D_2kHz_4000.mat
  MATLAB waveforms: [4000 samples, 2000 time]; transpose h5py's waveforms array.
data/time_frequency/stft.npz
  Float32 stft: [4000 samples, 129 frequency bins, 33 frames]; includes axes/IDs.
data/handcrafted_features.csv
  Fifteen aligned features per sample, used by the RBF-SVM baseline.
data/metadata.csv, data/class_mapping.json, data/splits/{train,val,test}.csv
  Sample IDs, labels, source groups, channel ranges and fixed split membership.

The package contains 4000 samples, 800 per class, sampled at 2 kHz.
Internal label 'vehicle' refers to bicycles only.
Each 2000-point input has 41 leading zeros followed by 1959 measured points.
One source recording contributes four simultaneous spatial patches; these
must stay in the SAME subset. Fixed chronological train/validation/test
counts are 2400/800/800 samples (600/200/200 recording groups).
Independent physical-event or session boundaries are not available.
Single-channel representations use the fixed 25th channel of each patch.
2D MAT instances are min-max normalized; 1D waveforms and STFT maps use
different standardization. Consult docs/dataset.md in the repository before
comparing representations; do not treat stored amplitudes as physical units.

GETTING STARTED
For standalone use, extract this archive into its own folder, preserving
LICENSE and README.txt. MATLAB: load a class MAT file and inspect instances.
Python: read v7.3 MAT with h5py, NPZ with numpy.load(allow_pickle=False).
For the full benchmark, clone the repository and use its downloader:
  python scripts/download_data.py --archive /path/to/das-perimeter-v1.0.2.zip
It maps this archive's LICENSE to LICENSE-DATA without replacing the code's
MIT LICENSE. Do not manually unpack this archive over the code repository.
Code, checkpoints, figures, readers and benchmark results are in the GitHub
repository, not this dataset ZIP. See its CITATION.cff for citation metadata.

v1.0.2 changes packaging only: scientific data, labels and splits are identical
to v1.0.0. The existing feature table and metadata/splits are now included.
