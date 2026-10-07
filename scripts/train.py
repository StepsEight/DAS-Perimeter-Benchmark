"""Train the paper baselines using the released fixed protocol."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['all','svm','cnn1d','cnn2d','stft_cnn'], default='all')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    methods = ['svm','cnn1d','cnn2d','stft_cnn'] if args.model=='all' else [args.model]
    for method in methods:
        if method=='svm':
            from train_svm import train_svm
            train_svm(config)
        else:
            from trainer import train
            train(method, config, resume=args.resume)


if __name__ == '__main__':
    main()
