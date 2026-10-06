"""Evaluate a released best model on a named fixed split."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from utils import ROOT, load_config, get_device, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['svm','cnn1d','cnn2d','stft_cnn'], required=True)
    parser.add_argument('--split', choices=['val','test'], default='test')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    from dataset import get_dataloaders, get_feature_data
    from metrics import save_evaluation
    config = load_config(args.config)
    seed_everything(config['seed'])
    output = args.output or ROOT/'runs/evaluation'/f'{args.model}_{args.split}'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose an empty evaluation output directory: {output}')
    if args.model == 'svm':
        import joblib
        checkpoint = args.checkpoint or ROOT/'checkpoints/svm.joblib'
        artifact = joblib.load(checkpoint)
        if artifact['class_names'] != config['class_names']:
            raise ValueError('Checkpoint class order mismatch')
        data = get_feature_data(config)
        prediction = artifact['model'].predict(artifact['scaler'].transform(data[f'X_{args.split}']))
        result = save_evaluation(output, args.split, data[f'y_{args.split}'], prediction,
            config['class_names'], data[f'ids_{args.split}'])
    else:
        import torch
        from models import build_model
        from trainer import evaluate_loader
        torch.set_num_threads(config['training']['torch_threads'])
        device = get_device(config)
        checkpoint = args.checkpoint or ROOT/'checkpoints'/f'{args.model}.pt'
        artifact = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if artifact['method'] != args.model or artifact['class_names'] != config['class_names']:
            raise ValueError('Checkpoint method/class order mismatch')
        model = build_model(args.model, 5, config['training']['dropout']).to(device)
        model.load_state_dict(artifact['model_state'])
        loader = get_dataloaders(config, args.model)[args.split]
        evaluation = evaluate_loader(model, loader, device,
            use_amp=bool(config['training']['amp']) and device.type=='cuda')
        result = save_evaluation(output, args.split, evaluation['y_true'], evaluation['y_pred'],
            config['class_names'], evaluation['sample_ids'], evaluation['probabilities'])
    print(f"{args.model} {args.split}: Accuracy={result['accuracy']:.6f}, Macro-F1={result['macro_f1']:.6f}")


if __name__ == '__main__':
    main()
