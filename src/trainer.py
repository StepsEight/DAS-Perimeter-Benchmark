"""Unified training with validation-only selection and resumable checkpoints.

The test loader is never iterated during fitting or checkpoint selection.
An epoch tie keeps the earlier checkpoint. A completed run is read back rather
than fitting/evaluating again, so each selected model has one final test pass.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

try:
    from .dataset import get_dataloaders
    from .metrics import classification_metrics, save_evaluation
    from .models import build_model, parameter_count
    from .run_utils import data_artifact_fingerprints, experiment_fingerprint, output_directory
    from .utils import ROOT, dump_json, get_device, load_config, seed_everything, sha256_file, source_code_manifest
except ImportError:
    from dataset import get_dataloaders
    from metrics import classification_metrics, save_evaluation
    from models import build_model, parameter_count
    from run_utils import data_artifact_fingerprints, experiment_fingerprint, output_directory
    from utils import ROOT, dump_json, get_device, load_config, seed_everything, sha256_file, source_code_manifest


def setting(config, name, default):
    return config.get("training", {}).get(name, config.get(name, default))


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _rng_state(loader):
    generator = getattr(loader, "generator", None)
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_loader_generator": generator.get_state() if generator is not None else None,
    }


def _restore_rng(state, loader):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    generator = getattr(loader, "generator", None)
    if generator is not None and state["train_loader_generator"] is not None:
        generator.set_state(state["train_loader_generator"].cpu())


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_loader(model, loader, device, use_amp=False, criterion=None):
    """One ordered pass; predictions and IDs retain the fixed split order."""
    model.eval()
    actual, predicted, probabilities, identifiers = [], [], [], []
    loss_sum, count = 0.0, 0
    _synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for inputs, targets, sample_ids in loader:
            inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device, dtype=torch.long, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(inputs)
                if criterion is not None:
                    loss = criterion(logits, targets)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite evaluation logits; stop rather than emit invalid metrics.")
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            actual.append(targets.cpu().numpy())
            predicted.append(logits.argmax(dim=1).cpu().numpy())
            identifiers.extend(str(identifier) for identifier in sample_ids)
            count += len(targets)
            if criterion is not None:
                loss_sum += float(loss.item()) * len(targets)
    _synchronize(device)
    elapsed = time.perf_counter() - started
    if count == 0:
        raise ValueError("Empty evaluation split.")
    return {
        "y_true": np.concatenate(actual), "y_pred": np.concatenate(predicted),
        "probabilities": np.concatenate(probabilities), "sample_ids": identifiers,
        "loss": loss_sum / count if criterion is not None else None,
        "seconds": elapsed, "inference_ms_per_sample": elapsed * 1000 / count,
    }


def _write_history(path, history):
    fields = ["epoch", "learning_rate", "train_loss", "train_accuracy", "val_loss",
              "val_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1",
              "epoch_seconds", "is_new_best"]
    temporary = Path(path).with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def train(method, config, resume=False):
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    torch.set_num_threads(int(setting(config, "torch_threads", 4)))
    device = torch.device(get_device(config))
    output = output_directory(config, method)
    fingerprint = experiment_fingerprint(config, method)
    data_fingerprints = data_artifact_fingerprints(config)
    code_manifest = source_code_manifest()
    summary_path = output / "run_summary.json"
    if summary_path.exists():
        completed = json.loads(summary_path.read_text(encoding="utf-8"))
        if completed.get("experiment_fingerprint") != fingerprint:
            raise RuntimeError("A completed output has a different configuration/split. Use another output_root.")
        print(f"{method}: completed run exists; returning saved results without another test evaluation.", flush=True)
        return completed
    if (output / "latest.pt").exists() and not resume:
        raise FileExistsError(f"{output / 'latest.pt'} exists. Pass --resume to continue the same experiment.")

    loaders = get_dataloaders(config, method)
    class_names = list(config["class_names"])
    epochs = int(setting(config, "epochs", 100))
    batch_size = int(loaders["train"].batch_size)
    use_amp = bool(setting(config, "amp", False)) and device.type == "cuda"
    model = build_model(method, len(class_names), float(setting(config, "dropout", 0.3))).to(device)
    n_parameters = parameter_count(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(setting(config, "learning_rate", 1e-3)),
                                 weight_decay=float(setting(config, "weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history, first_epoch, best_f1, best_epoch, previous_seconds = [], 1, -float("inf"), None, 0.0
    if resume and (output / "latest.pt").exists():
        # Trusted local checkpoint includes NumPy/Python RNG tuples for exact continuation.
        checkpoint = torch.load(output / "latest.pt", map_location=device, weights_only=False)
        if checkpoint["experiment_fingerprint"] != fingerprint:
            raise RuntimeError("Resume configuration or fixed splits changed; checkpoint continuation refused.")
        code_manifest = checkpoint.get("source_code_manifest", code_manifest)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["grad_scaler_state"])
        history = checkpoint["history"]
        first_epoch = int(checkpoint["epoch"]) + 1
        best_f1, best_epoch = checkpoint["best_val_macro_f1"], checkpoint["best_epoch"]
        previous_seconds = float(checkpoint["train_seconds"])
        _restore_rng(checkpoint["rng_state"], loaders["train"])
    dump_json(output / "config_snapshot.json", config)
    print(f"{method}: device={device}, parameters={n_parameters}, batch={batch_size}, "
          f"AMP={use_amp}, epochs={first_epoch}..{epochs}, train/val/test="
          f"{len(loaders['train'].dataset)}/{len(loaders['val'].dataset)}/{len(loaders['test'].dataset)}", flush=True)
    started = time.perf_counter()
    for epoch in range(first_epoch, epochs + 1):
        epoch_started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        model.train()
        train_loss_sum, train_correct, count = 0.0, 0, 0
        for inputs, targets, _ in loaders["train"]:
            inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.item()) * len(targets)
            train_correct += int((logits.argmax(dim=1) == targets).sum().item())
            count += len(targets)
        validation = evaluate_loader(model, loaders["val"], device, use_amp, criterion)
        val_metrics = classification_metrics(validation["y_true"], validation["y_pred"], class_names)
        new_best = float(val_metrics["macro_f1"]) > best_f1
        if new_best:
            best_f1, best_epoch = float(val_metrics["macro_f1"]), epoch
        scheduler.step()
        row = {
            "epoch": epoch, "learning_rate": learning_rate, "train_loss": train_loss_sum / count,
            "train_accuracy": train_correct / count, "val_loss": validation["loss"],
            "val_accuracy": val_metrics["accuracy"], "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"], "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": time.perf_counter() - epoch_started, "is_new_best": new_best,
        }
        history.append(row)
        checkpoint = {
            "format_version": 1, "method": method, "epoch": epoch,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "grad_scaler_state": scaler.state_dict(),
            "validation_metrics": val_metrics, "config": config,
            "class_names": class_names, "class_mapping": {name: i for i, name in enumerate(class_names)},
            "rng_state": _rng_state(loaders["train"]), "history": history,
            "best_val_macro_f1": best_f1, "best_epoch": best_epoch,
            "experiment_fingerprint": fingerprint, "parameter_count": n_parameters,
            "actual_batch_size": batch_size, "amp_enabled": use_amp,
            "data_artifact_fingerprints": data_fingerprints, "source_code_manifest": code_manifest,
            "train_seconds": previous_seconds + time.perf_counter() - started,
            "selection_rule": "validation macro-F1; strict improvement; earliest epoch wins ties",
        }
        if new_best:
            atomic_torch_save(checkpoint, output / "best.pt")
        atomic_torch_save(checkpoint, output / "latest.pt")
        _write_history(output / "history.csv", history)
        print(f"{method} epoch={epoch:03d}/{epochs} train_loss={row['train_loss']:.6f} "
              f"val_F1={row['val_macro_f1']:.6f} best={best_f1:.6f}@{best_epoch} "
              f"seconds={row['epoch_seconds']:.2f}", flush=True)
    train_seconds = previous_seconds + time.perf_counter() - started
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    validation = evaluate_loader(model, loaders["val"], device, use_amp, criterion)
    validation_metrics = save_evaluation(output, "val", validation["y_true"], validation["y_pred"],
                                         class_names, validation["sample_ids"], validation["probabilities"])
    test_started_path = output / "final_test_started.json"
    test_metrics_path = output / "test_metrics.json"
    if test_started_path.exists():
        raise RuntimeError("Final test was already started. Refusing another test pass; inspect saved artifacts.")
    dump_json(test_started_path, {"best_epoch": int(best["epoch"]), "experiment_fingerprint": fingerprint,
                                  "selection_complete": True})
    final_test = evaluate_loader(model, loaders["test"], device, use_amp, criterion)
    test_metrics = save_evaluation(output, "test", final_test["y_true"], final_test["y_pred"],
                                   class_names, final_test["sample_ids"], final_test["probabilities"])
    assert test_metrics_path.exists(), "The evaluation writer must persist test_metrics.json."
    summary = {
        "method": method, "best_epoch": int(best["epoch"]), "epochs_completed": epochs,
        "parameter_count": n_parameters, "train_seconds": train_seconds,
        "inference_ms_per_sample": final_test["inference_ms_per_sample"],
        "inference_timing_scope": "single final test pass including batches, transfer, softmax and CPU collection",
        "model_path": str(output / "best.pt"), "model_size_bytes": (output / "best.pt").stat().st_size,
        "latest_checkpoint_path": str(output / "latest.pt"), "actual_batch_size": batch_size,
        "amp_enabled": use_amp, "amp_dtype": "float16" if use_amp else None,
        "weight_dtype": "float32", "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "seed": seed, "augmentation": "none", "class_names": class_names,
        "experiment_fingerprint": fingerprint, "validation": validation_metrics, "test": test_metrics,
        "test_evaluation_count": 1, "selection_metric": "validation macro_f1", "tie_policy": "earliest epoch",
        "data_artifact_fingerprints": data_fingerprints, "source_code_manifest": code_manifest,
    }
    dump_json(summary_path, summary)
    dump_json(output / "training_summary.json", summary)
    print(f"{method}: complete, best epoch={summary['best_epoch']}, "
          f"val Macro-F1={validation_metrics['macro_f1']:.6f}, test Macro-F1={test_metrics['macro_f1']:.6f}", flush=True)
    return summary


def main(method, argv=None):
    parser = argparse.ArgumentParser(description=f"Train {method} using fixed chronological splits.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", action="store_true", help="Continue latest.pt without resetting optimizer/scheduler/RNG.")
    arguments = parser.parse_args(argv)
    return train(method, load_config(arguments.config), resume=arguments.resume)
