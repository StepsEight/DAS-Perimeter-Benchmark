"""B0: exactly 15 fixed-channel features, training-only scaling and RBF-SVM."""

import argparse
import csv
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from .dataset import get_feature_data
    from .metrics import classification_metrics, save_evaluation
    from .trainer import data_artifact_fingerprints, experiment_fingerprint, output_directory
    from .utils import dump_json, load_config, seed_everything, source_code_manifest
except ImportError:
    from dataset import get_feature_data
    from metrics import classification_metrics, save_evaluation
    from trainer import data_artifact_fingerprints, experiment_fingerprint, output_directory
    from utils import dump_json, load_config, seed_everything, source_code_manifest


def train_svm(config):
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    output = output_directory(config, "svm")
    fingerprint = experiment_fingerprint(config, "svm")
    data_fingerprints = data_artifact_fingerprints(config)
    code_manifest = source_code_manifest()
    summary_path = output / "run_summary.json"
    if summary_path.exists():
        completed = json.loads(summary_path.read_text(encoding="utf-8"))
        if completed.get("experiment_fingerprint") != fingerprint:
            raise RuntimeError("Saved SVM run has a different configuration or fixed split; use another output_root.")
        print("svm: completed run exists; retaining its one final test evaluation.", flush=True)
        return completed
    data = get_feature_data(config)
    class_names = list(config["class_names"])
    if data["X_train"].shape[1] != 15:
        raise ValueError("The primary SVM baseline requires exactly 15 features.")
    for split in ("train", "val", "test"):
        if not np.isfinite(data[f"X_{split}"]).all():
            raise ValueError(f"Non-finite handcrafted features in {split} split.")
    dump_json(output / "config_snapshot.json", config)
    started = time.perf_counter()
    scaler = StandardScaler().fit(data["X_train"])
    train_features = scaler.transform(data["X_train"])
    validation_features = scaler.transform(data["X_val"])
    search = config.get("svm", {})
    c_values = search.get("C", search.get("C_grid", [0.1, 1, 10, 100]))
    gamma_values = search.get("gamma", search.get("gamma_grid", ["scale", 0.001, 0.01, 0.1, 1]))
    grid_results, best_score, best_model, best_parameters = [], -float("inf"), None, None
    for c_value in c_values:
        for gamma_value in gamma_values:
            candidate = SVC(C=float(c_value), gamma=gamma_value, kernel="rbf", class_weight=None,
                            probability=False, random_state=seed)
            fit_started = time.perf_counter()
            candidate.fit(train_features, data["y_train"])
            prediction = candidate.predict(validation_features)
            metrics = classification_metrics(data["y_val"], prediction, class_names)
            score = float(metrics["macro_f1"])
            grid_results.append({"C": c_value, "gamma": gamma_value, "val_macro_f1": score,
                                 "val_accuracy": metrics["accuracy"], "seconds": time.perf_counter() - fit_started})
            print(f"svm C={c_value} gamma={gamma_value} val_Macro-F1={score:.6f}", flush=True)
            # Deterministic ties retain the first grid candidate; no test-based tie breaking.
            if score > best_score:
                best_score, best_model = score, candidate
                best_parameters = {"C": float(c_value), "gamma": gamma_value, "kernel": "rbf"}
    train_seconds = time.perf_counter() - started
    with (output / "validation_grid.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(grid_results[0]))
        writer.writeheader()
        writer.writerows(grid_results)
    dump_json(output / "best_parameters.json", best_parameters)
    # Keep the training-only fit. Do not refit on train+validation after selection.
    joblib.dump(scaler, output / "scaler.joblib")
    joblib.dump(best_model, output / "best_model.joblib")
    joblib.dump({"scaler": scaler, "model": best_model, "class_names": class_names,
                 "parameters": best_parameters, "config": config, "experiment_fingerprint": fingerprint,
                 "data_artifact_fingerprints": data_fingerprints, "source_code_manifest": code_manifest},
                output / "best_pipeline.joblib")
    validation_prediction = best_model.predict(validation_features)
    validation_metrics = save_evaluation(output, "val", data["y_val"], validation_prediction,
                                         class_names, data["ids_val"])
    marker = output / "final_test_started.json"
    if marker.exists():
        raise RuntimeError("Final SVM test pass was already started; inspect existing artifacts before rerunning.")
    dump_json(marker, {"parameters": best_parameters, "selection_complete": True,
                       "experiment_fingerprint": fingerprint})
    inference_started = time.perf_counter()
    test_prediction = best_model.predict(scaler.transform(data["X_test"]))
    inference_seconds = time.perf_counter() - inference_started
    test_metrics = save_evaluation(output, "test", data["y_test"], test_prediction,
                                   class_names, data["ids_test"])
    summary = {
        "method": "svm", "best_parameters": best_parameters, "parameter_count": None,
        "best_epoch": None, "train_seconds": train_seconds,
        "inference_ms_per_sample": inference_seconds * 1000 / len(test_prediction),
        "inference_timing_scope": "single final test pass including StandardScaler transformation",
        "model_path": str(output / "best_model.joblib"),
        "model_size_bytes": (output / "best_model.joblib").stat().st_size,
        "pipeline_path": str(output / "best_pipeline.joblib"), "actual_batch_size": None,
        "seed": seed, "class_names": class_names, "feature_count": 15,
        "scaler_fit_split": "train only", "final_fit_split": "train only",
        "selection_metric": "validation macro_f1", "tie_policy": "first grid candidate",
        "test_evaluation_count": 1, "experiment_fingerprint": fingerprint,
        "validation": validation_metrics, "test": test_metrics,
        "data_artifact_fingerprints": data_fingerprints, "source_code_manifest": code_manifest,
    }
    dump_json(summary_path, summary)
    dump_json(output / "training_summary.json", summary)
    print(f"svm: complete, params={best_parameters}, val Macro-F1={validation_metrics['macro_f1']:.6f}, "
          f"test Macro-F1={test_metrics['macro_f1']:.6f}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the 15-feature RBF-SVM baseline.")
    parser.add_argument("--config", default=None)
    arguments = parser.parse_args()
    train_svm(load_config(arguments.config))
