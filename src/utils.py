"""Configuration, deterministic execution and local artifact provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None) -> dict:
    file = Path(path) if path else ROOT / 'config.yaml'
    with file.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    config.setdefault('class_names', [])
    mapping = ROOT / 'data' / 'class_mapping.json'
    if not config['class_names'] and mapping.exists():
        labels = json.loads(mapping.read_text(encoding='utf-8'))
        if isinstance(labels, dict):
            if 'class_names' in labels:
                config['class_names'] = labels['class_names']
            elif 'class_to_index' in labels:
                config['class_names'] = sorted(labels['class_to_index'], key=labels['class_to_index'].get)
            else:
                config['class_names'] = sorted(labels, key=labels.get)
        else:
            config['class_names'] = labels
    return config


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime,)):
        return value.isoformat()
    raise TypeError(f'Cannot encode {type(value)}')


def dump_json(path, obj):
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(file)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except ImportError:
        pass


def get_device(config):
    import torch
    choice = config.get('device', 'auto')
    if choice == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if choice.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was explicitly requested but is not available')
    return torch.device(choice)


def sha256_file(path, block=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def record_environment(path=None):
    environment = {'recorded_at': datetime.now(timezone.utc).isoformat(),
                   'python_version': sys.version, 'python_executable': sys.executable,
                   'platform': platform.platform(), 'packages': {}}
    for name in ['numpy', 'scipy', 'pandas', 'matplotlib', 'scikit-learn', 'torch',
                 'h5py', 'PyYAML', 'joblib', 'psutil']:
        try:
            environment['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            environment['packages'][name] = None
    try:
        import torch
        environment.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                           cudnn_version=torch.backends.cudnn.version(),
                           cuda_available=torch.cuda.is_available(),
                           gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                           gpu_capabilities=[list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())])
    except ImportError:
        environment.update(cuda_available=False)
    dump_json(path or ROOT / 'reports' / 'environment.json', environment)
    return environment


def source_code_manifest():
    files = sorted((ROOT / 'src').rglob('*.py')) + [ROOT / 'config.yaml']
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in files}
