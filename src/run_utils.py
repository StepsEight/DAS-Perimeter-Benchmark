"""Run directories and input provenance shared by CPU SVM and CNN training."""
import hashlib
import json
from pathlib import Path

try:
    from .utils import ROOT
    from .dataset import data_directory, verify_paths
except ImportError:
    from utils import ROOT
    from dataset import data_directory, verify_paths


def output_directory(config, method):
    output_root = config.get('paths', {}).get('output_root', config.get('output_root', 'outputs'))
    root = Path(output_root)
    if not root.is_absolute():
        root = ROOT / root
    output = root / method
    output.mkdir(parents=True, exist_ok=True)
    return output


def data_artifact_fingerprints(config, method=None):
    manifest = json.loads((data_directory(config) / 'manifest.json').read_text(encoding='utf-8'))
    records = manifest['files']
    if method == 'svm':
        required = {'data/metadata.csv', 'data/class_mapping.json', 'data/handcrafted_features.csv',
                    *[f'data/splits/{split}.csv' for split in ('train', 'val', 'test')]}
        records = [record for record in records if record['path'] in required]
        if {record['path'] for record in records} != required:
            raise ValueError('SVM inputs are missing from the release manifest')
    verify_paths([record['path'] for record in records], config)
    return {record['path']: record['sha256'] for record in records}


def experiment_fingerprint(config, method):
    """Refuse accidental resume against a different configuration or input data."""
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode('utf-8'))
    digest.update(method.encode('utf-8'))
    for name, content_hash in data_artifact_fingerprints(config, method).items():
        digest.update(name.encode('utf-8'))
        digest.update(content_hash.encode('ascii'))
    return digest.hexdigest()
