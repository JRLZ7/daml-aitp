"""Shared run provenance and device helpers."""

import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def load_config(path):
    with Path(path).open() as stream:
        return yaml.safe_load(stream)


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Best-effort deterministic kernels; hardware/library changes may still vary.
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False


def choose_device(requested):
    if requested == 'auto':
        requested = ('cuda' if torch.cuda.is_available() else
                     'mps' if torch.backends.mps.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS requested but unavailable')
    return torch.device(requested)


def git_state():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    return {'commit': git('rev-parse', 'HEAD'),
            'dirty': bool(git('status', '--porcelain')),
            'diff': git('diff', 'HEAD'),
            'status': git('status', '--short')}
