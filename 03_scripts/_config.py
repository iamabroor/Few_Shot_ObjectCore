"""Shared helpers for the pipeline scripts: YAML configs -> objectcore.py arguments."""
import os
import sys
import types
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (HERE, REPO / "02_src" / "source_code"):  # flat copy in ~/objectcore, or the repo layout
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# keys used by the scripts but not objectcore.py flags
NON_FLAG_KEYS = {"base", "categories"}


def find_config(name):
    for c in (Path(name), HERE / name, REPO / "04_configs" / name, Path.home() / "objectcore" / "04_configs" / name):
        if c.exists():
            return c
    raise FileNotFoundError(f"config {name} not found")


def load_config(path):
    """Load a YAML config, following `base:` to the default config."""
    path = find_config(path)
    cfg = yaml.safe_load(open(path)) or {}
    if "base" in cfg:
        base = load_config(path.parent / cfg["base"])
        base.update({k: v for k, v in cfg.items() if k != "base"})
        cfg = base
    for k in ("data_root", "out_dir"):
        if isinstance(cfg.get(k), str):
            cfg[k] = os.path.expanduser(cfg[k])
    return cfg


def to_namespace(cfg, **overrides):
    """objectcore.py-style args namespace (same names as its argparse flags)."""
    d = dict(cfg)
    d.update({k: v for k, v in overrides.items() if v is not None})
    d.setdefault("max_test", None)
    d.setdefault("nms", None)
    return types.SimpleNamespace(**d)


def to_argv(cfg, category):
    """Command line for objectcore.py main() (used by run_from_config in training.py)."""
    argv = ["--category", category]
    for k, v in cfg.items():
        if k in NON_FLAG_KEYS or v is None or v is False:
            continue
        flag = f"--{k}"
        if v is True:
            argv.append(flag)
        elif isinstance(v, (list, tuple)):
            argv += [flag] + [str(x) for x in v]
        else:
            argv += [flag, str(v)]
    return argv
