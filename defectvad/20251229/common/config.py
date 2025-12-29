# src/defectvad/common/config.py

import os
import yaml
from copy import deepcopy
import re


def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return _replace(config, config)


def _replace(config, base_config):
    if isinstance(config, dict):
        return {k: _replace(v, base_config) for k, v in config.items()}
    elif isinstance(config, str):
        pattern = re.compile(r"\$\{([^}]+)\}")
        while True:
            match = pattern.search(config)
            if not match:
                break
            key = match.group(1)
            replacement = _replace(base_config.get(key, ""), base_config)
            config = config[:match.start()] + str(replacement) + config[match.end():]
        return config
    elif isinstance(config, list):
        return [_replace(item, base_config) for item in config]
    else:
        return config


def merge_configs(*configs):
    merged = {}
    for config in configs:
        if config is not None:
            merged = _deep_merge(merged, config)
    return merged


def _deep_merge(base, new):
    base = deepcopy(base)
    for key, value in new.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base
