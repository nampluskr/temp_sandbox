# src/anomaly_detection/config.py
import os
import re
import yaml


_VAR_PATTERN = re.compile(r"\$\{([^}^{]+)\}")


def _substitute_vars(value, context):
    def replacer(match):
        key = match.group(1)
        if key not in context:
            raise KeyError(f"Undefined variable in config: {key}")
        return context[key]

    return _VAR_PATTERN.sub(replacer, value)


def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    if not isinstance(raw_config, dict):
        raise ValueError("Config file must define a mapping")

    context = {
        "PWD": os.getcwd(),
    }

    resolved = {}

    for key, value in raw_config.items():
        if not isinstance(value, str):
            raise TypeError(f"Config value must be string: {key}")

        substituted = _substitute_vars(value, context)

        if not os.path.isabs(substituted):
            substituted = os.path.abspath(substituted)

        resolved[key] = substituted
        context[key] = substituted

    return resolved


def merge_configs(*configs):
    merged = {}
    for config in configs:
        merged.update(config)
    return merged
