"""Subsetted MATH config for quick smoke evaluations."""
from __future__ import annotations

import importlib.util
import os


def _load_base_datasets():
    base_path = os.path.join(os.path.dirname(__file__), "math_0shot_gen_393424.py")
    spec = importlib.util.spec_from_file_location("_math_base", base_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module.math_datasets


_base_math_datasets = _load_base_datasets()


def _with_slice(dataset: dict, start: int, stop: int) -> dict:
    demo = dict(dataset)
    demo['abbr'] = f"demo64_{demo['abbr']}"
    reader = dict(demo.get('reader_cfg', {}))
    reader['test_range'] = f'[{start}:{stop}]'
    demo['reader_cfg'] = reader
    return demo


math_demo64_datasets = [_with_slice(dataset, 0, 64) for dataset in _base_math_datasets]
