# tests/test_trainer.py
import pytest
from src.action_space import sample_random_spec
from src.trainer import evaluate_spec

def test_evaluate_spec_returns_float():
    spec = sample_random_spec(seed=42)
    acc = evaluate_spec(spec, max_steps=20, data_root="./data")
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

def test_evaluate_spec_different_seeds_reproducible():
    spec = sample_random_spec(seed=7)
    acc1 = evaluate_spec(spec, max_steps=20, data_root="./data", seed=0)
    acc2 = evaluate_spec(spec, max_steps=20, data_root="./data", seed=0)
    assert abs(acc1 - acc2) < 0.02  # same seed → similar result
