# tests/test_action_space.py
from src.action_space import ArchSpec, sample_random_spec, validate_spec

def test_sample_random_spec_returns_valid():
    spec = sample_random_spec(seed=42)
    assert isinstance(spec, ArchSpec)
    assert 1 <= len(spec.stages) <= 4

def test_validate_spec_accepts_valid_json():
    raw = {
        "stages": [{"out_channels": 64, "num_blocks": 2, "norm": "batch",
                     "activation": "relu", "downsample": "stride"}],
        "use_residual": True, "dropout": 0.1,
        "optimizer": {"type": "adamw", "lr": 0.001, "weight_decay": 0.01},
        "scheduler": "cosine", "augment": "medium"
    }
    spec = validate_spec(raw)
    assert spec is not None

def test_validate_spec_rejects_invalid_channel():
    raw = {
        "stages": [{"out_channels": 99, "num_blocks": 2, "norm": "batch",
                     "activation": "relu", "downsample": "stride"}],
        "use_residual": False, "dropout": 0.0,
        "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.0},
        "scheduler": "none", "augment": "none"
    }
    assert validate_spec(raw) is None

def test_spec_to_summary_is_string():
    spec = sample_random_spec(seed=0)
    summary = spec.to_summary()
    assert isinstance(summary, str)
    assert "stages" in summary

def test_sample_random_spec_seed_determinism():
    assert sample_random_spec(seed=7).to_dict() == sample_random_spec(seed=7).to_dict()

def test_validate_spec_rejects_invalid_lr():
    raw = {
        "stages": [{"out_channels": 64, "num_blocks": 1, "norm": "none",
                     "activation": "relu", "downsample": "stride"}],
        "use_residual": False, "dropout": 0.0,
        "optimizer": {"type": "adam", "lr": 0.123, "weight_decay": 0.0},
        "scheduler": "none", "augment": "none"
    }
    assert validate_spec(raw) is None
