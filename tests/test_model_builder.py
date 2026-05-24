# tests/test_model_builder.py
import torch
from torch.optim import SGD
from src.action_space import sample_random_spec, ArchSpec, StageSpec, OptimizerSpec
from src.model_builder import build_model, build_optimizer, build_scheduler, PlainBlock, ResBlock

def test_build_model_forward_pass():
    spec = sample_random_spec(seed=0)
    model = build_model(spec)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)

def test_build_model_residual_off():
    spec = sample_random_spec(seed=1)
    spec = spec.model_copy(update={"use_residual": False})
    model = build_model(spec)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
    assert not any(isinstance(m, ResBlock) for m in model.modules())

def test_build_optimizer_types():
    spec = sample_random_spec(seed=2)
    model = build_model(spec)
    opt = build_optimizer(spec, model.parameters())
    assert opt is not None

def test_build_scheduler_cosine():
    spec = sample_random_spec(seed=3)
    spec = spec.model_copy(update={"scheduler": "cosine"})
    model = build_model(spec)
    opt = build_optimizer(spec, model.parameters())
    sched = build_scheduler(spec, opt, total_steps=500)
    assert sched is not None

def test_build_scheduler_none_returns_none():
    spec = sample_random_spec(seed=4)
    spec = spec.model_copy(update={"scheduler": "none"})
    model = build_model(spec)
    opt = build_optimizer(spec, model.parameters())
    assert build_scheduler(spec, opt, total_steps=500) is None

def test_build_scheduler_step_small_steps():
    spec = sample_random_spec(seed=5)
    spec = spec.model_copy(update={"scheduler": "step"})
    model = build_model(spec)
    opt = build_optimizer(spec, model.parameters())
    sched = build_scheduler(spec, opt, total_steps=2)
    assert sched is not None  # step_size=max(1, 2//3)=1, no crash

def test_build_optimizer_sgd():
    spec = sample_random_spec(seed=0)
    spec = spec.model_copy(update={
        "optimizer": OptimizerSpec(type="sgd", lr=0.001, weight_decay=0.0, momentum=0.9)
    })
    model = build_model(spec)
    opt = build_optimizer(spec, model.parameters())
    assert isinstance(opt, SGD)
