# src/action_space.py
import bisect
import random
from typing import Literal, List, Optional
from pydantic import BaseModel, field_validator

VALID_CHANNELS = [32, 64, 128, 256]
VALID_BLOCKS   = [1, 2, 3]
VALID_NORMS    = ["none", "batch", "group"]
VALID_ACTS     = ["relu", "gelu", "silu"]
VALID_DOWN     = ["stride", "maxpool", "avgpool"]
VALID_LR       = [1e-3, 3e-3, 1e-2]
VALID_WD       = [0.0, 0.001, 0.01, 0.05]
VALID_DROP     = [0.0, 0.1, 0.2, 0.3, 0.5]
VALID_MOMENTUM = [0.9, 0.95]


class StageSpec(BaseModel):
    out_channels: Literal[32, 64, 128, 256]
    num_blocks: Literal[1, 2, 3]
    norm: Literal["none", "batch", "group"]
    activation: Literal["relu", "gelu", "silu"]
    downsample: Literal["stride", "maxpool", "avgpool"]


class OptimizerSpec(BaseModel):
    type: Literal["sgd", "adam", "adamw"]
    lr: Literal[0.001, 0.003, 0.01]  # type: ignore[valid-type]
    weight_decay: Literal[0.0, 0.001, 0.01, 0.05]  # type: ignore[valid-type]
    momentum: Optional[float] = None


class ArchSpec(BaseModel):
    stages: List[StageSpec]
    use_residual: bool
    dropout: Literal[0.0, 0.1, 0.2, 0.3, 0.5]
    optimizer: OptimizerSpec
    scheduler: Literal["none", "cosine", "step"]
    augment: Literal["none", "basic", "medium", "strong"]

    @field_validator("stages")
    @classmethod
    def check_stages(cls, v):
        if not (1 <= len(v) <= 4):
            raise ValueError("stages must have 1-4 elements")
        return v

    def to_summary(self) -> str:
        stage_str = ", ".join(
            f"C{s.out_channels}x{s.num_blocks}({s.norm},{s.activation},{s.downsample})"
            for s in self.stages
        )
        return (
            f"stages=[{stage_str}] residual={self.use_residual} "
            f"drop={self.dropout} opt={self.optimizer.type}(lr={self.optimizer.lr},wd={self.optimizer.weight_decay}) "
            f"sched={self.scheduler} aug={self.augment}"
        )

    def to_dict(self) -> dict:
        return self.model_dump()


def validate_spec(raw: dict) -> Optional[ArchSpec]:
    try:
        return ArchSpec.model_validate(raw)
    except Exception:
        return None


def _nearest_num(value, valid: list):
    """Return the element in valid closest to value."""
    idx = bisect.bisect_left(valid, value)
    if idx == 0:
        return valid[0]
    if idx >= len(valid):
        return valid[-1]
    lo, hi = valid[idx - 1], valid[idx]
    return lo if abs(value - lo) <= abs(value - hi) else hi


def _nearest_str(value, valid: list):
    return value if value in valid else valid[0]


def coerce_spec(raw: dict) -> Optional[ArchSpec]:
    """
    Like validate_spec but fixes out-of-range values instead of rejecting them.
    Numeric fields snap to the nearest allowed value; string fields fall back to
    the first allowed value when unknown. Returns None only for structurally
    broken input (missing required keys, wrong types).
    """
    if not isinstance(raw, dict):
        return None
    try:
        raw = dict(raw)

        # stages
        stages = raw.get("stages", [])
        if not isinstance(stages, list) or not stages:
            return None
        coerced = []
        for s in stages[:4]:          # cap at 4 stages
            if not isinstance(s, dict):
                continue
            s = dict(s)
            s["out_channels"] = _nearest_num(s.get("out_channels", 64), VALID_CHANNELS)
            s["num_blocks"]   = _nearest_num(s.get("num_blocks", 2),    VALID_BLOCKS)
            s["norm"]         = _nearest_str(s.get("norm", "batch"),     VALID_NORMS)
            s["activation"]   = _nearest_str(s.get("activation", "relu"),VALID_ACTS)
            s["downsample"]   = _nearest_str(s.get("downsample", "stride"), VALID_DOWN)
            coerced.append(s)
        if not coerced:
            return None
        raw["stages"] = coerced

        # optimizer
        opt = dict(raw.get("optimizer", {}))
        opt["type"]         = _nearest_str(opt.get("type", "adamw"), ["sgd", "adam", "adamw"])
        opt["lr"]           = _nearest_num(float(opt.get("lr", 1e-3)), VALID_LR)
        opt["weight_decay"] = _nearest_num(float(opt.get("weight_decay", 0.0)), VALID_WD)
        if opt["type"] != "sgd":
            opt.pop("momentum", None)   # momentum is sgd-only
        elif "momentum" in opt:
            opt["momentum"] = _nearest_num(float(opt["momentum"]), VALID_MOMENTUM)
        raw["optimizer"] = opt

        # scalar fields
        raw["dropout"]   = _nearest_num(float(raw.get("dropout", 0.0)), VALID_DROP)
        raw["scheduler"] = _nearest_str(raw.get("scheduler", "cosine"), ["none", "cosine", "step"])
        raw["augment"]   = _nearest_str(raw.get("augment", "medium"),   ["none", "basic", "medium", "strong"])

        return ArchSpec.model_validate(raw)
    except Exception:
        return None


def sample_random_spec(seed: Optional[int] = None) -> ArchSpec:
    rng = random.Random(seed)
    n_stages = rng.randint(1, 4)
    stages = [
        StageSpec(
            out_channels=rng.choice(VALID_CHANNELS),
            num_blocks=rng.choice(VALID_BLOCKS),
            norm=rng.choice(VALID_NORMS),
            activation=rng.choice(VALID_ACTS),
            downsample=rng.choice(VALID_DOWN),
        )
        for _ in range(n_stages)
    ]
    opt_type = rng.choice(["sgd", "adam", "adamw"])
    momentum = rng.choice(VALID_MOMENTUM) if opt_type == "sgd" else None
    return ArchSpec(
        stages=stages,
        use_residual=rng.choice([True, False]),
        dropout=rng.choice(VALID_DROP),
        optimizer=OptimizerSpec(
            type=opt_type,
            lr=rng.choice(VALID_LR),
            weight_decay=rng.choice(VALID_WD),
            momentum=momentum,
        ),
        scheduler=rng.choice(["none", "cosine", "step"]),
        augment=rng.choice(["none", "basic", "medium", "strong"]),
    )
