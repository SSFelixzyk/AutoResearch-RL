# src/reward.py
from typing import List, Optional, Tuple
from src.action_space import ArchSpec


def spec_distance(a: ArchSpec, b: ArchSpec) -> float:
    """Normalised Hamming distance between two ArchSpec configs (0=identical, 1=all different)."""
    diffs = 0
    total = 0

    for va, vb in [
        (a.use_residual,          b.use_residual),
        (a.dropout,               b.dropout),
        (a.scheduler,             b.scheduler),
        (a.augment,               b.augment),
        (a.optimizer.type,        b.optimizer.type),
        (a.optimizer.lr,          b.optimizer.lr),
        (a.optimizer.weight_decay, b.optimizer.weight_decay),
    ]:
        total += 1
        if va != vb:
            diffs += 1

    n_stages = max(len(a.stages), len(b.stages))
    for i in range(n_stages):
        total += 5
        if i >= len(a.stages) or i >= len(b.stages):
            diffs += 5   # missing stage → all 5 fields differ
        else:
            sa, sb = a.stages[i], b.stages[i]
            for va, vb in [
                (sa.out_channels, sb.out_channels),
                (sa.num_blocks,   sb.num_blocks),
                (sa.norm,         sb.norm),
                (sa.activation,   sb.activation),
                (sa.downsample,   sb.downsample),
            ]:
                if va != vb:
                    diffs += 1

    return diffs / total if total > 0 else 0.0


def compute_rewards(
    raw_candidates: List[Optional[ArchSpec]],
    accs: List[float],
    best_so_far: float = 0.0,
    recent_specs: Optional[List[ArchSpec]] = None,
    invalid_penalty: float = 0.1,
    use_relative: bool = False,
    novelty_coef: float = 0.0,
    reward_floor: Optional[float] = None,
) -> Tuple[List[float], dict]:
    """
    Compute shaped rewards for GRPO.

    reward_floor applies only to valid candidates — invalid JSON keeps its
    full -invalid_penalty so the model still learns to avoid bad outputs
    even when all valid candidates are below the floor.

    Returns (rewards, stats) where stats contains per-component means for wandb.
    """
    rewards = []
    novelty_bonuses = []
    invalid_count = 0

    for spec, acc in zip(raw_candidates, accs):
        if spec is None:
            r = -invalid_penalty   # floor does NOT apply here
            novelty = 0.0
            invalid_count += 1
        else:
            r = (acc - best_so_far) if use_relative else acc
            novelty = 0.0
            if novelty_coef > 0 and recent_specs:
                min_d = min(spec_distance(spec, s) for s in recent_specs)
                novelty = novelty_coef * min_d
            r += novelty
            if reward_floor is not None:
                r = max(r, reward_floor)

        rewards.append(r)
        novelty_bonuses.append(novelty)

    n = len(rewards)
    stats = {
        "reward_mean":   sum(rewards) / n,
        "novelty_mean":  sum(novelty_bonuses) / n,
        "invalid_count": invalid_count,
        "acc_mean":      sum(accs) / n,
        "acc_best":      max(accs),
    }
    return rewards, stats
