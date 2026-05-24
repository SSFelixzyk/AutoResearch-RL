# experiments/grid_search.py
"""
Standalone grid search covering all major action-space dimensions.
Run from the cifar10_automl/ directory:

  python experiments/grid_search.py
  python experiments/grid_search.py --data-root /data --max-train-steps 200
  python experiments/grid_search.py --seeds 0 1 2
  python experiments/grid_search.py --parallel   # spawn all jobs simultaneously
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.action_space import ArchSpec, OptimizerSpec, StageSpec
from src.trainer import evaluate_spec


def _s(c, n, norm, act, down="stride"):
    return StageSpec(out_channels=c, num_blocks=n, norm=norm,  # type: ignore[arg-type]
                     activation=act, downsample=down)


def _spec(stages, *, res=True, drop=0.1,
          opt="adamw", lr=0.003, wd=0.01, mom=None,
          sched="cosine", aug="medium"):
    return ArchSpec(
        stages=stages,
        use_residual=res,
        dropout=drop,
        optimizer=OptimizerSpec(type=opt, lr=lr, weight_decay=wd, momentum=mom),  # type: ignore[arg-type]
        scheduler=sched,  # type: ignore[arg-type]
        augment=aug,      # type: ignore[arg-type]
    )


def _B3(norm, act):
    return [
        _s(64,  2, norm, act, "stride"),
        _s(128, 3, norm, act, "stride"),
        _s(256, 2, norm, act, "avgpool"),
    ]


# ---------------------------------------------------------------------------
# Grid: one-factor-at-a-time around a strong base + a few best-guess combos
# Base: 3-stage 64→128→256, batch norm, silu, residual, adamw lr=0.003,
#       wd=0.01, cosine, augment=medium, dropout=0.1
# ---------------------------------------------------------------------------
GRID = [
    # name                      stages                                            kwargs
    # ── Architecture: depth / width ──────────────────────────────────────────
    ("A1_2stage_64-128",     [_s(64,2,"batch","silu"), _s(128,3,"batch","silu","avgpool")]),
    ("A2_3stage_64-256",     _B3("batch","silu")),           # base
    ("A3_3stage_32-128",     [_s(32,2,"batch","silu"), _s(64,3,"batch","silu"), _s(128,2,"batch","silu","avgpool")]),
    ("A4_4stage_32-256",     [_s(32,2,"batch","silu"), _s(64,2,"batch","silu"), _s(128,2,"batch","silu"), _s(256,2,"batch","silu","avgpool")]),
    ("A5_3stage_deep",       [_s(64,3,"batch","silu"), _s(128,3,"batch","silu"), _s(256,3,"batch","silu","avgpool")]),
    ("A6_wide_128-256",      [_s(128,2,"batch","silu"), _s(256,3,"batch","silu","avgpool")]),
    # ── Norm ─────────────────────────────────────────────────────────────────
    ("N1_group",             _B3("group","silu")),
    ("N2_none",              _B3("none", "silu")),
    # ── Activation ───────────────────────────────────────────────────────────
    ("Act1_relu",            _B3("batch","relu")),
    ("Act2_gelu",            _B3("batch","gelu")),
    # ── Learning rate ────────────────────────────────────────────────────────
    ("LR1_0.001",            _B3("batch","silu"),  dict(lr=0.001)),
    ("LR3_0.01",             _B3("batch","silu"),  dict(lr=0.01)),
    # ── Weight decay ─────────────────────────────────────────────────────────
    ("WD1_0.001",            _B3("batch","silu"),  dict(wd=0.001)),
    ("WD2_0.05",             _B3("batch","silu"),  dict(wd=0.05)),
    # ── Optimizer ────────────────────────────────────────────────────────────
    ("Opt1_adam",            _B3("batch","silu"),  dict(opt="adam")),
    ("Opt2_sgd_mom0.9",      _B3("batch","silu"),  dict(opt="sgd", mom=0.9, lr=0.003)),
    # ── Scheduler ────────────────────────────────────────────────────────────
    ("Sched1_step",          _B3("batch","silu"),  dict(sched="step")),
    ("Sched2_none",          _B3("batch","silu"),  dict(sched="none")),
    # ── Augmentation ─────────────────────────────────────────────────────────
    ("Aug1_basic",           _B3("batch","silu"),  dict(aug="basic")),
    ("Aug2_strong",          _B3("batch","silu"),  dict(aug="strong")),
    # ── Dropout ──────────────────────────────────────────────────────────────
    ("Drop1_0.2",            _B3("batch","silu"),  dict(drop=0.2)),
    ("Drop2_0.3",            _B3("batch","silu"),  dict(drop=0.3)),
    # ── Residual ─────────────────────────────────────────────────────────────
    ("Res0_no_residual",     _B3("batch","silu"),  dict(res=False)),
    # ── Best-guess combos ─────────────────────────────────────────────────── (informed by above)
    ("C1_4stage_group_gelu", [_s(32,2,"group","gelu"), _s(64,2,"group","gelu"), _s(128,3,"group","gelu"), _s(256,2,"group","gelu","avgpool")],
                             dict(lr=0.003, aug="medium")),
    ("C2_3stage_deep_strong",[_s(64,3,"batch","silu"), _s(128,3,"batch","silu"), _s(256,3,"batch","silu","avgpool")],
                             dict(lr=0.001, aug="strong")),
    ("C3_wide_lr001_strong", [_s(128,2,"batch","gelu"), _s(256,3,"batch","gelu","avgpool")],
                             dict(lr=0.001, aug="strong", drop=0.2)),
]


def build_specs():
    specs = []
    for entry in GRID:
        name   = entry[0]
        stages = entry[1]
        kwargs = entry[2] if len(entry) > 2 else {}
        specs.append((name, _spec(stages, **kwargs)))
    return specs


def _eval_worker(args):
    name, spec, max_steps, data_root, seed = args
    acc = evaluate_spec(spec, max_steps=max_steps, data_root=data_root, seed=seed)
    return name, acc


def run_sequential(specs, max_steps, data_root, seeds):
    results = {}
    for name, spec in specs:
        accs = []
        for seed in seeds:
            t0 = time.time()
            acc = evaluate_spec(spec, max_steps=max_steps, data_root=data_root, seed=seed)
            elapsed = time.time() - t0
            accs.append(acc)
            print(f"  {name:<30}  seed={seed}  acc={acc*100:.2f}%  ({elapsed:.0f}s)")
        results[name] = accs
    return results


def run_parallel(specs, max_steps, data_root, seeds):
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    work = [(name, spec, max_steps, data_root, seed)
            for name, spec in specs
            for seed in seeds]
    print(f"Launching {len(work)} jobs in parallel ...")
    with ctx.Pool(processes=len(work)) as pool:
        raw = pool.map(_eval_worker, work)
    results = {}
    for name, acc in raw:
        results.setdefault(name, []).append(acc)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",       default="./data")
    parser.add_argument("--results-dir",     default="./results")
    parser.add_argument("--max-train-steps", type=int, default=200)
    parser.add_argument("--seeds",           type=int, nargs="+", default=[0])
    parser.add_argument("--parallel",        action="store_true")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, "C0_grid_search.csv")
    specs = build_specs()

    print(f"Grid search: {len(specs)} configs × {len(args.seeds)} seed(s) "
          f"= {len(specs) * len(args.seeds)} runs  |  max_steps={args.max_train_steps}")

    t_total = time.time()
    if args.parallel:
        results = run_parallel(specs, args.max_train_steps, args.data_root, args.seeds)
    else:
        results = run_sequential(specs, args.max_train_steps, args.data_root, args.seeds)
    elapsed_total = time.time() - t_total

    print("\n" + "=" * 65)
    print(f"{'Config':<30}  {'mean':>7}  {'max':>7}  per-seed")
    print("=" * 65)
    rows = []
    for name, spec in specs:
        accs = results.get(name, [])
        if not accs:
            continue
        mean_acc = sum(accs) / len(accs)
        max_acc  = max(accs)
        seed_str = "  ".join(f"{a*100:.2f}%" for a in accs)
        print(f"{name:<30}  {mean_acc*100:>6.2f}%  {max_acc*100:>6.2f}%  {seed_str}")
        rows.append(dict(name=name, spec=spec.to_summary(),
                         mean_acc=f"{mean_acc:.4f}", max_acc=f"{max_acc:.4f}",
                         accs=";".join(f"{a:.4f}" for a in accs)))
    print("=" * 65)
    print(f"Total wall time: {elapsed_total/60:.1f} min")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "spec", "mean_acc", "max_acc", "accs"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
