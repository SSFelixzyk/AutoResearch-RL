# experiments/run_all.py
"""
Usage:
  ./run.sh                                      # reads config.yaml
  ./run.sh config.yaml --conditions C1 C2       # override specific args
  python experiments/run_all.py --config config.yaml
  python experiments/run_all.py --conditions C1 --n-steps 5 --max-train-steps 50
"""
import argparse
import csv
import json
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.loop import run_research_loop


# ── C1 parallel worker (module-level so mp can pickle it) ────────────────────
def _c1_worker(kwargs):
    return run_research_loop(**kwargs)


# ── C2 batched-run loop ───────────────────────────────────────────────────────
def _run_c2_batched(generate_fn, n_runs, seeds, G, program_md,
                    results_dir, data_root, max_train_steps,
                    history_k, history_top_k,
                    invalid_penalty, use_relative_reward,
                    novelty_coef, reward_floor, n_steps,
                    wandb_run, parallel_eval):
    """
    Run C2 (no-history) for n_runs simultaneously.
    At each step every run uses the same prompt → generate G*n_runs candidates
    in one forward pass, then split and evaluate independently per run.
    """
    from pathlib import Path
    from src.action_space import sample_random_spec
    from src.history import HistoryBuffer
    from src.reward import compute_rewards
    from src.trainer import evaluate_spec
    from src.loop import evaluate_parallel

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    bufs = [HistoryBuffer(max_k=history_k, top_k=history_top_k) for _ in range(n_runs)]
    run_histories = [[] for _ in range(n_runs)]

    csv_files, writers = [], []
    for run_i in range(n_runs):
        name = f"C2_llm_nohist_G1_run{run_i}"
        p = open(os.path.join(results_dir, f"{name}_steps.csv"), "w", newline="")
        w = csv.writer(p)
        w.writerow(["step", "candidate", "acc", "is_best",
                    "best_so_far", "spec_summary", "wall_time"])
        csv_files.append(p)
        writers.append(w)

    for step in range(n_steps):
        t0 = time.time()
        # All runs share the same prompt (no history)
        prompt = bufs[0].build_prompt(program_md, use_history=False)

        # One LLM call for all runs combined
        all_raw = generate_fn(prompt, n=G * n_runs)
        wall = time.time() - t0

        for run_i in range(n_runs):
            seed_i = seeds[run_i]
            raw_cands = all_raw[run_i * G:(run_i + 1) * G]
            candidates = [
                c if c is not None else sample_random_spec(seed=seed_i + step * G + i)
                for i, c in enumerate(raw_cands)
            ]

            if parallel_eval and G > 1:
                accs = evaluate_parallel(candidates, max_train_steps, data_root, seed_i, G)
            else:
                accs = [evaluate_spec(s, max_steps=max_train_steps,
                                      data_root=data_root, seed=seed_i)
                        for s in candidates]

            prev_best = bufs[run_i].best_acc
            shaped, _ = compute_rewards(
                raw_cands, accs, best_so_far=prev_best,
                recent_specs=[e.spec for e in bufs[run_i].recent],
                invalid_penalty=invalid_penalty,
                use_relative=use_relative_reward,
                novelty_coef=novelty_coef,
                reward_floor=reward_floor,
            )
            for i, (spec, acc) in enumerate(zip(candidates, accs)):
                is_new = acc > bufs[run_i].best_acc
                bufs[run_i].add(spec, acc)
                writers[run_i].writerow([step, i, f"{acc:.4f}", is_new,
                                         f"{bufs[run_i].best_acc:.4f}",
                                         spec.to_summary(), f"{wall:.1f}"])
            run_histories[run_i].append(bufs[run_i].best_acc)
            csv_files[run_i].flush()

        best_across = max(b.best_acc for b in bufs)
        print(f"[C2_batched] step={step:3d}  "
              f"best_across_runs={best_across*100:.2f}%  wall={wall:.0f}s")

        if wandb_run is not None:
            for run_i in range(n_runs):
                wandb_run.log({
                    f"C2_llm_nohist_G1_run{run_i}/step": step,
                    f"C2_llm_nohist_G1_run{run_i}/best_acc": run_histories[run_i][-1],
                })

    for f in csv_files:
        f.close()
    return run_histories


def load_program_md():
    path = os.path.join(os.path.dirname(__file__), "..", "program.md")
    with open(path) as f:
        return f.read()


def load_yaml_config(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print("[warn] PyYAML not installed — run: pip install pyyaml")
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    # ── Phase 1: read --config before building full parser ──────────────────
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()

    yaml_defaults = {}
    if pre_args.config:
        raw = load_yaml_config(pre_args.config)
        # normalise keys: yaml uses snake_case, argparse dest also uses snake_case
        yaml_defaults = {k.replace("-", "_"): v for k, v in raw.items()}

    # ── Phase 2: full parser — YAML values become defaults, CLI overrides ───
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--max-train-steps", type=int, default=500)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--conditions", nargs="+", default=["C1", "C2", "C3", "C4", "C5"],
                        help="Conditions to run: C0 (grid search) C1 C2 C3 C4 C5")
    parser.add_argument("--use-kl-penalty", action="store_true", default=False)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--history-k", type=int, default=5)
    parser.add_argument("--history-top-k", type=int, default=3)
    parser.add_argument("--invalid-penalty", type=float, default=0.1)
    parser.add_argument("--use-relative-reward", action="store_true", default=False)
    parser.add_argument("--novelty-coef", type=float, default=0.0)
    parser.add_argument("--reward-floor", type=float, default=None)
    parser.add_argument("--parallel-eval", action="store_true", default=True)
    parser.add_argument("--no-parallel", action="store_true", default=False)
    parser.add_argument("--wandb-project", default="cifar10-automl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Independent repetitions per condition with different seeds")
    parser.add_argument("--parallel-runs", action="store_true", default=False,
                        help="C1: run all n_runs in parallel processes; "
                             "C2: batch G*n_runs candidates per step in one LLM call")
    parser.set_defaults(**yaml_defaults)
    args = parser.parse_args()

    # --no-parallel (CLI flag) takes priority over parallel_eval from YAML
    parallel_eval = args.parallel_eval and not args.no_parallel
    program_md = load_program_md()

    # ── Auto-generate a descriptive results subdirectory ─────────────────────
    # Only kicks in when results_dir is still the bare default ("./results").
    # CLI --results-dir always wins.
    _results_dir_is_default = (
        args.results_dir in ("./results", "results")
        and "--results-dir" not in sys.argv
    )
    if _results_dir_is_default:
        conds_tag = "".join(c.replace("C", "") for c in sorted(args.conditions))
        parts = [
            f"conds-{conds_tag}",
            f"steps-{args.n_steps}",
            f"maxsteps-{args.max_train_steps}",
        ]
        if "C5" in args.conditions:
            parts.append(f"lora-{args.lora_r}")
        parts.append(f"invalid-{args.invalid_penalty}")
        if args.use_relative_reward:
            parts.append("relative")
        if args.novelty_coef > 0:
            parts.append(f"novel-{args.novelty_coef}")
        floor_tag = "null" if args.reward_floor is None else str(args.reward_floor)
        parts.append(f"floor-{floor_tag}")
        if args.n_runs > 1:
            parts.append(f"runs-{args.n_runs}")
        args.results_dir = os.path.join("results", "--".join(parts))

    os.makedirs(args.results_dir, exist_ok=True)

    # ── Save effective config into the results folder ─────────────────────────
    _cfg_dst = os.path.join(args.results_dir, "config_effective.json")
    with open(_cfg_dst, "w") as _f:
        json.dump(vars(args), _f, indent=2)
    if pre_args.config and os.path.isfile(pre_args.config):
        import shutil as _shutil
        _shutil.copy2(pre_args.config, os.path.join(args.results_dir, "config.yaml"))
    print(f"[run] results_dir : {args.results_dir}")
    print(f"[run] conditions  : {args.conditions}")

    # wandb init
    wandb_run = None
    if args.wandb_project != "disabled":
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"n{args.n_steps}_steps{args.max_train_steps}",
            )
            # Give each condition its own step axis so logs don't collide
            for _cond in ["C1_random", "C2_llm_nohist_G1", "C3_llm_hist_G1",
                          "C4_llm_hist_G4", "C5_grpo_G4"]:
                wandb_run.define_metric(f"{_cond}/step")
                wandb_run.define_metric(f"{_cond}/*", step_metric=f"{_cond}/step")
        except Exception as e:
            print(f"[warn] wandb init failed: {e} — continuing without wandb")

    common = dict(
        n_steps=args.n_steps,
        program_md=program_md,
        results_dir=args.results_dir,
        data_root=args.data_root,
        max_train_steps=args.max_train_steps,
        parallel_eval=parallel_eval,
        wandb_run=wandb_run,
        seed=args.seed,
        history_k=args.history_k,
        history_top_k=args.history_top_k,
        invalid_penalty=args.invalid_penalty,
        use_relative_reward=args.use_relative_reward,
        novelty_coef=args.novelty_coef,
        reward_floor=args.reward_floor,
    )

    all_histories = {}

    # C0: Grid Search (oracle ceiling — no LLM needed)
    # Covers all major action-space dimensions: architecture depth/width, norm,
    # activation, optimizer, lr, scheduler, augment, dropout, residual.
    # Strategy: one-factor-at-a-time around a strong base, plus a few combos.
    if "C0" in args.conditions:
        print("\n=== C0: Grid Search (oracle ceiling) ===")
        import csv as _csv
        from src.action_space import ArchSpec, OptimizerSpec, StageSpec, coerce_spec
        from src.trainer import evaluate_spec as _eval_spec

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
                optimizer=OptimizerSpec(type=opt, lr=lr, weight_decay=wd,  # type: ignore[arg-type]
                                        momentum=mom),
                scheduler=sched,  # type: ignore[arg-type]
                augment=aug,      # type: ignore[arg-type]
            )

        # Shared base stages for single-factor sweeps
        _B3 = lambda norm, act: [  # 3-stage 64→128→256
            _s(64,  2, norm, act, "stride"),
            _s(128, 3, norm, act, "stride"),
            _s(256, 2, norm, act, "avgpool"),
        ]

        _grid = [
            # ── Architecture: depth / width ──────────────────────────────────
            ("A1_2stage_64-128",   [_s(64,2,"batch","silu"), _s(128,3,"batch","silu","avgpool")]),
            ("A2_3stage_64-256",   _B3("batch","silu")),       # base
            ("A3_3stage_32-128",   [_s(32,2,"batch","silu"), _s(64,3,"batch","silu"), _s(128,2,"batch","silu","avgpool")]),
            ("A4_4stage_32-256",   [_s(32,2,"batch","silu"), _s(64,2,"batch","silu"), _s(128,2,"batch","silu"), _s(256,2,"batch","silu","avgpool")]),
            ("A5_3stage_deep",     [_s(64,3,"batch","silu"), _s(128,3,"batch","silu"), _s(256,3,"batch","silu","avgpool")]),
            ("A6_wide_128-256",    [_s(128,2,"batch","silu"), _s(256,3,"batch","silu","avgpool")]),
            # ── Norm ─────────────────────────────────────────────────────────
            ("N1_group",           _B3("group","silu")),
            ("N2_none",            _B3("none", "silu")),
            # ── Activation ───────────────────────────────────────────────────
            ("Act1_relu",          _B3("batch","relu")),
            ("Act2_gelu",          _B3("batch","gelu")),
            # ── Learning rate ─────────────────────────────────────────────── (base lr=0.003)
            ("LR1_0.001",          _B3("batch","silu"), dict(lr=0.001)),
            ("LR3_0.01",           _B3("batch","silu"), dict(lr=0.01)),
            # ── Weight decay ─────────────────────────────────────────────────
            ("WD1_0.001",          _B3("batch","silu"), dict(wd=0.001)),
            ("WD2_0.05",           _B3("batch","silu"), dict(wd=0.05)),
            # ── Optimizer ────────────────────────────────────────────────────
            ("Opt1_adam",          _B3("batch","silu"), dict(opt="adam")),
            ("Opt2_sgd_mom0.9",    _B3("batch","silu"), dict(opt="sgd", mom=0.9, lr=0.003)),
            # ── Scheduler ────────────────────────────────────────────────────
            ("Sched1_step",        _B3("batch","silu"), dict(sched="step")),
            ("Sched2_none",        _B3("batch","silu"), dict(sched="none")),
            # ── Augmentation ──────────────────────────────────────────────── (base aug=medium)
            ("Aug1_basic",         _B3("batch","silu"), dict(aug="basic")),
            ("Aug2_strong",        _B3("batch","silu"), dict(aug="strong")),
            # ── Dropout ──────────────────────────────────────────────────────
            ("Drop1_0.2",          _B3("batch","silu"), dict(drop=0.2)),
            ("Drop2_0.3",          _B3("batch","silu"), dict(drop=0.3)),
            # ── No residual ───────────────────────────────────────────────── (base res=True)
            ("Res0_no_residual",   _B3("batch","silu"), dict(res=False)),
            # ── Best-guess combos ─────────────────────────────────────────── (informed by above)
            ("C1_4stage_group_gelu",  [_s(32,2,"group","gelu"), _s(64,2,"group","gelu"), _s(128,3,"group","gelu"), _s(256,2,"group","gelu","avgpool")], dict(lr=0.003, aug="medium")),
            ("C2_3stage_deep_strong", [_s(64,3,"batch","silu"), _s(128,3,"batch","silu"), _s(256,3,"batch","silu","avgpool")], dict(lr=0.001, aug="strong")),
            ("C3_wide_lr001_strong",  [_s(128,2,"batch","gelu"), _s(256,3,"batch","gelu","avgpool")], dict(lr=0.001, aug="strong", drop=0.2)),
        ]

        os.makedirs(args.results_dir, exist_ok=True)
        c0_csv = os.path.join(args.results_dir, "C0_grid_search.csv")
        c0_rows = []
        c0_best = 0.0

        with open(c0_csv, "w", newline="") as _f:
            _w = _csv.writer(_f)
            _w.writerow(["step", "name", "spec_summary", "acc", "wall_time"])

            for step, entry in enumerate(_grid):
                name   = entry[0]
                stages = entry[1]
                kwargs = entry[2] if len(entry) > 2 else {}
                spec = _spec(stages, **kwargs)
                t0 = time.time()
                acc = _eval_spec(spec, max_steps=args.max_train_steps,
                                 data_root=args.data_root, seed=args.seed)
                wall = time.time() - t0
                c0_best = max(c0_best, acc)
                _w.writerow([step, name, spec.to_summary(), f"{acc:.4f}", f"{wall:.1f}"])
                _f.flush()
                c0_rows.append((name, acc))
                print(f"  [{step+1:02d}/{len(_grid)}] {name:<30}  acc={acc*100:.2f}%  "
                      f"best={c0_best*100:.2f}%  ({wall:.0f}s)")

                if wandb_run is not None:
                    wandb_run.log({
                        "C0_grid/step": step,
                        "C0_grid/acc": acc,
                        "C0_grid/best_acc": c0_best,
                        "C0_grid/wall_time_s": wall,
                    })

        print(f"\n  [C0] Best: {c0_best*100:.2f}%  — results saved to {c0_csv}")
        all_histories["C0_grid"] = [r[1] for r in c0_rows]

    def _run_common(run_i):
        """Return a copy of common with the per-run seed."""
        seed = args.seed + run_i * 1000
        return {**common, "seed": seed}

    def _suffix(run_i):
        return f"_run{run_i}" if args.n_runs > 1 else ""

    def _run_header(cond_label, run_i):
        extra = f" (run {run_i + 1}/{args.n_runs})" if args.n_runs > 1 else ""
        print(f"\n=== {cond_label}{extra} ===")

    # C1: Random Search
    if "C1" in args.conditions:
        if args.parallel_runs and args.n_runs > 1:
            print(f"\n=== C1: Random Search  ({args.n_runs} runs in parallel) ===")
            import torch.multiprocessing as _mp
            ctx = _mp.get_context("spawn")
            worker_kwargs = [
                dict(condition_name=f"C1_random_run{run_i}",
                     G=1, use_history=False, generate_fn=None,
                     wandb_run=None,          # wandb not picklable
                     **{k: v for k, v in _run_common(run_i).items()
                        if k != "wandb_run"})
                for run_i in range(args.n_runs)
            ]
            with ctx.Pool(args.n_runs) as pool:
                results = pool.map(_c1_worker, worker_kwargs)
            for run_i, history in enumerate(results):
                all_histories[f"C1_random_run{run_i}"] = history
        else:
            for run_i in range(args.n_runs):
                _run_header("C1: Random Search", run_i)
                key = f"C1_random{_suffix(run_i)}"
                history = run_research_loop(
                    condition_name=key,
                    G=1, use_history=False, generate_fn=None,
                    **_run_common(run_i)
                )
                all_histories[key] = history

    # C2: LLM no-history G=1
    if "C2" in args.conditions:
        from src.agent import LLMAgent
        agent = LLMAgent(model_name=args.model,
                         output_log=os.path.join(args.results_dir, "C2_outputs.jsonl"))
        if args.parallel_runs and args.n_runs > 1:
            print(f"\n=== C2: LLM no-history G=1  "
                  f"({args.n_runs} runs batched, G×{args.n_runs} per step) ===")
            seeds = [args.seed + run_i * 1000 for run_i in range(args.n_runs)]
            run_histories = _run_c2_batched(
                generate_fn=agent.generate_specs,
                n_runs=args.n_runs, seeds=seeds, G=1,
                program_md=program_md,
                results_dir=args.results_dir,
                data_root=args.data_root,
                max_train_steps=args.max_train_steps,
                history_k=args.history_k, history_top_k=args.history_top_k,
                invalid_penalty=args.invalid_penalty,
                use_relative_reward=args.use_relative_reward,
                novelty_coef=args.novelty_coef,
                reward_floor=args.reward_floor,
                n_steps=args.n_steps,
                wandb_run=wandb_run,
                parallel_eval=parallel_eval,
            )
            for run_i, h in enumerate(run_histories):
                all_histories[f"C2_llm_nohist_G1_run{run_i}"] = h
        else:
            for run_i in range(args.n_runs):
                _run_header("C2: LLM no-history G=1", run_i)
                key = f"C2_llm_nohist_G1{_suffix(run_i)}"
                history = run_research_loop(
                    condition_name=key,
                    G=1, use_history=False,
                    generate_fn=agent.generate_specs,
                    **_run_common(run_i)
                )
                all_histories[key] = history
        del agent

    # C3: LLM history G=1  (agent loaded once, reused across runs)
    if "C3" in args.conditions:
        from src.agent import LLMAgent
        agent = LLMAgent(model_name=args.model,
                         output_log=os.path.join(args.results_dir, "C3_outputs.jsonl"))
        for run_i in range(args.n_runs):
            _run_header("C3: LLM history G=1", run_i)
            key = f"C3_llm_hist_G1{_suffix(run_i)}"
            history = run_research_loop(
                condition_name=key,
                G=1, use_history=True,
                generate_fn=agent.generate_specs,
                **_run_common(run_i)
            )
            all_histories[key] = history
        del agent

    # C4: LLM history G=4  (agent loaded once, reused across runs)
    if "C4" in args.conditions:
        from src.agent import LLMAgent
        agent = LLMAgent(model_name=args.model,
                         output_log=os.path.join(args.results_dir, "C4_outputs.jsonl"))
        for run_i in range(args.n_runs):
            _run_header("C4: LLM history G=4 (best-of-G)", run_i)
            key = f"C4_llm_hist_G4{_suffix(run_i)}"
            history = run_research_loop(
                condition_name=key,
                G=4, use_history=True,
                generate_fn=agent.generate_specs,
                grpo_update_fn=None,
                **_run_common(run_i)
            )
            all_histories[key] = history
        del agent

    # C5: GRPO G=4  (fresh LoRA weights per run — each run is an independent training)
    if "C5" in args.conditions:
        import torch
        from src.grpo import setup_grpo_model, grpo_update
        from src.agent import extract_json, format_prompt, decode_thinking_and_answer
        from src.action_space import coerce_spec

        for run_i in range(args.n_runs):
            _run_header("C5: GRPO G=4", run_i)
            suf = _suffix(run_i)

            model, ref_model, tokenizer, optimizer = setup_grpo_model(
                args.model,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                use_kl_penalty=args.use_kl_penalty,
            )

            os.makedirs(args.results_dir, exist_ok=True)
            _c5_log = open(
                os.path.join(args.results_dir, f"C5_outputs{suf}.jsonl"),
                "w", encoding="utf-8"
            )
            _c5_call = 0

            def grpo_generate(prompt, n, _log=_c5_log):
                nonlocal _c5_call
                formatted = format_prompt(tokenizer, prompt)
                inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
                prompt_len = inputs["input_ids"].shape[1]
                inputs_batch = {k: v.repeat(n, 1) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs_batch, max_new_tokens=2048,
                        do_sample=True, temperature=0.8,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                results = []
                for i in range(n):
                    output_ids = outputs[i][prompt_len:].tolist()
                    thinking, answer = decode_thinking_and_answer(tokenizer, output_ids)
                    raw = extract_json(answer)
                    spec = coerce_spec(raw) if raw else None
                    _log.write(json.dumps({
                        "call": _c5_call, "candidate": i,
                        "thinking_chars": len(thinking),
                        "thinking": thinking,
                        "answer": answer, "extracted": raw, "valid": spec is not None,
                    }, ensure_ascii=False) + "\n")
                    _log.flush()
                    _c5_call += 1
                    results.append(spec)
                return results

            def grpo_update_fn(prompt, candidates, accs):
                formatted = format_prompt(tokenizer, prompt)
                metrics = grpo_update(
                    model, ref_model, tokenizer, optimizer,
                    formatted, candidates, accs,
                    kl_coef=args.kl_coef,
                )
                print(f"  [GRPO] loss={metrics['loss']:.4f}  "
                      f"pg={metrics['pg_loss']:.4f}  kl={metrics['kl_loss']:.4f}")
                return metrics["loss"]

            key = f"C5_grpo_G4{suf}"
            c5_common = {**_run_common(run_i), "parallel_eval": False}
            history = run_research_loop(
                condition_name=key,
                G=4, use_history=True,
                generate_fn=grpo_generate,
                grpo_update_fn=grpo_update_fn,
                **c5_common
            )
            all_histories[key] = history
            _c5_log.close()

            save_path = os.path.join(args.results_dir, f"C5_lora_final{suf}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  [C5] LoRA weights saved to {save_path}")
            del model, ref_model, optimizer

    # Save summary
    os.makedirs(args.results_dir, exist_ok=True)
    summary_path = os.path.join(args.results_dir, "all_histories.json")
    with open(summary_path, "w") as f:
        json.dump(all_histories, f, indent=2)
    print(f"\nAll results saved to {args.results_dir}/")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
