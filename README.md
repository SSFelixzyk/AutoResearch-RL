# CIFAR-10 AutoML with GRPO

A lightweight reproduction of the AutoResearch-RL pipeline on CIFAR-10. An LLM agent (Qwen3-1.7B) proposes neural architecture and training-recipe configurations as structured JSON. A deterministic harness compiles each JSON into a PyTorch model, trains it for a fixed number of gradient steps, and returns the validation accuracy as the reward signal. Five controlled conditions isolate the contributions of history-aware prompting and online RL weight updates (GRPO).

---

## Motivation

The original AutoResearch-RL paper trains a PPO agent to write Python code diffs that improve a training script. This project simplifies the action space to a **structured JSON schema** (no free-form code generation), making reward evaluation deterministic and reproducible.

> Does giving the LLM its own past results (history) help? And does reinforcing the LLM with GRPO on top of that help further?

---

## Project Structure

```
cifar10_automl/
├── src/
│   ├── action_space.py      # JSON schema (Pydantic v2), random sampler, coerce_spec
│   ├── model_builder.py     # ArchSpec → nn.Module + optimizer + scheduler
│   ├── trainer.py           # Fixed-step CIFAR-10 training harness → val_acc
│   ├── history.py           # Sliding-window buffer + top-K buffer + prompt builder
│   ├── reward.py            # Shaped reward: relative, novelty bonus, coerce penalty
│   ├── agent.py             # Qwen3 batched inference, JSON extraction, thinking split
│   ├── grpo.py              # LoRA setup, advantage computation, gradient update
│   └── loop.py              # Core research loop shared by all conditions
├── experiments/
│   ├── run_all.py           # CLI entry point — runs any subset of C0–C5
│   ├── grid_search.py       # Standalone grid search (also invoked as C0)
│   └── validate_top_configs.py  # Post-hoc stability: re-evaluate top-K configs × N seeds
├── tests/
│   ├── test_action_space.py
│   ├── test_model_builder.py
│   ├── test_trainer.py
│   ├── test_history.py
│   └── test_agent.py
├── program.md               # System prompt / action space spec shown to the LLM
├── config.yaml              # Default experiment configuration
├── run.sh                   # Single-experiment launcher (reads config.yaml)
├── launch.sh                # Multi-condition parallel launcher (C0 + C5 on separate GPUs)
├── validate.sh              # Post-hoc stability validation launcher
└── results/                 # Auto-created; one named subfolder per run
    └── conds-05--steps-60--maxsteps-300--lora-16--.../ 
        ├── C0_grid_search.csv
        ├── C5_grpo_G4_steps.csv
        ├── config.yaml              # copy of the config used
        ├── config_effective.json    # all merged CLI + YAML args
        └── all_histories.json
```

---

## Action Space

The LLM proposes configurations as JSON. All fields are discrete, making the space fully enumerable.

```
ArchSpec
├── stages: List[StageSpec]  (1–4 stages, ordered)
│   ├── out_channels : 32 | 64 | 128 | 256
│   ├── num_blocks   : 1 | 2 | 3
│   ├── norm         : none | batch | group
│   ├── activation   : relu | gelu | silu
│   └── downsample   : stride | maxpool | avgpool
├── use_residual : true | false
├── dropout      : 0.0 | 0.1 | 0.2 | 0.3 | 0.5
├── optimizer:
│   ├── type         : sgd | adam | adamw
│   ├── lr           : 0.001 | 0.003 | 0.01
│   ├── weight_decay : 0.0 | 0.001 | 0.01 | 0.05
│   └── momentum     : 0.9 | 0.95   (sgd only)
├── scheduler : none | cosine | step
└── augment   : none | basic | medium | strong
```

### Action Space Size

```
Per stage:  channels(4) × blocks(3) × norm(3) × act(3) × downsample(3) = 324

Stage configurations (ordered, 1–4 stages):
  1 stage:  324^1 =             324
  2 stages: 324^2 =         104,976
  3 stages: 324^3 =      34,012,224
  4 stages: 324^4 =  11,019,960,576
  Total:             ≈ 1.1 × 10^10

Non-architecture parameters:
  residual(2) × dropout(5) × scheduler(3) × augment(4) = 120
  optimizer: SGD(3×4×2) + Adam(3×4) + AdamW(3×4) = 48
  Non-arch total: 120 × 48 = 5,760

Total space: 1.1 × 10^10 × 5,760 ≈ 6.4 × 10^13
```

The 4-stage case dominates (99.9% of the space). With 60 steps × 4 candidates = 240 evaluations, C5 samples roughly **3.7 × 10⁻¹⁰ %** of the full space. Grid search or exhaustive enumeration of the ceiling is not feasible; the key comparison is between search strategies (C1 through C5).

---

## Experimental Conditions

All conditions share the same loop code, differing only in arguments.

| ID | Name | G | History | Weight Update | Purpose |
|----|------|---|---------|---------------|---------|
| **C0** | Grid Search | — | No | No | Strong human-designed baseline |
| **C1** | Random Search | 1 | No | No | Baseline: pure random |
| **C2** | LLM, no history | 1 | No | No | Does the LLM beat random? |
| **C3** | LLM + history | 1 | Yes | No | Does history context help? |
| **C4** | LLM + history, G=4 | 4 | Yes | No | Does best-of-G help? |
| **C5** | GRPO, G=4 | 4 | Yes | Yes (LoRA) | Does RL weight update further improve? |

**Primary metric**: best validation accuracy so far, as a function of research step.

**Ablation logic** (each step isolates one variable):
- C1 → C2: does the LLM prior help at all?
- C2 → C3: does seeing past results in-context help?
- C3 → C4: does sampling more candidates help?
- C4 → C5: does GRPO (updating weights) add value beyond best-of-G?

---

## Reward Design

Shaped rewards are computed after each G-rollout and passed to the GRPO update:

```python
# For invalid JSON output:
r = -invalid_penalty          # (default: -0.1)

# For valid but hallucinated field values (e.g. norm="instance"):
r = acc - best_so_far         # coerced to valid nearest value, small extra penalty

# For valid candidates:
r = acc - best_so_far         # relative improvement over current best
r += novelty_coef * min_dist  # bonus for diverse proposals (Hamming distance to recent)
# no floor — allows graded negative signal when all candidates are below best
```

Key design choices:
- **Relative reward** (`acc - best_so_far`): incentivises improvement, not absolute accuracy
- **No reward floor**: allows GRPO to distinguish "slightly below best" from "terrible config"; a floor of 0 collapses all below-best rewards to zero and kills the gradient signal after a good config is found
- **Novelty bonus**: encourages exploration when the model starts repeating configurations
- **Invalid penalty exempt from floor**: the model still receives -0.1 for broken JSON regardless of best_so_far

---

## Module Overview

### `src/action_space.py`

- `ArchSpec` / `StageSpec` / `OptimizerSpec`: Pydantic v2 models with `Literal` fields
- `validate_spec(raw)`: strict validation, returns `None` on any error
- `coerce_spec(raw)`: lenient — snaps numeric fields to nearest valid value, strings fall back to first valid option; returns `None` only for structurally broken input
- `sample_random_spec(seed)`: uniform random sample for C1 baseline

### `src/model_builder.py`

Converts `ArchSpec` → `nn.Module` (`CIFAR10Net`):
- Stem conv → per-stage blocks → global average pool → dropout → linear(10)
- `ResBlock` (skip connection) or `PlainBlock` depending on `use_residual`
- Group norm groups = min(32, channels); falls back to 1 if channels < 2

### `src/trainer.py`

`evaluate_spec(spec, max_steps, data_root, seed) -> float`

Trains for exactly `max_steps` gradient steps (not epochs). All architectures get the same compute budget regardless of size. Seeds PyTorch + NumPy + Python random for reproducibility.

### `src/history.py`

`HistoryBuffer(max_k, top_k)` maintains two views:
- **Recent**: last `max_k` experiments (sliding window)
- **Top**: best `top_k` experiments by accuracy (may overlap with recent)

`build_prompt(program_md, use_history)` constructs the full LLM prompt. Setting `max_k=0` silently disables history (no spurious "No experiments yet" messages).

### `src/reward.py`

`compute_rewards(raw_candidates, accs, ...)` returns shaped rewards and per-component stats for wandb logging (`reward_mean`, `novelty_mean`, `invalid_count`).

`spec_distance(a, b)` computes normalised Hamming distance between two `ArchSpec` objects for novelty computation.

### `src/agent.py`

`LLMAgent.generate_specs(prompt, n, temperature)` generates N completions in a single batched `model.generate` call (inputs repeated n times). Handles Qwen3 thinking tokens (`</think>`) by splitting on the last occurrence. Logs thinking content and extracted JSON to a JSONL file.

### `src/grpo.py`

Manual GRPO (no verl/trl dependency — our reward requires ~60s of CIFAR-10 training per sample, incompatible with synchronous reward frameworks).

```
A_i = (r_i - mean(r)) / (std(r) + ε)   # group-relative advantage
loss = -mean(A_i × log π_θ(completion_i | prompt))
     + β × KL(π_θ || π_ref)             # optional KL penalty
```

LoRA (r=16, α=32) applied to `q_proj` and `v_proj`. `use_cache=False` in log-prob computation prevents KV cache accumulation (~500MB/call) during training.

### `src/loop.py`

`run_research_loop(...)` is called by all conditions. Per step:
1. Build prompt from history
2. Generate G candidates (via `generate_fn`, or random if `None`)
3. Evaluate G candidates (parallel `mp.spawn` or sequential)
4. Compute shaped rewards (snapshot history before update, so `best_so_far` is pre-step)
5. Update history with all G results
6. Call `grpo_update_fn` if provided (C5 only)
7. Log to CSV + wandb

---

## Setup

```bash
# 1. Clone
git clone https://github.com/SSFelixzyk/AutoResearch-RL.git
cd AutoResearch-RL

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run tests (no GPU, no model download needed)
pytest tests/ -v   # expected: 23 passed

# 4. CIFAR-10 (~170MB) is auto-downloaded on first run
```

**Requirements**: Python 3.10+, PyTorch 2.2+, CUDA GPU recommended. For C2–C5, Qwen3-1.7B (~3.4GB) is downloaded from HuggingFace on first run.

---

## Running Experiments

All commands run from the `cifar10_automl/` directory.

### Recommended: use config.yaml + run.sh

```bash
# Edit config.yaml to set conditions, n_steps, max_train_steps, etc.
chmod +x run.sh
./run.sh config.yaml

# Results auto-saved to a named subfolder:
# results/conds-05--steps-60--maxsteps-300--lora-16--invalid-0.1--relative--novel-0.02--floor-null/
```

### Run grid search + GRPO in parallel (two GPUs)

```bash
chmod +x launch.sh
GPU0=0 GPU1=1 ./launch.sh   # C0 on GPU 0, C5 on GPU 1
tail -f logs/C0.log
tail -f logs/C5.log
```

### Single GPU: run sequentially

```bash
./run.sh config.yaml --conditions C0 && ./run.sh config.yaml --conditions C5
```

### Post-hoc stability validation

```bash
chmod +x validate.sh

# Auto-find latest results folder, validate top-5 configs across 5 seeds
./validate.sh

# Explicit folder, top-3, 3 seeds
./validate.sh results/conds-05--steps-60-... --top-k 3 --seeds 0 1 2
```

### Smoke test (~5 min, no GPU required)

```bash
python experiments/run_all.py \
    --conditions C1 \
    --n-steps 3 \
    --max-train-steps 20 \
    --no-parallel \
    --wandb-project disabled
```

### CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--conditions` | `C1 C2 C3 C4 C5` | Which conditions to run (C0–C5) |
| `--n-steps` | `60` | Research steps per condition |
| `--max-train-steps` | `300` | Gradient steps per candidate evaluation |
| `--model` | `Qwen/Qwen3-1.7B` | HuggingFace model for C2–C5 |
| `--data-root` | `./data` | CIFAR-10 download path |
| `--results-dir` | auto | Output directory (auto-named from config if not set) |
| `--lora-r` | `16` | LoRA rank for C5 |
| `--lora-alpha` | `32` | LoRA scaling factor |
| `--history-k` | `5` | Recent experiments shown in prompt |
| `--history-top-k` | `3` | Top experiments also shown in prompt |
| `--invalid-penalty` | `0.1` | Penalty for invalid JSON output |
| `--use-relative-reward` | off | `r = acc - best_so_far` instead of raw `acc` |
| `--novelty-coef` | `0.02` | Diversity bonus coefficient |
| `--reward-floor` | `null` | Floor for valid-candidate rewards (null = no floor) |
| `--use-kl-penalty` | off | Add KL(π_θ ‖ π_ref) penalty to GRPO loss |
| `--kl-coef` | `0.1` | β for KL term |
| `--no-parallel` | off | Force sequential evaluation |
| `--wandb-project` | `cifar10-automl` | W&B project; `disabled` to skip |

---

## Outputs

Each run creates a named subfolder under `results/`:

```
results/conds-05--steps-60--maxsteps-300--lora-16--invalid-0.1--relative--novel-0.02--floor-null/
├── config.yaml                  # copy of the config file used
├── config_effective.json        # all merged CLI + YAML args (for exact reproducibility)
├── C0_grid_search.csv           # grid search results (if C0 was run)
├── C5_grpo_G4_steps.csv         # per-candidate log: step, acc, is_best, spec_summary
├── C5_outputs.jsonl             # raw LLM outputs: thinking, answer, extracted JSON
├── all_histories.json           # best-acc-so-far list per condition
└── top5_validated_5seeds.csv    # (after validate.sh) stability results
```

CSV columns: `step, candidate, acc, is_best, best_so_far, spec_summary, wall_time`

### wandb metrics (per condition, per step)

| Key | Description |
|-----|-------------|
| `best_acc` | Best accuracy found so far |
| `step_best_acc` | Best among G candidates this step |
| `step_mean_acc` | Mean accuracy of G candidates |
| `wall_time_s` | Wall clock time for this step |
| `improved` | 1 if new best found, else 0 |
| `reward_mean` | Mean shaped reward (for GRPO signal monitoring) |
| `reward_novelty_mean` | Mean novelty bonus |
| `reward_invalid_count` | Number of invalid JSON outputs this step |
| `grpo_loss` | GRPO loss (C5 only) |

---

## Runtime Estimates (single A100 80GB, max_train_steps=300)

| Condition | Steps | Evaluations | Est. Time |
|-----------|-------|-------------|-----------|
| C0 Grid Search | 27 configs | 27 | ~30 min |
| C1 Random | 60 | 60 | ~45 min |
| C2 LLM G=1 | 60 | 60 | ~1h |
| C3 LLM G=1 | 60 | 60 | ~1h |
| C4 LLM G=4 | 60 | 240 | ~3h |
| C5 GRPO G=4 | 60 | 240 + updates | ~4h |

Each 300-step CIFAR-10 evaluation takes ~40s on A100. C5 forces sequential eval (LLM + 4 CNN trainings would exceed 24GB VRAM simultaneously).

---

## Design Notes

**Why JSON instead of code diffs?** Structured JSON makes the action space fully enumerable and the harness deterministic. The LLM cannot generate unrunnable code, and every output is either a valid `ArchSpec` or is coerced to the nearest valid value (with a small penalty) or falls back to random sampling.

**Why `coerce_spec` instead of hard rejection?** Hard rejection on slightly out-of-range numerics wastes evaluations. `coerce_spec` snaps values to the nearest allowed option so the config is always evaluated. String fields that are simply wrong (e.g. `norm: "instance"`) are coerced but flagged for a penalty so GRPO still learns to avoid them.

**Why no reward floor?** A floor of 0 with relative rewards collapses all below-best candidates to the same reward (0) once a good config is found. GRPO cannot distinguish "slightly bad" from "catastrophically bad" — gradient goes to zero. Without a floor, graded negative rewards allow continuous learning even after finding a local optimum.

**Why manual GRPO instead of verl/trl?** Our reward requires ~40–60s of CIFAR-10 training per sample. verl and trl assume millisecond rewards. Implementing GRPO manually (~100 lines) is simpler and avoids framework mismatch.

**Why fixed gradient steps instead of epochs?** Every architecture gets exactly the same compute budget regardless of model size, making accuracy directly comparable across configurations.

**Why all G results go into history, not just the best?** The LLM learns from failures. Seeing that `C64x3(batch,relu,stride)` got 42% while `C128x2(group,gelu,avgpool)` got 61% is more informative than only seeing the winner.

**Why batched generation (G candidates in one forward pass)?** Sequential generation for G=4 means 4× LLM inference latency per step. Batching with `inputs.repeat(n, 1)` generates all G candidates in a single `model.generate` call, significantly reducing per-step wall time.
