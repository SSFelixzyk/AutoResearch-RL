# experiments/run_all.py
"""
Usage:
  python experiments/run_all.py [--n-steps 60] [--max-train-steps 500] \
      [--model Qwen/Qwen3-1.7B] [--data-root ./data] [--results-dir ./results] \
      [--conditions C1 C2 C3 C4 C5] [--use-kl-penalty] [--kl-coef 0.1] \
      [--no-parallel] [--wandb-project cifar10-automl]
"""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.loop import run_research_loop


def load_program_md():
    path = os.path.join(os.path.dirname(__file__), "..", "program.md")
    with open(path) as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--max-train-steps", type=int, default=500)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--conditions", nargs="+", default=["C1", "C2", "C3", "C4", "C5"])
    parser.add_argument("--use-kl-penalty", action="store_true")
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--wandb-project", default="cifar10-automl")
    args = parser.parse_args()

    program_md = load_program_md()
    parallel_eval = not args.no_parallel

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
    )

    all_histories = {}

    # C1: Random Search
    if "C1" in args.conditions:
        print("\n=== C1: Random Search ===")
        history = run_research_loop(
            condition_name="C1_random",
            G=1, use_history=False, generate_fn=None,
            **common
        )
        all_histories["C1_random"] = history

    # C2: LLM no-history G=1
    if "C2" in args.conditions:
        print("\n=== C2: LLM no-history G=1 ===")
        from src.agent import LLMAgent
        agent = LLMAgent(model_name=args.model)
        history = run_research_loop(
            condition_name="C2_llm_nohist_G1",
            G=1, use_history=False,
            generate_fn=agent.generate_specs,
            **common
        )
        all_histories["C2_llm_nohist_G1"] = history
        del agent

    # C3: LLM history G=1
    if "C3" in args.conditions:
        print("\n=== C3: LLM history G=1 ===")
        from src.agent import LLMAgent
        agent = LLMAgent(model_name=args.model)
        history = run_research_loop(
            condition_name="C3_llm_hist_G1",
            G=1, use_history=True,
            generate_fn=agent.generate_specs,
            **common
        )
        all_histories["C3_llm_hist_G1"] = history
        del agent

    # C4: LLM history G=4 (best-of-G, no weight update)
    if "C4" in args.conditions:
        print("\n=== C4: LLM history G=4 (best-of-G) ===")
        from src.agent import LLMAgent
        agent = LLMAgent(model_name=args.model)
        history = run_research_loop(
            condition_name="C4_llm_hist_G4",
            G=4, use_history=True,
            generate_fn=agent.generate_specs,
            grpo_update_fn=None,
            **common
        )
        all_histories["C4_llm_hist_G4"] = history
        del agent

    # C5: GRPO G=4
    if "C5" in args.conditions:
        print("\n=== C5: GRPO G=4 ===")
        import torch
        from src.grpo import setup_grpo_model, grpo_update
        from src.agent import extract_json
        from src.action_space import validate_spec

        model, ref_model, tokenizer, optimizer = setup_grpo_model(
            args.model,
            use_kl_penalty=args.use_kl_penalty,
        )

        def grpo_generate(prompt, n):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            results = []
            for _ in range(n):
                with torch.no_grad():
                    output = model.generate(
                        **inputs, max_new_tokens=512,
                        do_sample=True, temperature=0.8,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = output[0][inputs["input_ids"].shape[1]:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                raw = extract_json(text)
                spec = validate_spec(raw) if raw else None
                results.append(spec)
            return results

        def grpo_update_fn(prompt, candidates, accs):
            metrics = grpo_update(
                model, ref_model, tokenizer, optimizer,
                prompt, candidates, accs,
                kl_coef=args.kl_coef,
            )
            print(f"  [GRPO] loss={metrics['loss']:.4f}  "
                  f"pg={metrics['pg_loss']:.4f}  kl={metrics['kl_loss']:.4f}")
            return metrics["loss"]

        history = run_research_loop(
            condition_name="C5_grpo_G4",
            G=4, use_history=True,
            generate_fn=grpo_generate,
            grpo_update_fn=grpo_update_fn,
            **common
        )
        all_histories["C5_grpo_G4"] = history

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
