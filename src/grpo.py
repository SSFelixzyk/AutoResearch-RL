# src/grpo.py
import json
import copy
import torch
from typing import List, Optional
from src.action_space import ArchSpec


def setup_grpo_model(
    model_name: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
    use_kl_penalty: bool = False,
) -> tuple:
    """
    Returns (model, ref_model, tokenizer, optimizer).
    ref_model is a frozen copy for KL penalty; None if use_kl_penalty=False.
    Imports transformers and peft lazily to avoid import errors when not used.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig, TaskType

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    ref_model = None
    if use_kl_penalty:
        ref_model = copy.deepcopy(base_model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5, weight_decay=0.01,
    )
    return model, ref_model, tokenizer, optimizer


def _completion_log_probs(model, tokenizer, prompt: str, completion: str) -> torch.Tensor:
    """
    Sum of log-probs over completion tokens only.
    Uses labels=-100 on prompt tokens so the model computes cross-entropy
    internally — avoids materialising the full (seq_len, vocab) log_probs tensor.
    """
    full_text = prompt + completion
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

    labels = inputs["input_ids"].clone()
    labels[0, :prompt_len] = -100   # ignore prompt tokens

    n_comp = int((labels[0] != -100).sum())
    if n_comp == 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    outputs = model(**inputs, labels=labels, use_cache=False)
    # outputs.loss = mean NLL over completion tokens → convert to sum of log-probs
    return -outputs.loss * n_comp


def grpo_update(
    model,
    ref_model,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    prompt: str,
    candidates: List[Optional[ArchSpec]],
    accs: List[float],
    kl_coef: float = 0.1,
    eps: float = 1e-8,
) -> dict:
    """
    One GRPO gradient step using per-candidate backward to save GPU memory.
    Each candidate's loss is backpropagated immediately so only one computation
    graph is live at a time (instead of accumulating all G graphs then backward).
    Returns dict with 'loss', 'pg_loss', 'kl_loss'.
    """
    rewards = torch.tensor(accs, dtype=torch.float32)
    advantages = (rewards - rewards.mean()) / (rewards.std() + eps)

    completions = [
        json.dumps(c.to_dict()) if c is not None else "{}"
        for c in candidates
    ]

    n_valid = sum(
        1 for comp, acc in zip(completions, accs)
        if not (comp == "{}" and acc == 0.0)
    )
    if n_valid == 0:
        return {"loss": 0.0, "pg_loss": 0.0, "kl_loss": 0.0}

    optimizer.zero_grad()
    total_pg = 0.0
    total_kl = 0.0

    for completion, adv, acc in zip(completions, advantages, accs):
        if completion == "{}" and acc == 0.0:
            continue

        log_prob = _completion_log_probs(model, tokenizer, prompt, completion)
        pg_term = -adv.to(model.device) * log_prob / n_valid

        if ref_model is not None:
            with torch.no_grad():
                ref_lp = _completion_log_probs(ref_model, tokenizer, prompt, completion)
            kl_term = (log_prob.detach() - ref_lp) / n_valid
            loss = pg_term + kl_coef * kl_term
            total_kl += kl_term.item()
        else:
            loss = pg_term

        # Backward immediately — frees this candidate's computation graph
        loss.backward()
        torch.cuda.empty_cache()
        total_pg += pg_term.item()

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0
    )
    optimizer.step()

    total_loss = total_pg + kl_coef * total_kl
    return {
        "loss": total_loss,
        "pg_loss": total_pg,
        "kl_loss": total_kl,
    }
