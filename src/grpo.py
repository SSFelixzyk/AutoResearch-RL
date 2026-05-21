# src/grpo.py
import json
import copy
import torch
import torch.nn.functional as F
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
    """Sum of log-probs over completion tokens only (not prompt tokens)."""
    full_text = prompt + completion
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

    logits = model(**inputs).logits  # (1, seq_len, vocab)
    log_probs = F.log_softmax(logits[0], dim=-1)
    target_ids = inputs["input_ids"][0]

    # Shift: position i predicts token i+1
    comp_lps = [
        log_probs[i, target_ids[i + 1]]
        for i in range(prompt_len, len(target_ids) - 1)
    ]
    if not comp_lps:
        return torch.tensor(0.0, device=model.device)
    return torch.stack(comp_lps).sum()


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
    One GRPO gradient step.
    Returns dict with 'loss', 'pg_loss', 'kl_loss'.
    """
    rewards = torch.tensor(accs, dtype=torch.float32)
    advantages = (rewards - rewards.mean()) / (rewards.std() + eps)

    completions = [
        json.dumps(c.to_dict()) if c is not None else "{}"
        for c in candidates
    ]

    pg_loss = torch.tensor(0.0, device=model.device)
    kl_loss = torch.tensor(0.0, device=model.device)
    n_valid = 0

    for completion, adv, acc in zip(completions, advantages, accs):
        if completion == "{}" and acc == 0.0:
            continue  # skip failed JSON parses

        with torch.enable_grad():
            log_prob = _completion_log_probs(model, tokenizer, prompt, completion)

        pg_loss = pg_loss - adv.to(model.device) * log_prob

        if ref_model is not None:
            with torch.no_grad():
                ref_lp = _completion_log_probs(ref_model, tokenizer, prompt, completion)
            kl_loss = kl_loss + (log_prob - ref_lp)

        n_valid += 1

    if n_valid == 0:
        return {"loss": 0.0, "pg_loss": 0.0, "kl_loss": 0.0}

    pg_loss = pg_loss / n_valid
    kl_loss = kl_loss / n_valid
    loss = pg_loss + kl_coef * kl_loss if ref_model is not None else pg_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0
    )
    optimizer.step()

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "kl_loss": kl_loss.item() if ref_model is not None else 0.0,
    }
