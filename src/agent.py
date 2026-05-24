# src/agent.py
import json
import re
import torch
from typing import Optional, List
from src.action_space import ArchSpec, coerce_spec

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"


def extract_json(text: str) -> Optional[dict]:
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def format_prompt(tokenizer, prompt: str) -> str:
    """Apply Qwen3 chat template with thinking enabled."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )


def decode_thinking_and_answer(tokenizer, output_ids: list):
    """
    Official Qwen3 pattern: find last </think> token by ID, split there.
    Returns (thinking: str, answer: str).
    """
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    try:
        index = len(output_ids) - output_ids[::-1].index(think_end_id)
    except ValueError:
        index = 0   # model skipped thinking
    thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
    answer   = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    return thinking, answer


class LLMAgent:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        output_log: Optional[str] = None,
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_new_tokens = 2048
        self._log_fh = open(output_log, "w", encoding="utf-8") if output_log else None
        self._call_idx = 0

    def _log(self, entry: dict):
        if self._log_fh:
            self._log_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._log_fh.flush()

    def generate_specs(self, prompt: str, n: int = 1, temperature: float = 0.8) -> List[Optional[ArchSpec]]:
        formatted = format_prompt(self.tokenizer, prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Batch all n candidates into a single model.generate call
        inputs_batch = {k: v.repeat(n, 1) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs_batch,
                max_new_tokens=self.max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        results = []
        for i in range(n):
            output_ids = outputs[i][prompt_len:].tolist()
            thinking, answer = decode_thinking_and_answer(self.tokenizer, output_ids)
            raw = extract_json(answer)
            spec = coerce_spec(raw) if raw is not None else None
            self._log({
                "call": self._call_idx, "candidate": i,
                "thinking_chars": len(thinking),
                "thinking": thinking,
                "answer": answer,
                "extracted": raw,
                "valid": spec is not None,
            })
            self._call_idx += 1
            results.append(spec)
        return results
