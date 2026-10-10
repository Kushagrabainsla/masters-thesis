"""Model execution helpers: decode-mode and prefill-mode logits, free running and teacher forced.

Tie breaking: argmax is always taken on CPU float32 logits so free-running generation and
the teacher-forced flip analysis agree on ties.
"""
import torch


def load(model_id: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    return model.to(device).eval()


@torch.no_grad()
def decode_run(model, device: str, prompt_ids, steps: int, forced=None):
    """Prefill the prompt, then `steps` single-token decode steps with a KV cache.

    Returns (tokens, logits) where logits[t] is the float32 CPU logit vector that predicts
    token t. If `forced` is None the model free runs (greedy). Otherwise token t is forced to
    forced[t] after logits[t] is recorded, so logits[t] is conditioned on forced[:t].
    """
    ids = torch.tensor([list(prompt_ids)], device=device)
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    logits_t = out.logits[0, -1]
    tokens, rows = [], []
    for t in range(steps):
        row = logits_t.float().cpu()
        rows.append(row)
        tok = int(row.argmax().item()) if forced is None else int(forced[t])
        tokens.append(tok)
        if t == steps - 1:
            break
        out = model(input_ids=torch.tensor([[tok]], device=device), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits_t = out.logits[0, -1]
    return tokens, torch.stack(rows)


@torch.no_grad()
def prefill_forced_logits(model, device: str, prompt_ids, forced_tokens):
    """One parallel forward over prompt + forced[:-1]. Returns logits predicting each forced token."""
    steps = len(forced_tokens)
    seq = list(prompt_ids) + list(forced_tokens[:-1])
    out = model(input_ids=torch.tensor([seq], device=device), use_cache=False)
    p = len(prompt_ids)
    return out.logits[0, p - 1:p - 1 + steps].float().cpu()
