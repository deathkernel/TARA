"""Runtime helpers for loading and generating text with a trained TARA LM.

Checkpoint loading is intentionally tensor-only. TARA never executes arbitrary
Python objects embedded in a checkpoint.
"""

from pathlib import Path

import torch

from src.tokenizer import CharTokenizer
from src.torch_language_model import FastTinyLanguageModel


def load_checkpoint(path):
    """Load a validated tensor/state-dict checkpoint without pickle execution."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a dictionary")

    required = {"tokenizer", "model_config", "model_state"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {sorted(missing)}")

    tokenizer_data = checkpoint["tokenizer"]
    if not isinstance(tokenizer_data, dict):
        raise ValueError("checkpoint tokenizer must be a dictionary")
    itos = tokenizer_data.get("itos")
    stoi = tokenizer_data.get("stoi")
    if not isinstance(itos, list) or not isinstance(stoi, dict):
        raise ValueError("checkpoint tokenizer is invalid")

    tokenizer = CharTokenizer("a")
    tokenizer.itos = list(itos)
    tokenizer.stoi = dict(stoi)
    config = checkpoint["model_config"]
    if not isinstance(config, dict):
        raise ValueError("checkpoint model_config must be a dictionary")
    model = FastTinyLanguageModel(**config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=256, temperature=0.8):
    """Generate a continuation from a prompt using temperature sampling."""
    if not prompt:
        raise ValueError("prompt must not be empty")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt produced no tokens")

    for _ in range(max_new_tokens):
        context_ids = ids[-model.max_context :]
        x = torch.tensor([context_ids], dtype=torch.long)
        logits = model(x)[0, -1] / temperature
        probabilities = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probabilities, 1).item()
        ids.append(next_id)

    return tokenizer.decode(ids)
