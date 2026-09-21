"""Runtime helpers for loading and generating text with a trained TARA LM."""

from pathlib import Path

import torch

from src.tokenizer import CharTokenizer
from src.torch_language_model import FastTinyLanguageModel


def load_checkpoint(path):
    """Load a saved algorithm-language-model checkpoint."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    tokenizer_data = checkpoint["tokenizer"]
    tokenizer = CharTokenizer("a")
    tokenizer.itos = tokenizer_data["itos"]
    tokenizer.stoi = tokenizer_data["stoi"]
    config = checkpoint["model_config"]
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
