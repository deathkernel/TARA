"""Interactive chat with a trained TARA Baby checkpoint."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from src.tara_mind.core.transformer import FastTinyLanguageModel


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = FastTinyLanguageModel(
        config["vocab_size"],
        config["embedding_dim"],
        config["ff_dim"],
        config["num_heads"],
        config["max_context"],
        config["num_layers"],
        config.get("dropout", 0.0),
        config.get("tie_embeddings", False),
        0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tokenizer_info = checkpoint.get("tokenizer")
    if not tokenizer_info or tokenizer_info.get("type") != "fast_bpe":
        raise ValueError("checkpoint does not contain a supported TARA BPE tokenizer")
    tokenizer = Tokenizer.from_str(tokenizer_info["json"])
    # Older checkpoints may have been serialized before the ByteLevel decoder was added.
    tokenizer.decoder = ByteLevelDecoder()
    return model, tokenizer, checkpoint


def sample_token(
    logits: torch.Tensor,
    rng: random.Random,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")

    values = (logits / temperature).float()
    probabilities = torch.softmax(values, dim=-1)

    k = min(top_k, probabilities.numel())
    top_values, top_indices = torch.topk(probabilities, k=k)
    if top_p < 1.0:
        cumulative = torch.cumsum(top_values, dim=-1)
        keep = cumulative <= top_p
        keep[0] = True
        top_values = top_values[keep]
        top_indices = top_indices[keep]

    top_values = top_values / top_values.sum()
    threshold = rng.random()
    cumulative = 0.0
    for probability, index in zip(top_values.tolist(), top_indices.tolist()):
        cumulative += probability
        if threshold < cumulative:
            return int(index)
    return int(top_indices[-1].item())


def generate_reply(
    model: FastTinyLanguageModel,
    tokenizer: Tokenizer,
    history: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    rng: random.Random,
) -> str:
    ids = tokenizer.encode(history).ids
    if not ids:
        raise ValueError("prompt produced no tokens")

    context_length = model.max_context
    original_len = len(ids)
    for _ in range(max_new_tokens):
        context_ids = ids[-context_length:]
        x = torch.tensor([context_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)[0, -1]
        next_id = sample_token(logits, rng, temperature, top_k, top_p)
        ids.append(next_id)

    generated = tokenizer.decode(ids[original_len:])
    return generated.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with TARA Baby")
    parser.add_argument("--checkpoint", default="checkpoints/tara_vnext_soda.pt")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be >= 1")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device)
    rng = random.Random(args.seed)

    print("TARA Baby chat")
    print(f"checkpoint: {args.checkpoint}")
    print(f"trained_step: {checkpoint.get('step', 'unknown')}")
    print(f"device: {device}")
    print("Type /exit to quit, /clear to clear conversation.\n")

    history = ""
    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            print("Bye!")
            break
        if user_text.lower() == "/clear":
            history = ""
            print("[conversation cleared]")
            continue

        history += f"User: {user_text}\nTARA:"
        reply = generate_reply(
            model,
            tokenizer,
            history,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            rng,
        )
        print(f"TARA: {reply}\n")
        history += f" {reply}\n"

        # Keep the prompt bounded as the conversation grows.
        encoded_history = tokenizer.encode(history).ids
        if len(encoded_history) > model.max_context * 2:
            history = tokenizer.decode(encoded_history[-model.max_context:])


if __name__ == "__main__":
    main()
