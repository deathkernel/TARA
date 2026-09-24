"""Train TARA Baby through a staged science-and-conversation curriculum.

Design:
- one tokenizer and one neural model across stages;
- stage-wise data ordering;
- replay from previous stages to reduce forgetting;
- validation after every stage, including previous stages;
- stage checkpoints stored locally, never committed to Git.

This follows current continual-learning and curriculum-learning practice while
keeping the implementation small enough for TARA's local research setup.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from src.tara_mind.core.tokenizer import FastBPETokenizer
from src.tara_mind.core.transformer import FastTinyLanguageModel
from src.tara_mind.data.registry import load_dataset_text


STAGES = {
    "A_social": ["soda", "empathetic_dialogues", "daily_dialog", "blended_skill_talk"],
    "B_math": ["gsm8k", "competition_math"],
    "C_physics": ["physics_eval"],
    "D_science": ["sciq", "ai2_arc"],
}

DEFAULT_ORDER = tuple(STAGES)


def windows(ids: list[int], context: int) -> list[tuple[list[int], list[int]]]:
    if len(ids) <= context:
        raise ValueError("tokenized data is too short for context")
    return [(ids[i : i + context], ids[i + 1 : i + context + 1]) for i in range(len(ids) - context)]


def batch(
    data: list[tuple[list[int], list[int]]],
    size: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not data:
        raise ValueError("cannot sample an empty dataset")
    chosen = [data[rng.randrange(len(data))] for _ in range(size)]
    x = torch.tensor([item[0] for item in chosen], dtype=torch.long, device=device)
    y = torch.tensor([item[1] for item in chosen], dtype=torch.long, device=device)
    return x, y


def evaluate(
    model: torch.nn.Module,
    data: list[tuple[list[int], list[int]]],
    batch_size: int,
    rng: random.Random,
    device: torch.device,
    batches: int = 8,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(max(1, batches)):
            x, y = batch(data, batch_size, rng, device)
            losses.append(float(model.loss(x, y).item()))
    return sum(losses) / len(losses)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model_config: dict[str, object],
    tokenizer: FastBPETokenizer,
    stage: str,
    step: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 9,
            "model_state": model.state_dict(),
            "model_config": model_config,
            "tokenizer": {
                "type": "fast_bpe",
                "json": tokenizer.to_json(),
                "vocab_size": tokenizer.vocab_size,
            },
            "stage": stage,
            "step": step,
            "metrics": metrics,
        },
        path,
    )


def load_stage_text(
    datasets: list[str],
    train_chars: int,
    validation_chars: int,
    seed: int,
) -> tuple[str, str]:
    train_parts = []
    validation_parts = []
    per_dataset_train = max(1, train_chars // len(datasets))
    per_dataset_validation = max(1, validation_chars // len(datasets))
    for index, dataset in enumerate(datasets):
        train_parts.append(
            load_dataset_text(dataset, per_dataset_train, "train", seed + index)
        )
        validation_parts.append(
            load_dataset_text(
                dataset,
                per_dataset_validation,
                "validation" if dataset not in {"gsm8k"} else "test",
                seed + index + 1000,
            )
        )
    return "\n\n".join(train_parts), "\n\n".join(validation_parts)


def parse_order(raw: str) -> list[str]:
    order = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in order if item not in STAGES]
    if unknown:
        raise ValueError(f"unknown stages: {', '.join(unknown)}")
    if not order:
        raise ValueError("at least one stage is required")
    return order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default=",".join(DEFAULT_ORDER))
    parser.add_argument("--steps-per-stage", type=int, default=200)
    parser.add_argument("--chars-per-dataset", type=int, default=65536)
    parser.add_argument("--validation-chars-per-dataset", type=int, default=16384)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--embedding", type=int, default=256)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--replay-ratio", type=float, default=0.20)
    parser.add_argument("--replay-capacity", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints/curriculum")
    parser.add_argument("--report", default="outputs/curriculum_report.json")
    args = parser.parse_args()

    if args.steps_per_stage < 1:
        raise ValueError("steps-per-stage must be >= 1")
    if not 0.0 <= args.replay_ratio <= 1.0:
        raise ValueError("replay-ratio must be in [0, 1]")
    if args.replay_capacity < 1:
        raise ValueError("replay-capacity must be >= 1")

    order = parse_order(args.stages)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"device={device}")
    print(f"stages={order}")
    print("[1/6] Loading curriculum text...")
    stage_text: dict[str, tuple[str, str]] = {}
    all_text = []
    for index, stage in enumerate(order):
        train_text, val_text = load_stage_text(
            STAGES[stage],
            args.chars_per_dataset,
            args.validation_chars_per_dataset,
            args.seed + index * 100,
        )
        stage_text[stage] = (train_text, val_text)
        all_text.extend([train_text, val_text])
        print(
            f"  {stage}: datasets={STAGES[stage]} "
            f"train_chars={len(train_text):,} val_chars={len(val_text):,}",
            flush=True,
        )

    print("[2/6] Training shared Rust-backed BPE tokenizer...")
    tokenizer = FastBPETokenizer("\n\n".join(all_text), vocab_size=args.vocab_size)
    stage_windows: dict[str, list[tuple[list[int], list[int]]]] = {}
    stage_validation: dict[str, list[tuple[list[int], list[int]]]] = {}
    for stage in order:
        train_ids = tokenizer.encode(stage_text[stage][0])
        val_ids = tokenizer.encode(stage_text[stage][1])
        stage_windows[stage] = windows(train_ids, args.context)
        stage_validation[stage] = windows(val_ids, args.context)

    model = FastTinyLanguageModel(
        tokenizer.vocab_size,
        args.embedding,
        args.ff_dim,
        args.heads,
        args.context,
        args.layers,
        0.1,
        True,
        args.seed,
    ).to(device)
    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "embedding_dim": args.embedding,
        "ff_dim": args.ff_dim,
        "num_heads": args.heads,
        "max_context": args.context,
        "num_layers": args.layers,
        "dropout": 0.1,
        "tie_embeddings": True,
    }
    print(f"[3/6] model_params={parameter_count(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    replay_pool: list[tuple[list[int], list[int]]] = []
    report: dict[str, object] = {"order": order, "device": str(device), "stages": []}

    print("[4/6] Sequential curriculum + replay training...")
    for stage_index, stage in enumerate(order, start=1):
        current = stage_windows[stage]
        replay_count = int(args.batch_size * args.replay_ratio)
        current_count = args.batch_size - replay_count
        if current_count < 1:
            current_count = 1
            replay_count = args.batch_size - 1

        metrics: dict[str, float] = {}
        before = evaluate(model, stage_validation[stage], args.batch_size, rng, device)
        metrics["val_before"] = before
        started = time.perf_counter()

        print(f"\n=== {stage_index}/{len(order)} {stage} ===", flush=True)
        for step in range(1, args.steps_per_stage + 1):
            model.train()
            current_batch = batch(current, current_count, rng, device)
            if replay_pool and replay_count:
                replay_batch = batch(replay_pool, replay_count, rng, device)
                x = torch.cat((current_batch[0], replay_batch[0]), dim=0)
                y = torch.cat((current_batch[1], replay_batch[1]), dim=0)
            else:
                x, y = current_batch

            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step == 1 or step % 25 == 0 or step == args.steps_per_stage:
                print(
                    f"stage={stage} step={step}/{args.steps_per_stage} "
                    f"train_loss={loss.item():.4f}",
                    flush=True,
                )

        metrics["val_after"] = evaluate(model, stage_validation[stage], args.batch_size, rng, device)
        for previous in order[: stage_index - 1]:
            metrics[f"forgetting_check_{previous}"] = evaluate(
                model,
                stage_validation[previous],
                args.batch_size,
                rng,
                device,
            )

        # Preserve a bounded replay pool of the current stage for future stages.
        if len(replay_pool) + len(current) <= args.replay_capacity:
            replay_pool.extend(current)
        else:
            replay_pool.extend(current[: max(1, args.replay_capacity - len(replay_pool))])
            rng.shuffle(replay_pool)
            del replay_pool[args.replay_capacity :]

        metrics["seconds"] = time.perf_counter() - started
        checkpoint = Path(args.checkpoint_dir) / f"tara_baby_{stage}.pt"
        save_checkpoint(
            checkpoint,
            model,
            model_config,
            tokenizer,
            stage,
            args.steps_per_stage,
            metrics,
        )
        report["stages"].append({"stage": stage, "metrics": metrics, "checkpoint": str(checkpoint)})
        print(
            f"checkpoint={checkpoint} val_before={before:.4f} "
            f"val_after={metrics['val_after']:.4f}",
            flush=True,
        )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[6/6] report={report_path}")


if __name__ == "__main__":
    main()
