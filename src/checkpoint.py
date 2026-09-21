"""Small dependency-free checkpoints for TARA training.

Research basis:
- Training systems commonly persist model parameters together with optimizer
  and training-progress state when resuming is required.

TARA stores a JSON-compatible checkpoint containing scalar parameter values,
training step, scheduler configuration, and optional metrics. The format is
simple enough to inspect and restore on a normal PC.
"""

import json
from pathlib import Path


def save_checkpoint(model, path, step, scheduler=None, metrics=None):
    """Save model parameters and training metadata to a JSON checkpoint."""
    if step < 0:
        raise ValueError("step must be non-negative")
    parameters = [float(parameter.data) for parameter in model.parameters()]
    payload = {
        "format_version": 1,
        "step": int(step),
        "parameters": parameters,
        "metrics": dict(metrics or {}),
    }
    if scheduler is not None:
        payload["scheduler"] = {
            "initial_lr": scheduler.initial_lr,
            "total_steps": scheduler.total_steps,
            "min_lr": scheduler.min_lr,
        }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_checkpoint(model, path):
    """Load scalar model parameters and return training metadata."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    parameters = payload.get("parameters")
    model_parameters = model.parameters()
    if not isinstance(parameters, list) or len(parameters) != len(model_parameters):
        raise ValueError("checkpoint parameter count does not match model")
    for parameter, value in zip(model_parameters, parameters):
        parameter.data = float(value)
    return {
        "step": int(payload["step"]),
        "metrics": dict(payload.get("metrics", {})),
        "scheduler": payload.get("scheduler"),
    }
