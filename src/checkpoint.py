"""Small dependency-free checkpoints for TARA training.

Research basis:
- Training systems commonly persist model parameters together with optimizer
  and training-progress state when resuming is required.

TARA stores JSON-compatible scalar model values and training state. Version 2
also persists optimizer state when supplied, allowing AdamW moment estimates
to survive an exact training resume. Version 1 checkpoints remain readable.
"""

import json
import math
from pathlib import Path


def save_checkpoint(model, path, step, scheduler=None, metrics=None, optimizer=None):
    """Save model parameters and optional optimizer/scheduler metadata."""
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("step must be a non-negative integer")
    parameters = [float(parameter.data) for parameter in model.parameters()]
    if any(not math.isfinite(value) for value in parameters):
        raise ValueError("cannot checkpoint non-finite model parameters")
    payload = {
        "format_version": 2 if optimizer is not None else 1,
        "step": step,
        "parameters": parameters,
        "metrics": dict(metrics or {}),
    }
    if scheduler is not None:
        payload["scheduler"] = {
            "initial_lr": scheduler.initial_lr,
            "total_steps": scheduler.total_steps,
            "min_lr": scheduler.min_lr,
        }
    if optimizer is not None:
        if not hasattr(optimizer, "state_dict"):
            raise TypeError("optimizer must provide state_dict()")
        payload["optimizer"] = optimizer.state_dict()
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_checkpoint(model, path, optimizer=None):
    """Load model parameters and optional optimizer state."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("format_version")
    if version not in (1, 2):
        raise ValueError("unsupported checkpoint format")
    parameters = payload.get("parameters")
    model_parameters = model.parameters()
    if not isinstance(parameters, list) or len(parameters) != len(model_parameters):
        raise ValueError("checkpoint parameter count does not match model")
    if any(not math.isfinite(float(value)) for value in parameters):
        raise ValueError("checkpoint contains non-finite parameters")
    step = payload.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")

    optimizer_state = payload.get("optimizer")
    if optimizer is not None:
        if optimizer_state is None:
            raise ValueError("checkpoint does not contain optimizer state")
        # Validate the complete optimizer state before mutating model values.
        optimizer.load_state_dict(optimizer_state)

    for parameter, value in zip(model_parameters, parameters):
        parameter.data = float(value)

    return {
        "step": step,
        "metrics": dict(payload.get("metrics", {})),
        "scheduler": payload.get("scheduler"),
        "optimizer": optimizer_state,
    }
