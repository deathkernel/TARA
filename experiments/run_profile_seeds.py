"""Run the trained Tiny-vs-Scaled comparison across independent seeds."""

import json
import statistics
import time

from experiments.compare_trained_profiles import compare, validate_result

SEEDS = (7, 17, 27)


def summarize(reports):
    summary = {}
    for profile in ("tiny", "scaled"):
        final_train_loss = [r[profile]["final_train"]["loss"] for r in reports]
        final_validation_loss = [r[profile]["final_validation"]["loss"] for r in reports]
        final_train_accuracy = [r[profile]["final_train"]["accuracy"] for r in reports]
        final_validation_accuracy = [r[profile]["final_validation"]["accuracy"] for r in reports]
        summary[profile] = {
            "parameters": reports[0][profile]["parameters"],
            "final_train_loss_mean": statistics.mean(final_train_loss),
            "final_train_loss_std": statistics.stdev(final_train_loss) if len(reports) > 1 else 0.0,
            "final_validation_loss_mean": statistics.mean(final_validation_loss),
            "final_validation_loss_std": statistics.stdev(final_validation_loss) if len(reports) > 1 else 0.0,
            "final_train_accuracy_mean": statistics.mean(final_train_accuracy),
            "final_train_accuracy_std": statistics.stdev(final_train_accuracy) if len(reports) > 1 else 0.0,
            "final_validation_accuracy_mean": statistics.mean(final_validation_accuracy),
            "final_validation_accuracy_std": statistics.stdev(final_validation_accuracy) if len(reports) > 1 else 0.0,
        }
    return summary


def main():
    reports = []
    started = time.perf_counter()
    for seed in SEEDS:
        report = compare(seed=seed)
        validate_result(report)
        reports.append(report)

    result = {
        "protocol": {
            "seeds": list(SEEDS),
            "profiles": ["tiny", "scaled"],
            "shared_budget": reports[0]["tiny"]["config"],
            "wall_clock_seconds": time.perf_counter() - started,
        },
        "summary": summarize(reports),
        "runs": reports,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
