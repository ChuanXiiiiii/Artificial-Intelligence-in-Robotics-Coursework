from pathlib import Path

import pandas as pd


def main() -> None:
    seeds = [42, 123, 2026]
    rows = []
    base = Path("outputs")
    for seed in seeds:
        before_log = (base / "logs" / f"hog_seed{seed}_before.log").read_text(
            encoding="utf-8", errors="ignore"
        )
        after_log = (base / "logs" / f"hog_seed{seed}_after.log").read_text(
            encoding="utf-8", errors="ignore"
        )
        before_warn = before_log.count("ConvergenceWarning")
        after_warn = after_log.count("ConvergenceWarning")

        before_metrics = pd.read_csv(
            base / "tables" / f"metrics_hog_svm_seed{seed}_test_before_tuning.csv"
        ).iloc[0]
        after_metrics = pd.read_csv(
            base / "tables" / f"metrics_hog_svm_seed{seed}_test_after_tuning.csv"
        ).iloc[0]

        rows.append(
            {
                "seed": seed,
                "before_warning_count": before_warn,
                "after_warning_count": after_warn,
                "before_accuracy": float(before_metrics["accuracy"]),
                "after_accuracy": float(after_metrics["accuracy"]),
                "accuracy_delta": float(after_metrics["accuracy"] - before_metrics["accuracy"]),
                "before_macro_f1": float(before_metrics["macro_f1"]),
                "after_macro_f1": float(after_metrics["macro_f1"]),
                "macro_f1_delta": float(after_metrics["macro_f1"] - before_metrics["macro_f1"]),
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(base / "tables" / "hog_convergence_before_after.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "stage": "before",
                "warning_count_total": int(comparison["before_warning_count"].sum()),
                "warning_count_mean": float(comparison["before_warning_count"].mean()),
                "accuracy_mean": float(comparison["before_accuracy"].mean()),
                "macro_f1_mean": float(comparison["before_macro_f1"].mean()),
            },
            {
                "stage": "after",
                "warning_count_total": int(comparison["after_warning_count"].sum()),
                "warning_count_mean": float(comparison["after_warning_count"].mean()),
                "accuracy_mean": float(comparison["after_accuracy"].mean()),
                "macro_f1_mean": float(comparison["after_macro_f1"].mean()),
            },
        ]
    )
    summary.to_csv(base / "tables" / "hog_convergence_before_after_summary.csv", index=False)

    print(comparison.to_string(index=False))
    print("\nSummary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
