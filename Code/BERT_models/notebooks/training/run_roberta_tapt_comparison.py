"""Run a matched standard-vs-TAPT RoBERTa-large comparison for one seed.

The training and evaluation implementation is loaded from
``BERT_Soft_Label_LR_Epoch_Tuning.ipynb`` so this runner stays aligned with
the project's canonical classifier code.  It changes only the selected seed
and the starting encoder checkpoint.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def find_project_dir() -> Path:
    """Find the repository root from the current working directory."""
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "data" / "annotate-lime.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the project root containing data/annotate-lime.csv. "
        "Start this command from inside the repository."
    )


def load_classifier_definitions(project_dir: Path) -> Path:
    """Execute only the setup/definition cells from the canonical notebook."""
    source_notebook = (
        project_dir
        / "Code"
        / "BERT_models"
        / "notebooks"
        / "training"
        / "BERT_Soft_Label_LR_Epoch_Tuning.ipynb"
    )
    notebook = json.loads(source_notebook.read_text())
    loaded_cells = 0

    for cell_index, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.lstrip().startswith("SMOKE_TEST = False"):
            break
        exec(
            compile(source, f"{source_notebook.name}:cell-{cell_index}", "exec"),
            globals(),
        )
        loaded_cells += 1

    print(
        f"Loaded {loaded_cells} setup/definition cells from "
        f"{source_notebook.name}."
    )
    return source_notebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate matched standard and TAPT RoBERTa-large "
            "classifiers for one random seed."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed whose TAPT checkpoint and data split should be used.",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Retrain runs whose completed result files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = find_project_dir()
    load_classifier_definitions(project_dir)

    comparison_output_dir = (
        project_dir
        / "Code"
        / "BERT_models"
        / "experiments"
        / "tapt"
        / "downstream_comparison"
    )
    tapt_checkpoint_dir = (
        project_dir
        / "Code"
        / "BERT_models"
        / "experiments"
        / "tapt"
        / "masked_language_modeling"
        / f"roberta_large_tapt_random_seed{args.seed}"
    )

    baseline_config = with_tuning_settings(
        ROBERTA_LARGE_CONFIG,
        learning_rate=1e-5,
        num_epochs=30,
        patience=6,
        scheduler_type="linear",
    )
    tapt_config = with_tuning_settings(
        make_roberta_large_tapt_config(tapt_checkpoint_dir),
        learning_rate=1e-5,
        num_epochs=30,
        patience=6,
        scheduler_type="linear",
    )
    comparison_configs = [baseline_config, tapt_config]
    comparison_output_dir.mkdir(parents=True, exist_ok=True)

    comparison_train_df, _, _ = make_random_split(
        dataset,
        args.seed,
        split_scheme="majority",
    )
    run_id = f"random_seed{args.seed}"
    validate_tapt_checkpoint_for_run(tapt_config, run_id, comparison_train_df)

    print(f"Comparison seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Output directory: {comparison_output_dir}")
    print(f"Validated TAPT checkpoint: {tapt_checkpoint_dir}")
    print(pd.DataFrame([asdict(config) for config in comparison_configs]))

    comparison_stem = f"roberta_large_tapt_comparison_seed{args.seed}"
    comparison_summary = run_random_split_suite(
        model_configs=comparison_configs,
        seeds=[args.seed],
        loss_mode="binomial_nll",
        output_dir=comparison_output_dir,
        summary_name=comparison_stem,
        skip_completed=not args.rerun_completed,
    )

    comparison_columns = [
        "experiment_name",
        "model_name",
        "run_id",
        "pooling",
        "best_epoch",
        "best_val_loss",
        "macro_mae",
        "mean_pearson",
        "majority_tuned_macro_f1",
    ]
    comparison_table = comparison_summary[comparison_columns].sort_values(
        "experiment_name"
    )
    key_metrics_path = comparison_output_dir / f"{comparison_stem}_key_metrics.csv"
    comparison_table.to_csv(key_metrics_path, index=False)

    print("\nCompleted comparison:")
    print(comparison_table.to_string(index=False))
    print(f"\nKey metrics: {key_metrics_path}")


if __name__ == "__main__":
    main()
