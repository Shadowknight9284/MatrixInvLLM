import argparse
import subprocess
import sys
from pathlib import Path

STEPS = [
    ("generate_matrices", "Generate raw matrices"),
    ("serialize_dataset", "Serialize dataset to prompts/targets"),
    ("build_splits", "Build train/val/test splits"),
    ("train_lora", "Train LoRA adapter"),
    ("run_experiment", "Run baseline/post-train evaluation"),
]
STEP_NAMES = [name for name, _ in STEPS]


def run_step(module: str, label: str) -> None:
    print("\n" + "=" * 60)
    print(f"RUNNING: {label}  [src.{module}]")
    print("=" * 60)
    cmd = [sys.executable, "-m", f"src.{module}"]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(
            f"\nPipeline stopped: src.{module} failed with exit code {result.returncode}."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the matrix inverse pipeline with optional step controls."
    )
    parser.add_argument(
        "--from-step",
        choices=STEP_NAMES,
        help="Start running at this step and continue to the end.",
    )
    parser.add_argument(
        "--to-step",
        choices=STEP_NAMES,
        help="Stop after this step.",
    )
    parser.add_argument(
        "--only-step",
        choices=STEP_NAMES,
        help="Run only one step.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip src.generate_matrices.",
    )
    parser.add_argument(
        "--skip-serialize",
        action="store_true",
        help="Skip src.serialize_dataset.",
    )
    parser.add_argument(
        "--skip-splits",
        action="store_true",
        help="Skip src.build_splits.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip src.train_lora.",
    )
    parser.add_argument(
        "--skip-experiment",
        action="store_true",
        help="Skip src.run_experiment.",
    )
    return parser.parse_args()


def selected_steps(args: argparse.Namespace):
    if args.only_step:
        return [step for step in STEPS if step[0] == args.only_step]

    steps = STEPS[:]

    if args.from_step:
        start_idx = STEP_NAMES.index(args.from_step)
        steps = steps[start_idx:]

    if args.to_step:
        end_idx = STEP_NAMES.index(args.to_step)
        steps = steps[: end_idx + 1]

    skip_map = {
        "generate_matrices": args.skip_generate,
        "serialize_dataset": args.skip_serialize,
        "build_splits": args.skip_splits,
        "train_lora": args.skip_train,
        "run_experiment": args.skip_experiment,
    }
    steps = [step for step in steps if not skip_map[step[0]]]

    if not steps:
        raise SystemExit("No pipeline steps selected. Adjust your flags and try again.")

    return steps


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    steps = selected_steps(args)

    print(f"Python: {sys.executable}")
    print(f"Repo root: {repo_root}")
    print("Selected steps:")
    for module, label in steps:
        print(f"- src.{module}: {label}")

    print("\nStarting pipeline...")
    for module, label in steps:
        run_step(module, label)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()