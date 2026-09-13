"""Run all four baseline scripts back-to-back (tune, then full 5-fold each).

Convenience wrapper only -- each script is fully usable standalone (see docs/baselines.md §7).
Intended for an unattended overnight pass once each model has been smoke-tested individually with
`--sample-rows`.

Usage:
    python run_all_baselines.py
    python run_all_baselines.py --skip tabnet   # e.g. if pytorch-tabnet isn't installed yet
"""

from __future__ import annotations

import argparse
import subprocess
import sys

RUNS = [
    ("mlp", ["baseline_mlp.py", "--tune", "--n-trials", "20", "--save-explanations"]),
    ("ft_transformer", ["baseline_ft_transformer.py", "--tune", "--n-trials", "12", "--save-explanations"]),
    ("tabnet", ["baseline_tabnet.py", "--tune", "--n-trials", "12", "--save-explanations"]),
    ("nam", ["baseline_nam.py", "--tune", "--n-trials", "25", "--save-explanations"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip", nargs="*", default=[], help="model names to skip, e.g. tabnet")
    args = parser.parse_args()

    for name, cmd in RUNS:
        if name in args.skip:
            print(f"== skipping {name} ==")
            continue
        print(f"\n== running {name}: {' '.join(cmd)} ==")
        result = subprocess.run([sys.executable, *cmd])
        if result.returncode != 0:
            print(f"!! {name} exited with code {result.returncode} -- stopping. "
                  f"Fix and re-run with --skip for the models already done.")
            sys.exit(result.returncode)

    print("\nAll baselines complete. See results/baseline_results.csv")


if __name__ == "__main__":
    main()
