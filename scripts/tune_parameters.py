#!/usr/bin/env python3
"""Sweep the `cli` binary's -b / --beta-scale / --eps / --max-diff parameters
against a fixed --file input and plot the resulting space overhead and
construction cost (hash_evaluations).

By default, for each parameter the other three are held at their baseline
(median) value while it's varied over the given list -- a one-at-a-time
sensitivity sweep, not a full grid search, so runtime stays linear in the
number of values tried instead of exploding combinatorially.

Pass --grid to instead run the full Cartesian product of all four
parameters' values. This is combinatorial (product of the list lengths), so
keep the lists short; it plots every combination's space/time tradeoff
instead of one line per parameter.

Usage:
    python3 scripts/tune_parameters.py --file path/to/kv.txt \
        --b 2 4 8 16 32 --beta-scale 0.5 1 2 --eps 0.05 0.1 0.2 \
        --max-diff 2 8 16 20

    python3 scripts/tune_parameters.py --file path/to/kv.txt --grid \
        --b 8 32 --beta-scale 0.5 1 --eps 0.01 0.1 --max-diff 8 16

Rebuild the binary in release mode first for realistic timings:
    cargo build --release --bin cli
"""

import argparse
import itertools
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULT_PATTERNS = {
    "space_bytes": re.compile(r"space:\s*(\d+)"),
    "space_overhead": re.compile(r"space overhead:\s*([-\d.eE]+)"),
    "hash_evaluations": re.compile(r"time:\s*(\d+)"),
}


def run_once(
    binary: Path,
    file: Path,
    b: int,
    beta_scale: float,
    eps: float,
    max_diff: float,
    timeout: float,
):
    cmd = [
        str(binary),
        "--file",
        str(file),
        "-b",
        str(b),
        "--beta-scale",
        str(beta_scale),
        "--eps",
        str(eps),
        "--max-diff",
        str(max_diff),
    ]
    label = f"b={b} beta_scale={beta_scale} eps={eps} max_diff={max_diff}"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  timed out after {timeout}s: {label}", file=sys.stderr)
        return "timeout"

    if proc.returncode != 0:
        print(
            f"  failed (exit {proc.returncode}): {label}\n"
            f"  stderr: {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else '(empty)'}",
            file=sys.stderr,
        )
        return None

    result = {}
    for key, pattern in RESULT_PATTERNS.items():
        match = pattern.search(proc.stdout)
        if not match:
            print(f"  couldn't find {key!r} in output for {label}", file=sys.stderr)
            return None
        result[key] = float(match.group(1))
    return result


def sweep(
    binary: Path, file: Path, param: str, values, baseline: dict, timeout: float
) -> tuple[pd.DataFrame, list]:
    rows = []
    timed_out = []
    for value in values:
        kwargs = {**baseline, param: value}
        print(
            f"running {param}={value} (b={kwargs['b']}, beta_scale={kwargs['beta_scale']}, "
            f"eps={kwargs['eps']}, max_diff={kwargs['max_diff']})"
        )
        result = run_once(
            binary,
            file,
            kwargs["b"],
            kwargs["beta_scale"],
            kwargs["eps"],
            kwargs["max_diff"],
            timeout,
        )
        if result == "timeout":
            timed_out.append(value)
        elif result is not None:
            rows.append({param: value, **result})
    return pd.DataFrame(rows), timed_out


def grid_sweep(
    binary: Path,
    file: Path,
    b_values,
    beta_scale_values,
    eps_values,
    max_diff_values,
    timeout: float,
) -> tuple[pd.DataFrame, list]:
    """Full Cartesian product over all four parameters, unlike `sweep`'s
    one-at-a-time. Cost is the product of the four list lengths, so this can
    get expensive fast -- keep the lists short when using --grid."""
    combos = list(
        itertools.product(b_values, beta_scale_values, eps_values, max_diff_values)
    )
    worst_case_minutes = len(combos) * timeout / 60
    print(
        f"grid sweep: {len(combos)} combinations, up to {worst_case_minutes:.1f} "
        "min worst case (if every run hit the timeout)"
    )

    rows = []
    timed_out = []
    for i, (b, beta_scale, eps, max_diff) in enumerate(combos):
        print(
            f"[{i + 1}/{len(combos)}] running b={b} beta_scale={beta_scale} "
            f"eps={eps} max_diff={max_diff}"
        )
        result = run_once(binary, file, b, beta_scale, eps, max_diff, timeout)
        params = {"b": b, "beta_scale": beta_scale, "eps": eps, "max_diff": max_diff}
        if result == "timeout":
            timed_out.append(params)
        elif result is not None:
            rows.append({**params, **result})
    return pd.DataFrame(rows), timed_out


def run_grid(args):
    df, timed_out = grid_sweep(
        args.bin, args.file, args.b, args.beta_scale, args.eps, args.max_diff, args.timeout
    )
    df.to_csv(args.csv_out, index=False)
    print(f"wrote {args.csv_out}")

    total = len(df) + len(timed_out)
    print(f"{len(df)}/{total} runs completed, {len(timed_out)} timed out")

    if df.empty:
        print("no successful runs; nothing to plot", file=sys.stderr)
        return

    best = df.loc[df["space_overhead"].idxmin()]
    print(
        f"best space overhead: {best['space_overhead']:.4f} bits/key at "
        f"b={best['b']:.0f} beta_scale={best['beta_scale']} eps={best['eps']} "
        f"max_diff={best['max_diff']}"
    )

    # The grid varies four parameters at once, so a per-parameter line plot
    # (like the one-at-a-time sweep uses) doesn't make sense here. Instead,
    # plot every combination's actual space/time tradeoff -- the thing you
    # actually want to pick a point from -- colored by b, with the best
    # point picked out.
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        df["hash_evaluations"], df["space_overhead"], c=df["b"], cmap="viridis", alpha=0.8
    )
    ax.scatter(
        [best["hash_evaluations"]],
        [best["space_overhead"]],
        color="red",
        marker="*",
        s=250,
        zorder=5,
        label=f"best overhead={best['space_overhead']:.3f}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("hash evaluations")
    ax.set_ylabel("space overhead (bits/key)")
    ax.set_title(
        f"Grid sweep against {args.file}\n{len(df)}/{total} runs completed, "
        f"{len(timed_out)} timed out"
    )
    fig.colorbar(scatter, label="b")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.plot_out, dpi=150)
    print(f"wrote {args.plot_out}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="key/value input file (as read by --file)",
    )
    parser.add_argument(
        "--bin",
        type=Path,
        default=Path("target/release/cli"),
        help="path to the built cli binary",
    )
    parser.add_argument("--b", type=int, nargs="+", default=[2, 8, 32, 64, 128])
    parser.add_argument(
        "--beta-scale",
        dest="beta_scale",
        type=float,
        nargs="+",
        default=[0.001, 0.1, 0.5, 1.0, 2, 4.0, 8, 16],
    )
    parser.add_argument("--eps", type=float, nargs="+", default=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, .5])
    parser.add_argument(
        "--max-diff",
        dest="max_diff",
        type=float,
        nargs="+",
        default=[2, 4, 8, 12, 16, 20, 25, 32],
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="per-run timeout in seconds"
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help=(
            "run a full grid (Cartesian product) sweep over all four "
            "parameters instead of one-at-a-time; cost is the product of "
            "the list lengths, so use short lists"
        ),
    )
    parser.add_argument("--csv-out", type=Path, default=Path("tuning_results.csv"))
    parser.add_argument("--plot-out", type=Path, default=Path("tuning_results.png"))
    args = parser.parse_args()

    if not args.bin.exists():
        sys.exit(
            f"binary not found: {args.bin} (did you `cargo build --release --bin cli`?)"
        )

    if not args.file.exists():
        sys.exit(f"file not found: {args.file}")

    args.b.sort()
    args.beta_scale.sort()
    args.eps.sort()
    args.max_diff.sort()

    if args.grid:
        run_grid(args)
        return

    baseline = {
        "b": args.b[len(args.b) // 2],
        "beta_scale": args.beta_scale[len(args.beta_scale) // 2],
        "eps": args.eps[len(args.eps) // 2],
        "max_diff": args.max_diff[len(args.max_diff) // 2],
    }
    sweeps = {
        "b": sweep(args.bin, args.file, "b", args.b, baseline, args.timeout),
        "beta_scale": sweep(
            args.bin, args.file, "beta_scale", args.beta_scale, baseline, args.timeout
        ),
        "eps": sweep(args.bin, args.file, "eps", args.eps, baseline, args.timeout),
        "max_diff": sweep(
            args.bin, args.file, "max_diff", args.max_diff, baseline, args.timeout
        ),
    }

    combined = pd.concat(
        [
            df.assign(swept_param=name)
            for name, (df, _timed_out) in sweeps.items()
            if not df.empty
        ],
        ignore_index=True,
    )
    combined.to_csv(args.csv_out, index=False)
    print(f"wrote {args.csv_out}")

    fig, axes = plt.subplots(2, 4, figsize=(17, 7), sharey="row")
    for col, (param, (df, timed_out)) in enumerate(sweeps.items()):
        if not df.empty:
            axes[0, col].plot(df[param], df["space_overhead"], marker="o")
            axes[1, col].plot(
                df[param], df["hash_evaluations"], marker="o", color="tab:orange"
            )

        for row in (0, 1):
            for value in timed_out:
                axes[row, col].axvline(
                    value,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="timed out" if value == timed_out[0] and row == 0 else None,
                )

        axes[0, col].set_title(f"space overhead vs {param}")
        axes[0, col].set_xlabel(param)
        axes[0, col].set_ylabel("space overhead (bits/key)")
        if timed_out:
            axes[0, col].legend()

        axes[1, col].set_xlabel(param)
        axes[1, col].set_ylabel("hash evaluations")
        if param == "b":
            axes[1, col].set_yscale("log")

    fig.suptitle(f"Parameter sweep against {args.file}\nbaseline: {baseline}")
    fig.tight_layout()
    fig.savefig(args.plot_out, dpi=150)
    print(f"wrote {args.plot_out}")


if __name__ == "__main__":
    main()
