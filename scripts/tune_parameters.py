#!/usr/bin/env python3
"""Sweep the `cli` binary's -b / --beta-scale / --eps-scale parameters
(the ones `Parameters::new_with_scales` takes) against a fixed --file input
and plot the resulting space overhead and construction cost
(hash_evaluations).

By default, for each parameter the other two are held at their baseline
(median) value while it's varied over the given list -- a one-at-a-time
sensitivity sweep, not a full grid search, so runtime stays linear in the
number of values tried instead of exploding combinatorially.

Pass --grid to instead run the full Cartesian product of all three
parameters' values. This is combinatorial (product of the list lengths), so
keep the lists short; it plots every combination's space/time tradeoff
instead of one line per parameter.

Usage:
    python3 scripts/tune_parameters.py --file path/to/kv.txt \
        --b 2 4 8 16 32 --beta-scale 0.5 1 2 --eps-scale 0.5 1 2

    python3 scripts/tune_parameters.py --file path/to/kv.txt --grid \
        --b 8 32 --beta-scale 0.5 1 --eps-scale 0.5 1

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
from matplotlib.lines import Line2D

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
    eps_scale: float,
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
        "--eps-scale",
        str(eps_scale),
    ]
    label = f"b={b} beta_scale={beta_scale} eps_scale={eps_scale}"
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
            f"eps_scale={kwargs['eps_scale']})"
        )
        result = run_once(
            binary,
            file,
            kwargs["b"],
            kwargs["beta_scale"],
            kwargs["eps_scale"],
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
    eps_scale_values,
    timeout: float,
) -> tuple[pd.DataFrame, list]:
    """Full Cartesian product over all three parameters, unlike `sweep`'s
    one-at-a-time. Cost is the product of the three list lengths, so this can
    get expensive fast -- keep the lists short when using --grid."""
    combos = list(itertools.product(b_values, beta_scale_values, eps_scale_values))
    worst_case_minutes = len(combos) * timeout / 60
    print(
        f"grid sweep: {len(combos)} combinations, up to {worst_case_minutes:.1f} "
        "min worst case (if every run hit the timeout)"
    )

    rows = []
    timed_out = []
    for i, (b, beta_scale, eps_scale) in enumerate(combos):
        print(
            f"[{i + 1}/{len(combos)}] running b={b} beta_scale={beta_scale} "
            f"eps_scale={eps_scale}"
        )
        result = run_once(binary, file, b, beta_scale, eps_scale, timeout)
        params = {"b": b, "beta_scale": beta_scale, "eps_scale": eps_scale}
        if result == "timeout":
            timed_out.append(params)
        elif result is not None:
            rows.append({**params, **result})
    return pd.DataFrame(rows), timed_out


def plot_tradeoff(
    df: pd.DataFrame,
    color_col: str,
    color_label: str,
    categorical: bool,
    title: str,
    out_path: Path,
    baseline: dict | None = None,
):
    """Scatter of every run's actual space/time tradeoff -- the thing you
    actually want to pick a point from -- colored by `color_col`, with the
    best (lowest space overhead) point picked out.

    When `categorical`, each category is one parameter's sweep line; on each
    of those lines, the point at that parameter's baseline value is marked
    with a square, its smallest swept value with a down-triangle, and its
    largest with an up-triangle (`baseline` maps category -> baseline
    value, since `color_col` is assumed to also be the parameter name whose
    swept values live in the like-named column)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    if categorical:
        cmap = plt.get_cmap("tab10")
        for i, category in enumerate(sorted(df[color_col].unique())):
            subset = df[df[color_col] == category]
            color = cmap(i % 10)
            ax.plot(
                subset["hash_evaluations"],
                subset["space_overhead"],
                color=color,
                alpha=0.8,
                label=str(category),
                marker=".",
            )

            # category is a parameter name (e.g. "b"); the values swept for
            # its own line live in the column of that same name.
            swept_values = subset[category]
            min_row = subset.loc[swept_values.idxmin()]
            max_row = subset.loc[swept_values.idxmax()]
            ax.scatter(min_row["hash_evaluations"], min_row["space_overhead"], color=color, marker="v", s=110, zorder=4)
            ax.scatter(max_row["hash_evaluations"], max_row["space_overhead"], color=color, marker="^", s=110, zorder=4)

            if baseline is not None and category in baseline:
                base_rows = subset[subset[category] == baseline[category]]
                if not base_rows.empty:
                    base_row = base_rows.iloc[0]
                    ax.scatter(
                        base_row["hash_evaluations"], base_row["space_overhead"], color=color, marker="s", s=90, zorder=4
                    )

        param_legend = ax.legend(title=color_label, loc="upper right")
        ax.add_artist(param_legend)
    else:
        scatter = ax.scatter(
            df["hash_evaluations"], df["space_overhead"], c=df[color_col], cmap="viridis", alpha=0.8
        )
        fig.colorbar(scatter, label=color_label)

    best = df.loc[df["space_overhead"].idxmin()]
    ax.scatter(
        [best["hash_evaluations"]],
        [best["space_overhead"]],
        color="red",
        marker="*",
        s=250,
        zorder=5,
    )

    marker_handles = [
        Line2D([], [], color="red", marker="*", linestyle="", markersize=13, label="best overhead"),
    ]
    if categorical:
        marker_handles += [
            Line2D([], [], color="black", marker="s", linestyle="", label="baseline"),
            Line2D([], [], color="black", marker="v", linestyle="", label="smallest value"),
            Line2D([], [], color="black", marker="^", linestyle="", label="largest value"),
        ]
    ax.legend(handles=marker_handles, loc="lower left")

    ax.set_xscale("log")
    ax.set_xlabel("hash evaluations")
    ax.set_ylabel("space overhead (bits/key)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    return best


def run_grid(args):
    df, timed_out = grid_sweep(
        args.bin, args.file, args.b, args.beta_scale, args.eps_scale, args.timeout
    )
    df.to_csv(args.csv_out, index=False)
    print(f"wrote {args.csv_out}")

    total = len(df) + len(timed_out)
    print(f"{len(df)}/{total} runs completed, {len(timed_out)} timed out")

    if df.empty:
        print("no successful runs; nothing to plot", file=sys.stderr)
        return

    best = plot_tradeoff(
        df,
        color_col="b",
        color_label="b",
        categorical=False,
        title=(
            f"Grid sweep against {args.file}\n{len(df)}/{total} runs completed, "
            f"{len(timed_out)} timed out"
        ),
        out_path=args.plot_out,
    )
    print(
        f"best space overhead: {best['space_overhead']:.4f} bits/key at "
        f"b={best['b']:.0f} beta_scale={best['beta_scale']} eps_scale={best['eps_scale']}"
    )


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
    parser.add_argument(
        "--eps-scale",
        dest="eps_scale",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0, 2.0, 4.0, 8.0],
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="per-run timeout in seconds"
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help=(
            "run a full grid (Cartesian product) sweep over all three "
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
    args.eps_scale.sort()

    if args.grid:
        run_grid(args)
        return

    baseline = {
        "b": args.b[len(args.b) // 2],
        "beta_scale": args.beta_scale[len(args.beta_scale) // 2],
        "eps_scale": args.eps_scale[len(args.eps_scale) // 2],
    }
    sweeps = {
        "b": sweep(args.bin, args.file, "b", args.b, baseline, args.timeout),
        "beta_scale": sweep(
            args.bin, args.file, "beta_scale", args.beta_scale, baseline, args.timeout
        ),
        "eps_scale": sweep(
            args.bin, args.file, "eps_scale", args.eps_scale, baseline, args.timeout
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

    total_timed_out = sum(len(timed_out) for _df, timed_out in sweeps.values())
    total = len(combined) + total_timed_out
    print(f"{len(combined)}/{total} runs completed, {total_timed_out} timed out")

    if combined.empty:
        print("no successful runs; nothing to plot", file=sys.stderr)
        return

    # One-at-a-time sweeps vary a different parameter each time, so rather
    # than one line-plot per parameter, combine every run into a single
    # space/time tradeoff scatter (like --grid's), colored by which
    # parameter was swept for that point.
    best = plot_tradeoff(
        combined,
        color_col="swept_param",
        color_label="swept parameter",
        categorical=True,
        title=(
            f"Parameter sweep against {args.file}\nbaseline: {baseline}\n"
            f"{len(combined)}/{total} runs completed, {total_timed_out} timed out"
        ),
        out_path=args.plot_out,
        baseline=baseline,
    )
    print(
        f"best space overhead: {best['space_overhead']:.4f} bits/key "
        f"(swept {best['swept_param']}={best[best['swept_param']]})"
    )


if __name__ == "__main__":
    main()
