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

Pass --b-curve to instead sweep --b once per --beta-scale value (solid
lines) and once per --eps-scale value (dashed lines), plotting each as its
own curve -- useful for seeing how the space/time tradeoff across b shifts
as beta_scale or eps_scale changes.

Every mode also writes a --table-out Typst file (compiled to a PDF
alongside it) breaking each run's space down into consensus vector / raw
insertion vector / select structure, as percentages of their total.

Usage:
    python3 scripts/tune_parameters.py --file path/to/kv.txt \
        --b 2 4 8 16 32 --beta-scale 0.5 1 2 --eps-scale 0.5 1 2

    python3 scripts/tune_parameters.py --file path/to/kv.txt --grid \
        --b 8 32 --beta-scale 0.5 1 --eps-scale 0.5 1

    python3 scripts/tune_parameters.py --file path/to/kv.txt --b-curve \
        --b 2 4 8 16 32 --beta-scale 0.5 1 2 --eps-scale 0.5 1 2

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
    "space_bytes": re.compile(r"space \[byte\]:\s*(\d+)"),
    "space_overhead": re.compile(r"space overhead \[bits/key\]:\s*([-\d.eE]+)"),
    "raw_insertion_vec_bits": re.compile(r"raw_insertion_vec_bits \[bit\]:\s*(\d+)"),
    "select_structure_bits": re.compile(r"select_structure_bits \[bit\]:\s*(\d+)"),
    "consensus_vec_bits": re.compile(r"consensus_vec_bits \[bit\]:\s*(\d+)"),
    "hash_evaluations": re.compile(r"time \[hash evaluations\]:\s*(\d+)"),
}

# The three space components a run's total (variable-part) space breaks
# down into -- used both for the CSV columns above and for the percentage
# breakdown table.
SPACE_COMPONENTS = ["consensus_vec_bits", "raw_insertion_vec_bits", "select_structure_bits"]


def _annotate_timeouts(ax, entries: list[tuple[object, str]]):
    """Stacks one colored, boxed line of text per entry in the bottom-right
    corner (the one spot the curve-identity legend at upper-right and the
    marker-meaning legend at lower-left both leave free): `entries` is
    (color, text) pairs. A timed-out run has no real (x, y) to plot -- it
    never produced space_overhead/hash_evaluations -- so rather than invent
    a position for it (which risks looking like real data), state plainly
    in the color of the curve/category it belongs to which values timed
    out."""
    if not entries:
        return
    ax.annotate(
        "timed out:",
        xy=(0.99, 0.01 + 0.04 * len(entries)),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="black",
    )
    for i, (color, text) in enumerate(entries):
        ax.annotate(
            text,
            xy=(0.99, 0.01 + 0.04 * (len(entries) - i - 1)),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.85),
        )


def _typst_escape(text: str) -> str:
    """Escapes markup-significant characters so arbitrary cell text is
    always rendered literally in Typst content mode."""
    return text.replace("\\", "\\\\").replace("_", "\\_").replace("*", "\\*").replace("#", "\\#")


def write_typst_table(df: pd.DataFrame, id_cols: list[str], out_path: Path):
    """Writes a Typst table with the space breakdown -- consensus vector /
    raw insertion vector / select structure, as a percentage of their total
    (the variable, input-size-scaling part of the structure) -- one row per
    successful run -- then compiles it to a PDF alongside it."""
    if df.empty:
        print(f"no successful runs; not writing {out_path}", file=sys.stderr)
        return

    total = df[SPACE_COMPONENTS].sum(axis=1)
    table = df[[c for c in id_cols if c in df.columns]].copy()
    for col in SPACE_COMPONENTS:
        table[col] = df[col].astype("Int64")
        table[col.removesuffix("_bits") + "_%"] = (df[col] / total * 100).round(2)

    def cell(v) -> str:
        return "" if pd.isna(v) else _typst_escape(str(v))

    header_cells = ", ".join(f"[*{_typst_escape(c)}*]" for c in table.columns)
    row_lines = [
        ", ".join(f"[{cell(v)}]" for v in row) for row in table.itertuples(index=False)
    ]

    parts = [
        '#set page(paper: "a4", flipped: true, margin: 1.5cm)',
        "#set text(size: 8pt)",
        "= Space breakdown",
        "",
        "#table(",
        f"  columns: {len(table.columns)} * (auto,),",
        "  align: center,",
        f"  {header_cells},",
        *(f"  {line}," for line in row_lines),
        ")",
        "",
    ]
    out_path.write_text("\n".join(parts))
    print(f"wrote {out_path}")

    pdf_path = out_path.with_suffix(".pdf")
    try:
        proc = subprocess.run(
            ["typst", "compile", str(out_path), str(pdf_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"typst compile failed:\n{proc.stderr}", file=sys.stderr)
        else:
            print(f"wrote {pdf_path}")
    except FileNotFoundError:
        print("typst not found on PATH; left the .typ file uncompiled", file=sys.stderr)


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
    timed_out_by_category: dict | None = None,
    timed_out: list | None = None,
):
    """Scatter of every run's actual space/time tradeoff -- the thing you
    actually want to pick a point from -- colored by `color_col`, with the
    best (lowest space overhead) point picked out.

    When `categorical`, each category is one parameter's sweep line; on each
    of those lines, the point at that parameter's baseline value is marked
    with a square, its smallest swept value with a down-triangle, and its
    largest with an up-triangle (`baseline` maps category -> baseline
    value, since `color_col` is assumed to also be the parameter name whose
    swept values live in the like-named column). `timed_out_by_category`
    maps category -> list of values that timed out on that category's line.

    When not `categorical`, `timed_out` is a flat list of the param dicts
    (b/beta_scale/eps_scale) that timed out, listed in a corner box (since
    there's no discrete per-category color to attach them to)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    if categorical:
        cmap = plt.get_cmap("tab10")
        timeout_entries = []
        categories = sorted(set(df[color_col].unique()) | set((timed_out_by_category or {}).keys()))
        for i, category in enumerate(categories):
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
            # its own line live in the column of that same name. A category
            # can be entirely absent from `df` if every value on its line
            # timed out -- still give it a legend entry rather than letting
            # it vanish silently.
            if not subset.empty:
                swept_values = subset[category]
                min_row = subset.loc[swept_values.idxmin()]
                max_row = subset.loc[swept_values.idxmax()]
                ax.scatter(min_row["hash_evaluations"], min_row["space_overhead"], color=color, marker="v", s=110, zorder=4)
                ax.scatter(max_row["hash_evaluations"], max_row["space_overhead"], color=color, marker="^", s=110, zorder=4)

            if baseline is not None and category in baseline and category in subset.columns:
                base_rows = subset[subset[category] == baseline[category]]
                if not base_rows.empty:
                    base_row = base_rows.iloc[0]
                    ax.scatter(
                        base_row["hash_evaluations"], base_row["space_overhead"], color=color, marker="s", s=90, zorder=4
                    )

            timed_out_values = sorted((timed_out_by_category or {}).get(category, []))
            if timed_out_values:
                timeout_entries.append(
                    (color, f"{category}={','.join(f'{v:g}' for v in timed_out_values)}")
                )

        param_legend = ax.legend(title=color_label, loc="upper right")
        ax.add_artist(param_legend)
        _annotate_timeouts(ax, timeout_entries)
    else:
        # b drives point color (shared colormap/scale across markers below);
        # marker shape is one symbol per (beta_scale, eps_scale) combo, so
        # all three swept dimensions are visible in one scatter.
        markers = ["o", "s", "^", "v", "D", "P", "X", "*", "h", "8", "<", ">", "p", "H", "d"]
        combos = sorted(
            df[["beta_scale", "eps_scale"]].drop_duplicates().itertuples(index=False, name=None)
        )
        vmin, vmax = df[color_col].min(), df[color_col].max()
        scatter = None
        for i, (beta_scale, eps_scale) in enumerate(combos):
            subset = df[(df["beta_scale"] == beta_scale) & (df["eps_scale"] == eps_scale)]
            scatter = ax.scatter(
                subset["hash_evaluations"],
                subset["space_overhead"],
                c=subset[color_col],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                marker=markers[i % len(markers)],
                edgecolors="black",
                linewidths=0.4,
                alpha=0.85,
                label=f"beta_scale={beta_scale:g}, eps_scale={eps_scale:g}",
            )
        fig.colorbar(scatter, label=color_label)
        combo_legend = ax.legend(
            title="(beta_scale, eps_scale)", loc="upper right", fontsize=7, title_fontsize=7
        )
        ax.add_artist(combo_legend)

        if timed_out:
            shown = timed_out[:5]
            lines = [
                "b={b} beta_scale={beta_scale} eps_scale={eps_scale}".format(**p) for p in shown
            ]
            if len(timed_out) > len(shown):
                lines.append(f"... and {len(timed_out) - len(shown)} more")
            _annotate_timeouts(ax, [("firebrick", line) for line in lines])

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


def plot_b_curve(
    combined: pd.DataFrame,
    timeouts_by_curve: dict[tuple[str, float], list[float]],
    title: str,
    out_path: Path,
):
    """One curve per (curve_type, curve_value): curve_type is "beta_scale"
    or "eps_scale" (solid vs dashed line), curve_value is the held value of
    whichever of those two wasn't varied. Each curve's points are its `b`
    sweep, colored distinctly and labeled in the legend; the smallest `b`
    on a curve is marked with a down-triangle, the largest with an
    up-triangle.

    `timeouts_by_curve` maps (curve_type, curve_value) -> the `b` values
    that timed out on that curve, so a curve that timed out at its largest
    (or every) `b` doesn't just quietly stop short looking complete --
    those are listed in a corner box, in that curve's own color. A curve
    where every `b` timed out still gets a (empty) line and legend entry,
    so it doesn't silently vanish either."""
    fig, ax = plt.subplots(figsize=(10, 7))

    curve_keys = sorted(
        set(combined.groupby(["curve_type", "curve_value"]).groups.keys())
        | set(timeouts_by_curve.keys()),
        key=lambda k: (k[0], k[1]),
    )
    cmap = plt.get_cmap("tab10")
    timeout_entries = []
    for i, (curve_type, curve_value) in enumerate(curve_keys):
        subset = combined[
            (combined["curve_type"] == curve_type) & (combined["curve_value"] == curve_value)
        ].sort_values("b")
        color = cmap(i % 10)
        linestyle = "-" if curve_type == "beta_scale" else "--"
        ax.plot(
            subset["hash_evaluations"],
            subset["space_overhead"],
            color=color,
            linestyle=linestyle,
            marker=".",
            alpha=0.85,
            label=f"{curve_type}={curve_value:g}",
        )

        if not subset.empty:
            min_row = subset.loc[subset["b"].idxmin()]
            max_row = subset.loc[subset["b"].idxmax()]
            ax.scatter(min_row["hash_evaluations"], min_row["space_overhead"], color=color, marker="v", s=110, zorder=4)
            ax.scatter(max_row["hash_evaluations"], max_row["space_overhead"], color=color, marker="^", s=110, zorder=4)

        timed_out_bs = sorted(timeouts_by_curve.get((curve_type, curve_value), []))
        if timed_out_bs:
            timeout_entries.append(
                (color, f"{curve_type}={curve_value:g}: b={','.join(f'{b:g}' for b in timed_out_bs)}")
            )

    curve_legend = ax.legend(
        title="curve (solid=beta_scale, dashed=eps_scale)", loc="upper right", fontsize=8
    )
    ax.add_artist(curve_legend)
    _annotate_timeouts(ax, timeout_entries)

    best = combined.loc[combined["space_overhead"].idxmin()]
    ax.scatter([best["hash_evaluations"]], [best["space_overhead"]], color="red", marker="*", s=250, zorder=5)

    marker_handles = [
        Line2D([], [], color="red", marker="*", linestyle="", markersize=13, label="best overhead"),
        Line2D([], [], color="black", marker="v", linestyle="", label="smallest b"),
        Line2D([], [], color="black", marker="^", linestyle="", label="largest b"),
    ]
    ax.legend(handles=marker_handles, loc="lower left")

    ax.set_xscale("log")
    ax.set_xlabel("hash evaluations")
    ax.set_ylabel("space overhead (bits/key)")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title}\nsmallest overhead: {best['space_overhead']:.4f} bits/key")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    return best


def run_b_curve(args):
    baseline_beta_scale = args.beta_scale[len(args.beta_scale) // 2]
    baseline_eps_scale = args.eps_scale[len(args.eps_scale) // 2]
    total_runs = len(args.b) * (len(args.beta_scale) + len(args.eps_scale))
    print(
        f"b-curve sweep: {len(args.beta_scale)} beta_scale curves + "
        f"{len(args.eps_scale)} eps_scale curves, {len(args.b)} b values each "
        f"({total_runs} runs total)"
    )

    frames = []
    timeouts_by_curve = {}
    for beta_scale in args.beta_scale:
        print(f"-- curve beta_scale={beta_scale} --")
        df, timed_out = sweep(
            args.bin,
            args.file,
            "b",
            args.b,
            {"beta_scale": beta_scale, "eps_scale": baseline_eps_scale},
            args.timeout,
        )
        if timed_out:
            timeouts_by_curve[("beta_scale", beta_scale)] = timed_out
        if not df.empty:
            frames.append(df.assign(curve_type="beta_scale", curve_value=beta_scale))

    for eps_scale in args.eps_scale:
        print(f"-- curve eps_scale={eps_scale} --")
        df, timed_out = sweep(
            args.bin,
            args.file,
            "b",
            args.b,
            {"beta_scale": baseline_beta_scale, "eps_scale": eps_scale},
            args.timeout,
        )
        if timed_out:
            timeouts_by_curve[("eps_scale", eps_scale)] = timed_out
        if not df.empty:
            frames.append(df.assign(curve_type="eps_scale", curve_value=eps_scale))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(args.csv_out, index=False)
    print(f"wrote {args.csv_out}")
    write_typst_table(combined, ["curve_type", "curve_value", "b"], args.table_out)

    total_timed_out = sum(len(v) for v in timeouts_by_curve.values())
    completed = len(combined)
    total = completed + total_timed_out
    print(f"{completed}/{total} runs completed, {total_timed_out} timed out")

    if combined.empty:
        print("no successful runs; nothing to plot", file=sys.stderr)
        return

    best = plot_b_curve(
        combined,
        timeouts_by_curve,
        title=(
            f"b-curve sweep against {args.file}\n"
            f"baseline: beta_scale={baseline_beta_scale}, eps_scale={baseline_eps_scale}\n"
            f"{completed}/{total} runs completed, {total_timed_out} timed out"
        ),
        out_path=args.plot_out,
    )
    print(
        f"best space overhead: {best['space_overhead']:.4f} bits/key at "
        f"b={best['b']:.0f} {best['curve_type']}={best['curve_value']}"
    )


def run_grid(args):
    df, timed_out = grid_sweep(
        args.bin, args.file, args.b, args.beta_scale, args.eps_scale, args.timeout
    )
    df.to_csv(args.csv_out, index=False)
    print(f"wrote {args.csv_out}")
    write_typst_table(df, ["b", "beta_scale", "eps_scale"], args.table_out)

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
        timed_out=timed_out,
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
    parser.add_argument(
        "--b-curve",
        dest="b_curve",
        action="store_true",
        help=(
            "for every --beta-scale and --eps-scale value, sweep --b and "
            "plot it as its own curve (solid line per beta_scale value, "
            "dashed per eps_scale value), instead of one-at-a-time"
        ),
    )
    parser.add_argument("--csv-out", type=Path, default=Path("tuning_results.csv"))
    parser.add_argument("--plot-out", type=Path, default=Path("tuning_results.png"))
    parser.add_argument(
        "--table-out",
        type=Path,
        default=Path("tuning_breakdown.typ"),
        help=(
            "Typst file (compiled to a same-named .pdf) for the per-run "
            "space breakdown (consensus vector / raw insertion vector / "
            "select structure, as percentages)"
        ),
    )
    args = parser.parse_args()

    if not args.bin.exists():
        sys.exit(
            f"binary not found: {args.bin} (did you `cargo build --release --bin cli`?)"
        )

    if not args.file.exists():
        sys.exit(f"file not found: {args.file}")

    if args.grid and args.b_curve:
        sys.exit("--grid and --b-curve are mutually exclusive")

    args.b.sort()
    args.beta_scale.sort()
    args.eps_scale.sort()

    if args.grid:
        run_grid(args)
        return

    if args.b_curve:
        run_b_curve(args)
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
    write_typst_table(
        combined, ["swept_param", "b", "beta_scale", "eps_scale"], args.table_out
    )

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
        timed_out_by_category={name: t for name, (_df, t) in sweeps.items() if t},
    )
    print(
        f"best space overhead: {best['space_overhead']:.4f} bits/key "
        f"(swept {best['swept_param']}={best[best['swept_param']]})"
    )


if __name__ == "__main__":
    main()
