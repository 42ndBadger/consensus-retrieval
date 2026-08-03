#!/usr/bin/env python3
"""Generate a b-curve tuning plot for each of several value distributions and
combine them into a single PDF via typst.

For every distribution below this script generates a key/value dataset with
the `cli` binary, runs `tune_parameters.py --b-curve` against it, and then
lays the resulting plots out in one document.

Usage:
    cargo build --release --bin cli
    python3 scripts/tuning_report.py
"""

# ---------------------------------------------------------------------------
# Swept parameter values -- these apply to every distribution.
# ---------------------------------------------------------------------------
B_VALUES = [ 16, 32, 64]
BETA_SCALES = [0.05, 0.1]
EPS_SCALES = [1.0, 2.0, 4.0]

# Number of key/value pairs generated per distribution.
NUM_KEYS = 10_000

# Per-run timeout, in seconds, handed to tune_parameters.py.
TIMEOUT = 30.0

# (slug, human-readable title, `--distribution` arguments)
DISTRIBUTIONS = [
    ("geometric", "Geometric, p = 1/2", ["geometric", "0.5"]),
    ("zipf", "Zipf, 10^6 values, exponent 2", ["zipf", "1000000", "2"]),
    ("uniform", "Uniform, 64 values", ["uniform", "64"]),
]
# ---------------------------------------------------------------------------

import argparse
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TUNE_SCRIPT = SCRIPT_DIR / "tune_parameters.py"

BEST_PATTERN = re.compile(r"best space overhead: (.+)$", re.MULTILINE)


def run(cmd) -> str:
    """Runs `cmd`, echoing its output live *and* returning it, so long sweeps
    show progress while callers can still scrape the result lines. stderr is
    folded into stdout so warnings interleave in order."""
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    lines = []
    proc = subprocess.Popen(
        cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    if proc.wait() != 0:
        sys.exit(f"command failed with exit {proc.returncode}")
    return "".join(lines)


def generate_dataset(binary: Path, dist_args: list[str], n: int, out_path: Path):
    run([str(binary), "--distribution", *dist_args, "-n", str(n), "--output", str(out_path)])


def run_b_curve(kv_path: Path, plot_path: Path, csv_path: Path, binary: Path, timeout: float):
    stdout = run(
        [
            sys.executable,
            "-u",  # unbuffered, so the sweep's progress streams live
            str(TUNE_SCRIPT),
            "--file", str(kv_path),
            "--bin", str(binary),
            "--b-curve",
            "--b", *[str(b) for b in B_VALUES],
            "--beta-scale", *[str(s) for s in BETA_SCALES],
            "--eps-scale", *[str(s) for s in EPS_SCALES],
            "--timeout", str(timeout),
            "--plot-out", str(plot_path),
            "--csv-out", str(csv_path),
        ]
    )
    match = BEST_PATTERN.search(stdout)
    return match.group(1) if match else "n/a"


def write_typst(out_dir: Path, sections: list[tuple[str, str, str]], n: int) -> Path:
    """`sections` is a list of (title, plot filename, best-overhead line)."""
    parts = [
        '#set page(paper: "a4", margin: 2cm)',
        "#set text(size: 10pt)",
        '#set par(justify: true)',
        "",
        "#align(center)[",
        "  #text(size: 17pt, weight: \"bold\")[Consensus retrieval parameter tuning]",
        "",
        f"  #text(size: 9pt)[{n} keys per dataset · "
        f"b {fmt_list(B_VALUES)} · beta-scale {fmt_list(BETA_SCALES)} · "
        f"eps-scale {fmt_list(EPS_SCALES)}]",
        "]",
        "",
        "Each plot sweeps `b` along every curve: one solid curve per `beta-scale`",
        "value and one dashed curve per `eps-scale` value, with the other held at",
        "its baseline. Down/up triangles mark the smallest/largest `b` on a curve.",
        "",
    ]
    for title, plot_name, best in sections:
        parts += [
            f"== {title}",
            f"Best: {best}",
            "",
            f'#figure(image("{plot_name}", width: 100%))',
            "",
            "#pagebreak(weak: true)",
            "",
        ]

    typ_path = out_dir / "report.typ"
    typ_path.write_text("\n".join(parts))
    print(f"wrote {typ_path}")
    return typ_path


def fmt_list(values) -> str:
    return ", ".join(f"{v:g}" for v in values)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bin", type=Path, default=Path("target/release/cli"))
    parser.add_argument("--out-dir", type=Path, default=Path("tuning_report"))
    parser.add_argument("-n", "--num-keys", type=int, default=NUM_KEYS)
    parser.add_argument("--timeout", type=float, default=TIMEOUT)
    parser.add_argument(
        "--reuse-data",
        action="store_true",
        help="keep existing generated datasets instead of regenerating them",
    )
    args = parser.parse_args()

    if not args.bin.exists():
        sys.exit(f"binary not found: {args.bin} (did you `cargo build --release --bin cli`?)")
    if not TUNE_SCRIPT.exists():
        sys.exit(f"missing {TUNE_SCRIPT}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sections = []
    for slug, title, dist_args in DISTRIBUTIONS:
        print(f"\n=== {title} ===")
        kv_path = args.out_dir / f"{slug}.kv"
        if args.reuse_data and kv_path.exists():
            print(f"reusing {kv_path}")
        else:
            generate_dataset(args.bin, dist_args, args.num_keys, kv_path)

        plot_name = f"{slug}.png"
        best = run_b_curve(
            kv_path,
            args.out_dir / plot_name,
            args.out_dir / f"{slug}.csv",
            args.bin,
            args.timeout,
        )
        sections.append((title, plot_name, best))

    typ_path = write_typst(args.out_dir, sections, args.num_keys)
    pdf_path = args.out_dir / "report.pdf"
    run(["typst", "compile", str(typ_path), str(pdf_path)])
    print(f"\nwrote {pdf_path}")


if __name__ == "__main__":
    main()
