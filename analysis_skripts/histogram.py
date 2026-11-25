#!/usr/bin/env python3
"""Plot a histogram of per-run time-averaged order from compute_order.csv.

Place beside compute_order.csv and compute_order.csv.cols.txt.
Each finite O_mean value contributes one sample; time series are not recomputed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Settings

# Match the stored tau column (tau for external alignment, J for pair alignment).
TAU_VALUE = 0.145
# Tolerances for numeric filters.
TAU_ATOL = 1e-12
TAU_RTOL = 1e-9

# None leaves a parameter unfiltered.
# cc is the cell count; p0 = 2 * permult / sqrt(A0); rand is the run index.
FILTERS: Dict[str, Any] = {
    "cc": None,
    "permult": None,
    "v": None,
    "nu": None,
    "gamma": None,
    "line": None,
    "k": None,
}

# Histogram settings
BINS = 10              # or use "auto"
DENSITY = False        # False = counts, True = probability density
SHOW_MEAN_LINE = False
SHOW_MEDIAN_LINE = False
SHOW_RUG = False        # Mark individual run values.

# Plot/output settings
FIGSIZE = (7.2, 4.8)
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 14
OUTPUT_DPI = 300
OUTPUT_DIRNAME = "Omean_hist_out"


def _read_cols_file(cols_path: Path) -> List[str]:
    txt = cols_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Cols:\s*([\s\S]+)", txt)
    if not m:
        raise ValueError(f"Could not find 'Cols:' in {cols_path}")
    tail = m.group(1).strip()
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            return [c.strip() for c in line.split(",") if c.strip()]
    raise ValueError(f"Could not parse column names from {cols_path}")


def _csv_has_header(csv_path: Path, cols: List[str]) -> bool:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
        first = handle.readline().strip()
    toks = [t.strip() for t in first.split(",")]
    return sum(1 for t in toks if t in set(cols)) >= 2


def _read_compute_order(here: Path) -> pd.DataFrame:
    csv_path = here / "compute_order.csv"
    cols_path = here / "compute_order.csv.cols.txt"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}")
    if not cols_path.exists():
        raise SystemExit(f"Missing {cols_path}")

    cols = _read_cols_file(cols_path)
    if _csv_has_header(csv_path, cols):
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None, names=cols)
    return df


def _filter_numeric_close(df: pd.DataFrame, col: str, value: float, atol: float, rtol: float) -> pd.DataFrame:
    if col not in df.columns:
        raise SystemExit(f"Missing required column: {col}")
    vals = pd.to_numeric(df[col], errors="coerce")
    return df[np.isclose(vals.to_numpy(float), float(value), atol=atol, rtol=rtol)].copy()


def _apply_optional_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    used: Dict[str, Any] = {}

    for col, value in filters.items():
        if value is None:
            continue
        if col not in out.columns:
            print(f"[WARN] Filter column {col!r} does not exist; ignoring it.")
            continue

        # Numeric filters use np.isclose, string filters use exact equality.
        numeric = pd.to_numeric(out[col], errors="coerce")
        try:
            value_float = float(value)
            if np.isfinite(numeric).any():
                out = out[np.isclose(numeric.to_numpy(float), value_float, atol=TAU_ATOL, rtol=TAU_RTOL)]
            else:
                out = out[out[col].astype(str) == str(value)]
        except Exception:
            out = out[out[col].astype(str) == str(value)]

        used[col] = value

    return out, used


def _safe_tag(x: Any) -> str:
    return str(x).replace(".", "p").replace("-", "m").replace(" ", "")


def main() -> None:
    here = Path(__file__).resolve().parent
    df = _read_compute_order(here)

    for col in ["tau", "O_mean"]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")

    df = _filter_numeric_close(df, "tau", TAU_VALUE, TAU_ATOL, TAU_RTOL)
    df, used_filters = _apply_optional_filters(df, FILTERS)

    if df.empty:
        raise SystemExit("No rows left after filtering. Check TAU_VALUE and FILTERS.")

    O = pd.to_numeric(df["O_mean"], errors="coerce").to_numpy(dtype=float)
    O = O[np.isfinite(O)]
    if O.size == 0:
        raise SystemExit("Rows matched, but no finite O_mean values found.")

    valid_rows = df.loc[np.isfinite(pd.to_numeric(df["O_mean"], errors="coerce"))]
    varying = [
        col for col in FILTERS
        if col in valid_rows.columns and valid_rows[col].nunique(dropna=False) > 1
    ]
    if varying:
        print(
            "[WARN] Histogram combines runs with different "
            + ", ".join(varying)
            + ". Set FILTERS to select a single parameter combination if intended."
        )

    outdir = here / OUTPUT_DIRNAME
    outdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)

    ax.hist(O, bins=BINS, density=DENSITY, color="crimson", edgecolor="black", alpha=0.75)

    if SHOW_MEAN_LINE:
        ax.axvline(float(np.mean(O)), linestyle="--", linewidth=2.0, label=f"mean = {np.mean(O):.3g}")
    if SHOW_MEDIAN_LINE:
        ax.axvline(float(np.median(O)), linestyle=":", linewidth=2.0, label=f"median = {np.median(O):.3g}")

    if SHOW_RUG:
        ymin, ymax = ax.get_ylim()
        rug_y0 = ymin
        rug_y1 = ymin + 0.04 * (ymax - ymin)
        ax.vlines(O, rug_y0, rug_y1, linewidth=0.8, alpha=0.5)
        ax.set_ylim(ymin, ymax)

    ax.set_xlabel(r"$\langle O\rangle_t$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("count" if not DENSITY else "density", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_SIZE)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, axis="y", alpha=0.3)
    if SHOW_MEAN_LINE or SHOW_MEDIAN_LINE:
        ax.legend(frameon=False)

    fig.tight_layout()

    tag_parts = [f"tau{_safe_tag(f'{TAU_VALUE:g}')}"] + [f"{k}{_safe_tag(v)}" for k, v in used_filters.items()]
    tag = "__".join(tag_parts)
    png_path = outdir / f"Omean_hist__{tag}.png"
    csv_path = outdir / f"Omean_values__{tag}.csv"

    fig.savefig(png_path, dpi=OUTPUT_DPI)
    plt.close(fig)

    # Export all matching rows, including rows with invalid O_mean values.
    df.to_csv(csv_path, index=False)

    print("[OK] Wrote:")
    print(" ", png_path)
    print(" ", csv_path)
    print("[INFO] n =", len(O))
    print("[INFO] mean =", float(np.mean(O)))
    print("[INFO] median =", float(np.median(O)))
    print("[INFO] std =", float(np.std(O, ddof=1)) if len(O) > 1 else float("nan"))


if __name__ == "__main__":
    main()
