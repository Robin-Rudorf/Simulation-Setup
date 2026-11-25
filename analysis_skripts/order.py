#!/usr/bin/env python3
"""Plot time-averaged order against alignment strength for each system size.

Place beside compute_order.csv and compute_order.csv.cols.txt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# Settings
MODE = "tau"  # "tau" (external, J=1/tau) or "J" (pair, tau column already is J)

PERMULT_LIST = None  # None includes all permult values.

# None selects the most frequent value for each filter.
FILTERS = {
    "v": None,
    "nu": None,
    "gamma": None,
    "line": None,
    "k": None,
}

# Plot choices
CMAP_NAME = "Blues"
CMAP_MIN = 0.25
CMAP_MAX = 1.00
SHOW_STD_SHADE = False  # Standard deviation across runs, not standard error.
OUTPUT_DPI = 300


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
            cols = [c.strip() for c in line.split(",") if c.strip()]
            return cols
    raise ValueError(f"Could not parse column names from {cols_path}")


def _csv_has_header(csv_path: Path, cols: List[str]) -> bool:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
        first = handle.readline().strip()
    toks = [t.strip() for t in first.split(",")]
    hits = sum(1 for t in toks if t in set(cols))
    return hits >= 2


def _auto_pick_mode_value(df: pd.DataFrame, key: str):
    if key not in df.columns:
        return None
    s = df[key].dropna()
    if s.empty:
        return None
    return s.value_counts().idxmax()


def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    chosen = {}
    out = df.copy()
    for k, v in filters.items():
        if k not in out.columns:
            continue
        if v is None:
            v = _auto_pick_mode_value(out, k)
        chosen[k] = v
        if v is not None:
            out = out[out[k] == v]
    return out, chosen


def _compute_J(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "tau" not in out.columns:
        raise ValueError("Expected column 'tau' to exist in compute_order.csv")
    if MODE.lower() == "tau":
        tau = pd.to_numeric(out["tau"], errors="coerce")
        out["J"] = 1.0 / tau
    elif MODE.lower() == "j":
        out["J"] = pd.to_numeric(out["tau"], errors="coerce")
    else:
        raise ValueError("MODE must be 'tau' or 'J'")
    out = out[np.isfinite(out["J"]) & (out["J"] > 0)]
    return out


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["O_mean", "cc", "permult", "J"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    df2 = df.copy()
    df2["O_mean"] = pd.to_numeric(df2["O_mean"], errors="coerce")
    df2["cc"] = pd.to_numeric(df2["cc"], errors="coerce")
    df2["permult"] = pd.to_numeric(df2["permult"], errors="coerce")
    df2 = df2[np.isfinite(df2["O_mean"]) & np.isfinite(df2["cc"]) & np.isfinite(df2["permult"])]
    # Average the time-averaged order values across runs.
    g = df2.groupby(["permult", "cc", "J"], dropna=False)["O_mean"]
    agg = g.agg(["mean", "std", "count"]).reset_index().rename(columns={"mean": "Obar", "std": "Ostd", "count": "n"})
    return agg


def _plot_one(agg: pd.DataFrame, permult_val: float, outdir: Path) -> Path:
    sub = agg[agg["permult"] == permult_val].copy()
    if sub.empty:
        raise ValueError(f"No data for permult={permult_val} after filtering.")

    sizes = sorted(sub["cc"].unique())
    sqrtN = np.array([math.sqrt(float(s)) for s in sizes], dtype=float)
    norm = mpl.colors.Normalize(vmin=float(sqrtN.min()), vmax=float(sqrtN.max()))
    base_cmap = plt.get_cmap(CMAP_NAME)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        f"{CMAP_NAME}_trunc",
        base_cmap(np.linspace(CMAP_MIN, CMAP_MAX, 256))
    )

    fig = plt.figure(figsize=(7.6, 5.0))
    ax = fig.add_subplot(111)

    for cc in sizes:
        d = sub[sub["cc"] == cc].sort_values("J")
        x = d["J"].to_numpy(float)
        y = d["Obar"].to_numpy(float)
        color = cmap(norm(math.sqrt(float(cc))))
        ax.plot(x, y, marker="o", linewidth=2.0, markersize=4.0, color=color)
        if SHOW_STD_SHADE:
            ystd = d["Ostd"].to_numpy(float)
            if np.isfinite(ystd).any():
                lo = y - np.nan_to_num(ystd, nan=0.0)
                hi = y + np.nan_to_num(ystd, nan=0.0)
                ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel(r"$J$")
    ax.set_ylabel(r"$\langle O \rangle_t$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\sqrt{N}$", rotation=90)

    fig.tight_layout()
    fname = f"order_vs_J__permult{permult_val:g}__mode{MODE.lower()}.png"
    outpath = outdir / fname
    fig.savefig(outpath, dpi=OUTPUT_DPI)
    plt.close(fig)
    return outpath


def main():
    here = Path(__file__).resolve().parent
    csv_path = here / "compute_order.csv"
    cols_path = here / "compute_order.csv.cols.txt"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}")
    if not cols_path.exists():
        raise SystemExit(f"Missing {cols_path}")

    cols = _read_cols_file(cols_path)
    has_header = _csv_has_header(csv_path, cols)
    if has_header:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None, names=cols)

    df = _compute_J(df)
    df_f, chosen = _apply_filters(df, FILTERS)
    if df_f.empty:
        raise SystemExit("No rows left after filtering. Try loosening FILTERS.")

    agg = _aggregate(df_f)
    outdir = here / "order_vs_J_out"
    outdir.mkdir(parents=True, exist_ok=True)

    permults = sorted(agg["permult"].unique()) if PERMULT_LIST is None else PERMULT_LIST

    written = []
    for p in permults:
        try:
            written.append(_plot_one(agg, float(p), outdir))
        except Exception as e:
            print(f"[WARN] Skipping permult={p}: {e}")

    print("[INFO] MODE:", MODE)
    print("[INFO] Filters used:", chosen)
    print("[INFO] Wrote:")
    for w in written:
        print(" ", w)


if __name__ == "__main__":
    main()
