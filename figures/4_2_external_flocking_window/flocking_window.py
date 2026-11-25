#!/usr/bin/env python3
"""Plot collective order versus alignment strength for selected shape parameters.

Place beside compute_order.csv and compute_order.csv.cols.txt.
The default overlay uses N=1000, p0 colours, and a J**0.1 axis scale.
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

MODE = "tau"  # "tau": J = 1/tau; "J": the tau column stores J.
PLOT_MODE = "permults_by_size"  # Alternatively: "sizes_by_perm".

A0 = 3.0  # Preferred cell area.

PERMULT_LIST = None  # For sizes_by_perm; None includes all permult values.

CC_VALUE = 1000  # Cell count for the overlay.

PERMULT_OVERLAY_LIST = None  # Raw permult values, not p0; None includes all.

# None selects the most frequent value for each filter.
FILTERS = {
    "v": None,
    "nu": None,
    "gamma": None,
    "line": None,
    "k": None,
}

SIZE_CMAP_NAME = "Blues"
SIZE_CMAP_MIN = 0.25
SIZE_CMAP_MAX = 1.00

PERMULT_CMAP_NAME = "plasma"
PERMULT_CMAP_MIN = 0.00
PERMULT_CMAP_MAX = 1.00

SHOW_STD_SHADE = False  # Standard deviation across runs, not standard error.
OUTPUT_DPI = 300

# Options: "linear", "log", "symlog", "power", "equal_points".
X_AXIS_MODE = "power"
X_POWER_EXPONENT = 0.1
X_SYMLOG_LINTHRESH = 0.05
X_EQUAL_POINTS_MAX_TICKS = 12


def permult_to_p0(permult):
    """Convert permult to the dimensionless target shape index."""
    if A0 <= 0:
        raise ValueError("A0 must be positive.")
    return 2.0 * np.asarray(permult, dtype=float) / np.sqrt(A0)


def _read_cols_file(cols_path: Path) -> List[str]:
    txt = cols_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Cols:\s*([\s\S]+)", txt)

    if not match:
        raise ValueError(f"Could not find 'Cols:' in {cols_path}")

    for line in match.group(1).strip().splitlines():
        line = line.strip()
        if line and "," in line:
            return [
                column.strip()
                for column in line.split(",")
                if column.strip()
            ]

    raise ValueError(f"Could not parse column names from {cols_path}")


def _csv_has_header(csv_path: Path, cols: List[str]) -> bool:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
        first = handle.readline().strip()

    tokens = [token.strip() for token in first.split(",")]
    return sum(token in set(cols) for token in tokens) >= 2


def _auto_pick_mode_value(df: pd.DataFrame, key: str):
    if key not in df.columns:
        return None

    values = df[key].dropna()
    if values.empty:
        return None

    return values.value_counts().idxmax()


def _apply_filters(
    df: pd.DataFrame,
    filters: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    chosen = {}
    out = df.copy()

    for key, value in filters.items():
        if key not in out.columns:
            continue

        if value is None:
            value = _auto_pick_mode_value(out, key)

        chosen[key] = value
        if value is not None:
            out = out[out[key] == value]

    return out, chosen


def _compute_J(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "tau" not in out.columns:
        raise ValueError("Expected column 'tau' in compute_order.csv")

    values = pd.to_numeric(out["tau"], errors="coerce")

    if MODE.lower() == "tau":
        out["J"] = 1.0 / values
    elif MODE.lower() == "j":
        out["J"] = values
    else:
        raise ValueError("MODE must be 'tau' or 'J'")

    return out[np.isfinite(out["J"]) & (out["J"] > 0)]


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    required = ["O_mean", "cc", "permult", "J"]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    data = df.copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    valid = np.ones(len(data), dtype=bool)
    for column in required:
        valid &= np.isfinite(data[column].to_numpy(dtype=float))
    data = data.loc[valid]

    # Average per-run temporal means at each parameter combination.
    grouped = data.groupby(
        ["permult", "cc", "J"], dropna=False
    )["O_mean"]

    return (
        grouped.agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "Obar",
            "std": "Ostd",
            "count": "n",
        })
    )


def _format_tick(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) >= 100 or (0 < abs(value) < 0.01):
        return f"{value:.1e}"
    return f"{value:g}"


def _equal_point_map(j_values: np.ndarray) -> Dict[float, float]:
    values = sorted(
        float(value)
        for value in np.unique(j_values[np.isfinite(j_values)])
    )
    return {value: float(index) for index, value in enumerate(values)}


def _nearest_equal_positions(
    values: np.ndarray,
    position_map: Dict[float, float],
) -> np.ndarray:
    keys = np.array(list(position_map.keys()), dtype=float)
    positions = []

    for value in values:
        index = int(np.argmin(np.abs(keys - float(value))))
        positions.append(position_map[float(keys[index])])

    return np.array(positions, dtype=float)


def _setup_x_axis(
    ax,
    j_values: np.ndarray,
    position_map: Dict[float, float] | None = None,
) -> None:
    mode = X_AXIS_MODE.lower().strip()
    finite_j = np.asarray(j_values[np.isfinite(j_values)], dtype=float)

    if mode == "linear":
        ax.set_xlabel(r"$J$")
        return

    if mode == "log":
        if finite_j.size and np.all(finite_j > 0):
            ax.set_xscale("log")
            ax.set_xlabel(r"$J$ (log scale)")
        else:
            print("[WARN] Log scale requires positive J; using linear.")
            ax.set_xlabel(r"$J$")
        return

    if mode == "symlog":
        ax.set_xscale(
            "symlog",
            linthresh=float(X_SYMLOG_LINTHRESH),
        )
        ax.set_xlabel(r"$J$ (symlog scale)")
        return

    if mode == "power":
        exponent = float(X_POWER_EXPONENT)
        if not 0 < exponent <= 1:
            raise ValueError("X_POWER_EXPONENT must lie in (0, 1].")

        def forward(values):
            values = np.asarray(values, dtype=float)
            return np.sign(values) * np.abs(values) ** exponent

        def inverse(values):
            values = np.asarray(values, dtype=float)
            return np.sign(values) * np.abs(values) ** (1.0 / exponent)

        ax.set_xscale("function", functions=(forward, inverse))
        ax.set_xlabel(
            rf"$J$ (power scale, $J^{{{exponent:g}}}$)"
        )
        return

    if mode == "equal_points":
        if position_map is None:
            raise ValueError("Equal-point mode requires a position map.")

        original_j = np.array(sorted(position_map), dtype=float)
        positions = np.array(
            [position_map[value] for value in original_j],
            dtype=float,
        )

        if len(positions) > X_EQUAL_POINTS_MAX_TICKS:
            indices = np.unique(
                np.linspace(
                    0,
                    len(positions) - 1,
                    X_EQUAL_POINTS_MAX_TICKS,
                ).round().astype(int)
            )
            positions = positions[indices]
            original_j = original_j[indices]

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [_format_tick(value) for value in original_j],
            rotation=35,
            ha="right",
        )
        ax.set_xlabel(r"$J$ (equal spacing of measured points)")
        return

    raise ValueError(
        "X_AXIS_MODE must be 'linear', 'log', 'symlog', "
        "'power', or 'equal_points'."
    )


def _truncated_cmap(name: str, lower: float, upper: float):
    base = mpl.colormaps[name]
    return mpl.colors.LinearSegmentedColormap.from_list(
        f"{name}_truncated",
        base(np.linspace(lower, upper, 256)),
    )


def _shade_std(ax, x, y, data, color):
    if not SHOW_STD_SHADE:
        return

    std = data["Ostd"].to_numpy(dtype=float)
    if np.isfinite(std).any():
        std = np.nan_to_num(std, nan=0.0)
        ax.fill_between(
            x,
            y - std,
            y + std,
            color=color,
            alpha=0.15,
            linewidth=0,
        )


def _plot_sizes_by_perm_one(
    agg: pd.DataFrame,
    permult_val: float,
    outdir: Path,
) -> Path:
    sub = agg[
        np.isclose(agg["permult"].astype(float), permult_val)
    ].copy()

    if sub.empty:
        raise ValueError(f"No data for permult={permult_val}.")

    sizes = sorted(sub["cc"].unique())
    sqrt_n = np.array([math.sqrt(float(size)) for size in sizes])
    norm = mpl.colors.Normalize(
        vmin=float(sqrt_n.min()),
        vmax=float(sqrt_n.max()),
    )
    cmap = _truncated_cmap(
        SIZE_CMAP_NAME, SIZE_CMAP_MIN, SIZE_CMAP_MAX
    )

    all_j = sub["J"].to_numpy(dtype=float)
    position_map = (
        _equal_point_map(all_j)
        if X_AXIS_MODE.lower().strip() == "equal_points"
        else None
    )

    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    for cc in sizes:
        data = sub[sub["cc"] == cc].sort_values("J")
        j = data["J"].to_numpy(dtype=float)
        x = (
            _nearest_equal_positions(j, position_map)
            if position_map is not None
            else j
        )
        y = data["Obar"].to_numpy(dtype=float)
        color = cmap(norm(math.sqrt(float(cc))))

        ax.plot(
            x, y,
            marker="o",
            linewidth=2.0,
            markersize=1.0,
            color=color,
        )
        _shade_std(ax, x, y, data, color)

    _setup_x_axis(ax, all_j, position_map)
    ax.set_ylabel(r"$\langle O\rangle_t$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    p0 = float(permult_to_p0(permult_val))
    ax.set_title(rf"$p_0={p0:.3f}$")

    scalar_map = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_map.set_array([])
    colorbar = fig.colorbar(
        scalar_map, ax=ax, fraction=0.046, pad=0.04
    )
    colorbar.set_label(r"$\sqrt{N}$", rotation=90)

    fig.tight_layout()
    filename = (
        f"order_vs_J__permult{permult_val:g}"
        f"__mode{MODE.lower()}.png"
    )
    outpath = outdir / filename
    fig.savefig(outpath, dpi=OUTPUT_DPI)
    plt.close(fig)
    return outpath


def _plot_permults_by_size(
    agg: pd.DataFrame,
    cc_value: float,
    outdir: Path,
) -> Path:
    sub = agg[
        np.isclose(agg["cc"].astype(float), float(cc_value))
    ].copy()

    if sub.empty:
        available = ", ".join(
            f"{value:g}" for value in sorted(agg["cc"].dropna().unique())
        )
        raise ValueError(
            f"No data for N={cc_value}. Available values: {available}"
        )

    if PERMULT_OVERLAY_LIST is not None:
        keep = np.array(PERMULT_OVERLAY_LIST, dtype=float)
        sub = sub[
            sub["permult"].apply(
                lambda value: np.any(np.isclose(float(value), keep))
            )
        ]
        if sub.empty:
            raise ValueError(
                "No data remain after applying PERMULT_OVERLAY_LIST."
            )

    permults = sorted(sub["permult"].unique())

    # p0 = 2 * permult / sqrt(A0) sets both curve colours and colourbar values.
    p0_values = permult_to_p0(permults)
    norm = mpl.colors.Normalize(
        vmin=float(p0_values.min()),
        vmax=float(p0_values.max()),
    )
    cmap = _truncated_cmap(
        PERMULT_CMAP_NAME,
        PERMULT_CMAP_MIN,
        PERMULT_CMAP_MAX,
    )

    all_j = sub["J"].to_numpy(dtype=float)
    position_map = (
        _equal_point_map(all_j)
        if X_AXIS_MODE.lower().strip() == "equal_points"
        else None
    )

    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    for permult in permults:
        data = sub[sub["permult"] == permult].sort_values("J")
        j = data["J"].to_numpy(dtype=float)
        x = (
            _nearest_equal_positions(j, position_map)
            if position_map is not None
            else j
        )
        y = data["Obar"].to_numpy(dtype=float)

        p0 = float(permult_to_p0(permult))
        color = cmap(norm(p0))

        ax.plot(
            x, y,
            marker="o",
            linewidth=2.0,
            markersize=1.0,
            color=color,
        )
        _shade_std(ax, x, y, data, color)

    _setup_x_axis(ax, all_j, position_map)
    ax.set_ylabel(r"$\langle O\rangle_t$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)


    scalar_map = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_map.set_array([])
    colorbar = fig.colorbar(
        scalar_map, ax=ax, fraction=0.046, pad=0.04
    )
    colorbar.set_label(r"$p_0$", rotation=90)

    fig.tight_layout()
    filename = (
        f"order_vs_J__permults_overlay__N{float(cc_value):g}"
        f"__mode{MODE.lower()}.png"
    )
    outpath = outdir / filename
    fig.savefig(outpath, dpi=OUTPUT_DPI)
    fig.savefig(outpath.with_suffix(".pdf"))
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
    if _csv_has_header(csv_path, cols):
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None, names=cols)

    df = _compute_J(df)
    filtered, chosen = _apply_filters(df, FILTERS)

    if filtered.empty:
        raise SystemExit("No rows remain after filtering.")

    agg = _aggregate(filtered)
    if agg.empty:
        raise SystemExit("No valid data remain after aggregation.")

    outdir = here / "order_vs_J_out"
    outdir.mkdir(parents=True, exist_ok=True)
    written = []

    if PLOT_MODE.lower() == "sizes_by_perm":
        permults = (
            sorted(agg["permult"].unique())
            if PERMULT_LIST is None
            else PERMULT_LIST
        )
        for permult in permults:
            try:
                written.append(
                    _plot_sizes_by_perm_one(
                        agg, float(permult), outdir
                    )
                )
            except Exception as exc:
                print(f"[WARN] Skipping permult={permult}: {exc}")

    elif PLOT_MODE.lower() == "permults_by_size":
        if CC_VALUE is None:
            raise SystemExit(
                "Set CC_VALUE for PLOT_MODE='permults_by_size'."
            )
        written.append(
            _plot_permults_by_size(agg, float(CC_VALUE), outdir)
        )

    else:
        raise SystemExit(
            "PLOT_MODE must be 'sizes_by_perm' or 'permults_by_size'."
        )

    print("[INFO] MODE:", MODE)
    print("[INFO] PLOT_MODE:", PLOT_MODE)
    print("[INFO] Filters used:", chosen)
    print("[INFO] A0:", A0)
    print("[INFO] Wrote:")
    for path in written:
        print(" ", path)


if __name__ == "__main__":
    main()
