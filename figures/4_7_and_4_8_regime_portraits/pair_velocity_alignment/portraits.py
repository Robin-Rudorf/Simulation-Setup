#!/usr/bin/env python3
"""Plot tissue snapshots, centre-of-mass trajectories, order, and T1 activity.

Set run paths, selectors, and the integration time step below.
"""

from pathlib import Path
from collections import defaultdict
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter


# Settings

ROOT_RUNS_DIR = "samos_runs"
OUT_DIR = "regime_portraits"

# Match directory parameters; tau stores J for the pair-alignment runs.
SELECTORS = [
    {
        "label": "liquid_flock",
        "cc": 1000,
        "permult": 3.2,
        "tau": 0.1,
        "rand": 1,
    },
    {
        "label": "solid_flock",
        "cc": 1000,
        "permult": 3.2,
        "tau": 2.0,
        "rand": 2,
    },
]

TIME_PER_STEP = 0.01  # Integration time step in simulation units.

# None selects the last cell snapshot; otherwise select the nearest saved step.
SNAPSHOT_STEP = None

T1_SMOOTH_WINDOW = 5

FIGSIZE = (10.0, 7.0)
DPI = 300

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.labelpad": 5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def get_step(path):
    match = re.search(r"(?:_|^cell)(\d+)\.(?:dat|fc|vtp)$", path.name)
    if match is None:
        raise ValueError(f"No step index in filename: {path.name}")
    return int(match.group(1))


def get_cell_files(run_dir):
    # Use one dump family throughout the time series.
    families = [
        sorted(run_dir.glob(pattern), key=get_step)
        for pattern in [
            "cell_[0-9]*.dat",
            "cell_full_[0-9]*.dat",
            "cell[0-9]*.dat",
        ]
    ]
    return max(families, key=len)


def matches_selector(path, selector):
    for key, value in selector.items():
        if key == "label":
            continue

        match = re.search(
            rf"(?:^|/){re.escape(key)}_([-+0-9.eE]+)(?:/|$)",
            path.as_posix(),
        )

        if match is None:
            return False

        if not np.isclose(
            float(match.group(1)), float(value),
            rtol=0, atol=1e-9,
        ):
            return False

    return True


def find_run(root, selector):
    candidates = []

    for path in [root, *root.rglob("*")]:
        if (
            path.is_dir()
            and matches_selector(path, selector)
            and get_cell_files(path)
        ):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No run found for: {selector}"
        )

    if len(candidates) > 1:
        raise ValueError(
            "Multiple runs matched. Refine SELECTORS:\n"
            + "\n".join(str(path) for path in candidates)
        )

    return candidates[0]


def read_cells(path):
    with path.open(encoding="utf-8", errors="ignore") as handle:
        first = handle.readline().strip()

    if first.lower().startswith("keys:"):
        columns = first.split(":", 1)[1].split()
        df = pd.read_csv(
            path,
            sep=r"\s+",
            names=columns,
            skiprows=1,
            comment="#",
        )
    else:
        df = pd.read_csv(path, sep=r"\s+", comment="#")

    df.columns = [str(column).lower() for column in df.columns]

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def internal_cells(df):
    if "boundary" in df.columns:
        df = df.loc[df["boundary"] == 0].copy()
    else:
        df = df.copy()

    if "id" in df.columns:
        df = df.sort_values("id")

    return df.reset_index(drop=True)


def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-12)


def compute_order(df):
    # Use all rows of the cell dump for O(t).
    required = ["x", "y", "vx", "vy"]

    if any(column not in df.columns for column in required):
        raise ValueError("O(t) requires x, y, vx, and vy.")

    values = df[required].to_numpy(dtype=float)

    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Invalid position or velocity data.")

    positions = values[:, :2]
    velocities = values[:, 2:]

    # Use laboratory-frame velocities.
    vhat = normalize(velocities)

    P = np.linalg.norm(vhat.mean(axis=0))

    centre = positions.mean(axis=0, keepdims=True)
    rhat = normalize(positions - centre)

    R = np.mean(
        rhat[:, 0] * vhat[:, 1]
        - rhat[:, 1] * vhat[:, 0]
    )

    delta = np.mean(np.sum(rhat * vhat, axis=1))

    return float(np.sqrt(P**2 + R**2 + delta**2))


def compute_time_series(cell_files):
    rows = []

    for path in cell_files:
        df = read_cells(path)
        cells = internal_cells(df)

        if cells.empty:
            raise ValueError(f"No tissue cells in {path}")

        rows.append({
            "step": get_step(path),
            "t": get_step(path) * TIME_PER_STEP,
            "O": compute_order(df),
            "x_com": cells["x"].mean(),
            "y_com": cells["y"].mean(),
            "N": len(cells),
        })

    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def read_neighbor_pairs(path):
    edge_to_faces = defaultdict(set)

    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()

        if len(parts) != 4:
            continue

        try:
            _, face, vertex1, vertex2 = map(int, parts)
        except ValueError:
            continue

        edge = tuple(sorted((vertex1, vertex2)))
        edge_to_faces[edge].add(face)

    # Count each neighbour pair once.
    return {
        tuple(sorted(faces))
        for faces in edge_to_faces.values()
        if len(faces) == 2
    }


def compute_t1(face_files, time_series):
    face_map = {get_step(path): path for path in face_files}
    n_map = dict(zip(time_series["step"], time_series["N"]))
    steps = sorted(set(face_map) & set(n_map))

    if len(steps) < 2:
        return pd.DataFrame(columns=["t", "rate", "smoothed"])

    previous_step = steps[0]
    previous_pairs = read_neighbor_pairs(face_map[previous_step])
    rows = []

    for step in steps[1:]:
        pairs = read_neighbor_pairs(face_map[step])

        gained = pairs - previous_pairs
        lost = previous_pairs - pairs

        # One ideal T1 event removes one pair and creates another.
        events = (len(gained) + len(lost)) / 2.0

        dt = (step - previous_step) * TIME_PER_STEP
        n_cells = n_map[step]

        rate = events / (n_cells * dt)

        rows.append((step * TIME_PER_STEP, rate))

        previous_step = step
        previous_pairs = pairs

    result = pd.DataFrame(rows, columns=["t", "rate"])
    result["smoothed"] = (
        result["rate"]
        .rolling(
            window=T1_SMOOTH_WINDOW,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    return result


def read_polygons(path):
    piece = ET.parse(path).getroot().find(".//Piece")

    if piece is None:
        raise ValueError(f"No VTK Piece in {path}")

    def read_array(element, dtype):
        if element is None:
            raise ValueError(f"Missing DataArray in {path}")

        if element.get("format", "ascii").lower() != "ascii":
            raise ValueError("This script requires ASCII VTP files.")

        return np.asarray(
            (element.text or "").split(),
            dtype=dtype,
        )

    points = read_array(
        piece.find("./Points/DataArray"), float
    ).reshape(-1, 3)[:, :2]

    connectivity = read_array(
        piece.find("./Polys/DataArray[@Name='connectivity']"),
        int,
    )
    offsets = read_array(
        piece.find("./Polys/DataArray[@Name='offsets']"),
        int,
    )

    polygons = []
    start = 0

    for stop in offsets:
        polygons.append(points[connectivity[start:stop]])
        start = int(stop)

    return polygons


def make_snapshot(cell_file, dual_file):
    cells = internal_cells(read_cells(cell_file))
    polygons = read_polygons(dual_file)

    if "nx" not in cells.columns or "ny" not in cells.columns:
        raise ValueError("Cell colours require nx and ny.")

    if len(cells) == 0 or len(polygons) == 0:
        raise ValueError("No cells or polygons available for the snapshot.")

    if len(cells) != len(polygons):
        raise ValueError(
            f"Cell/polygon count mismatch: {len(cells)} cells, "
            f"{len(polygons)} polygons. Check the matching dump files."
        )

    # Assumes polygons follow the order of cells sorted by ID.
    angles = np.arctan2(
        cells["ny"].to_numpy(dtype=float),
        cells["nx"].to_numpy(dtype=float),
    )

    mesh = PolyCollection(
        polygons,
        array=angles,
        cmap="hsv",
        norm=Normalize(-np.pi, np.pi),
        edgecolors="black",
        linewidths=0.16,
    )

    return mesh, np.vstack(polygons)


def square_limits(ax, xy, padding=0.05):
    xy = np.asarray(xy, dtype=float)

    minimum = xy.min(axis=0)
    maximum = xy.max(axis=0)
    centre = (minimum + maximum) / 2.0

    half = np.max(maximum - minimum) / 2.0
    if half <= 0:
        half = 1.0

    half *= 1.0 + padding

    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect("equal", adjustable="box")


def format_time_axis(ax, times):
    times = np.asarray(times, dtype=float)

    if len(times):
        minimum = float(times.min())
        maximum = float(times.max())

        padding = (
            0.02 * (maximum - minimum)
            if maximum > minimum
            else 0.5
        )
        ax.set_xlim(minimum - padding, maximum + padding)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlabel("time")


def make_portrait(run_dir, label, out_dir):
    cell_files = get_cell_files(run_dir)

    dual_files = sorted(
        run_dir.glob("cell_dual*.vtp"),
        key=get_step,
    )
    face_files = sorted(
        run_dir.glob("faces*.fc"),
        key=get_step,
    )

    if not cell_files:
        raise FileNotFoundError(f"No cell files in {run_dir}")
    if not dual_files:
        raise FileNotFoundError(f"No mesh files in {run_dir}")

    series = compute_time_series(cell_files)
    t1 = compute_t1(face_files, series)

    if SNAPSHOT_STEP is None:
        snapshot_cell = cell_files[-1]
    else:
        snapshot_cell = min(
            cell_files,
            key=lambda path: abs(get_step(path) - SNAPSHOT_STEP),
        )

    snapshot_step = get_step(snapshot_cell)
    snapshot_dual = min(
        dual_files,
        key=lambda path: abs(get_step(path) - snapshot_step),
    )

    if get_step(snapshot_dual) != snapshot_step:
        raise ValueError(
            f"No mesh matches cell snapshot step {snapshot_step}. "
            "Select a step with both cell and mesh data."
        )

    fig, axes = plt.subplots(
        2, 2,
        figsize=FIGSIZE,
        dpi=DPI,
        constrained_layout=True,
    )

    ax1, ax2, ax3, ax4 = axes.ravel()

    # Snapshot coloured by polarity angle.

    mesh, xy = make_snapshot(snapshot_cell, snapshot_dual)
    ax1.add_collection(mesh)
    square_limits(ax1, xy, padding=0.04)
    ax1.set_xticks([])
    ax1.set_yticks([])

    angle_bar = fig.colorbar(
        mesh,
        ax=ax1,
        fraction=0.046,
        pad=0.03,
    )
    angle_bar.set_ticks([
        -np.pi, -np.pi / 2, 0, np.pi / 2, np.pi,
    ])
    angle_bar.set_ticklabels([
        r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$",
    ])
    angle_bar.set_label("polarity angle")

    # Centre-of-mass displacement.

    xy_com = series[["x_com", "y_com"]].to_numpy(dtype=float)
    xy_com -= xy_com[0]

    times = series["t"].to_numpy(dtype=float)

    if len(times) >= 2:
        segments = np.stack(
            (xy_com[:-1], xy_com[1:]),
            axis=1,
        )

        trajectory = LineCollection(
            segments,
            cmap="viridis",
            norm=Normalize(times.min(), times.max()),
            linewidths=2.8,
        )
        trajectory.set_array(times[:-1])
        ax2.add_collection(trajectory)

        fig.colorbar(
            trajectory,
            ax=ax2,
            fraction=0.046,
            pad=0.03,
            label="time",
        )
    else:
        ax2.plot(xy_com[:, 0], xy_com[:, 1])

    ax2.scatter(
        xy_com[[0, -1], 0],
        xy_com[[0, -1], 1],
        color="black",
        s=22,
        zorder=3,
    )

    square_limits(ax2, xy_com, padding=0.10)
    ax2.set_xlabel(r"$\Delta x_{\mathrm{COM}}$")
    ax2.set_ylabel(r"$\Delta y_{\mathrm{COM}}$")
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # Collective order.

    ax3.plot(
        series["t"],
        series["O"],
        color="black",
        linewidth=2.0,
    )
    ax3.set_ylabel(r"$O(t)$")

    # Do not clip order values above one.
    upper = max(1.0, float(series["O"].max()) * 1.03)
    ax3.set_ylim(0, upper)
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=5))
    format_time_axis(ax3, times)
    ax3.grid(True, alpha=0.25)

    # Raw and smoothed T1 rate.

    if not t1.empty:
        ax4.plot(
            t1["t"],
            t1["rate"],
            color="crimson",
            alpha=0.30,
            linewidth=1.0,
            label="raw",
        )
        ax4.plot(
            t1["t"],
            t1["smoothed"],
            color="crimson",
            linewidth=2.0,
            label="smoothed",
        )
        ax4.legend(frameon=False, loc="best")
    else:
        ax4.text(
            0.5, 0.5,
            "No usable T1 data",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )

    ax4.set_ylabel(r"$r_{\mathrm{T1}}$")
    ax4.set_ylim(bottom=0)
    format_time_axis(ax4, times)
    ax4.grid(True, alpha=0.25)

    # Scientific notation affects only the axis labels.
    t1_formatter = ScalarFormatter(
        useMathText=True,
        useOffset=False,
    )
    t1_formatter.set_scientific(True)
    t1_formatter.set_powerlimits((0, 0))

    ax4.yaxis.set_major_formatter(t1_formatter)
    ax4.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax4.yaxis.get_offset_text().set_fontsize(12)

    # Save both formats.

    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"{label}.png"
    pdf_path = out_dir / f"{label}.pdf"

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] {png_path}")
    print(f"[OK] {pdf_path}")


def main():
    root = Path(ROOT_RUNS_DIR).expanduser().resolve()
    out_dir = Path(OUT_DIR).expanduser().resolve()

    if not root.is_dir():
        raise SystemExit(f"Directory not found: {root}")

    for selector in SELECTORS:
        label = selector["label"]

        try:
            run_dir = find_run(root, selector)
            print(f"[INFO] {label}: {run_dir}")
            make_portrait(run_dir, label, out_dir)

        except Exception as error:
            print(f"[ERROR] {label}: {error}")


if __name__ == "__main__":
    main()
