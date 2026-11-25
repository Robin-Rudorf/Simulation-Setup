#!/usr/bin/env python3
"""Plot ordering survival curves and median observed switching times.

Read time/order CSVs from time_series/ in the current working directory.
CSV times are integration steps, converted to simulation time with dt = 0.01.
Times are measured from simulation start. Medians include switched runs only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Settings
TIME_SERIES_DIRNAME = "time_series"   # set "" to search current folder only
GLOB_PATTERN = "*.csv"

# Convert integration steps to simulation time. Use 1.0 if CSV times are already scaled.
TIME_SCALE = 0.01
TIME_UNIT = "simulation time units"

# A switch requires HOLD_POINTS consecutive samples at or above O_HIGH.
O_HIGH = 0.80
HOLD_POINTS = 10

# Observation starts at this absolute time after applying TIME_SCALE.
TRANSIENT_CUT_TIME = 0.0

# Plot settings
DPI = 300
FONTSIZE = 13
AXIS_LABEL_FONTSIZE = 15
MEDIAN_LABEL_FONTSIZE = 18
MEDIAN_TICK_FONTSIZE = 14
LINEWIDTH = 2.2
MAX_SURVIVAL_CURVES = 8
CMAP_NAME = "viridis"


@dataclass
class RunRecord:
    path: Path
    cc: Optional[int]
    rand: Optional[int]
    tau: Optional[float]
    permult: Optional[float]
    J: Optional[float]
    T_end: float
    O_at_t0: float
    ordered_at_t0: bool
    t_switch: Optional[float]
    switched: bool
    censored: bool


def parse_params_from_filename(name: str) -> Dict[str, Optional[float]]:
    tokens = name.replace(".csv", "").split(";")
    out: Dict[str, Optional[float]] = {"cc": None, "rand": None, "tau": None, "permult": None}
    for tok in tokens:
        if tok.startswith("cc_"):
            out["cc"] = int(tok.split("_", 1)[1])
        elif tok.startswith("rand_"):
            out["rand"] = int(tok.split("_", 1)[1])
        elif tok.startswith("tau_"):
            out["tau"] = float(tok.split("_", 1)[1])
        elif tok.startswith("permult_"):
            out["permult"] = float(tok.split("_", 1)[1])
    return out


def read_timeseries_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"{path}: expected >=2 columns (time, O)")
    t_raw = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    O = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(t_raw) & np.isfinite(O)
    t_raw, O = t_raw[mask], O[mask]
    if len(t_raw) < 5:
        raise ValueError(f"{path}: too few valid samples")
    # Keep absolute times; do not subtract the observation-window start.
    t = t_raw * float(TIME_SCALE)
    idx = np.argsort(t)
    return t[idx], O[idx]


def first_sustained_crossing(t: np.ndarray, O: np.ndarray, thresh: float, hold: int) -> Optional[float]:
    if len(O) < hold:
        return None
    above = (O >= thresh).astype(np.int32)
    conv = np.convolve(above, np.ones(hold, dtype=np.int32), mode="valid")
    hits = np.where(conv == hold)[0]
    if len(hits) == 0:
        return None
    return float(t[int(hits[0])])


def kaplan_meier(
    times: np.ndarray, censored: np.ndarray, end_time: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the conditional survival curve, including observation endpoints."""
    order = np.argsort(times)
    times = times[order]
    censored = censored[order]

    at_risk = len(times)
    S = 1.0
    t_out, S_out = [float(TRANSIENT_CUT_TIME)], [1.0]

    for time in np.unique(times):
        mask = times == time
        events = int((~censored[mask]).sum())
        withdrawals = int(censored[mask].sum())
        # Runs censored at this time remain at risk for simultaneous events.
        if events:
            S *= 1.0 - events / at_risk
            t_out.append(float(time))
            S_out.append(float(S))
        at_risk -= events + withdrawals

    end = float(np.max(times)) if end_time is None else float(end_time)
    if t_out[-1] < end:
        t_out.append(end)
        S_out.append(float(S))
    return np.array(t_out), np.array(S_out)


def downsample(vals: List[float], k: int) -> List[float]:
    if len(vals) <= k:
        return vals
    idx = np.linspace(0, len(vals) - 1, k).round().astype(int)
    return [vals[i] for i in idx]


def compute_run_record(path: Path) -> RunRecord:
    params = parse_params_from_filename(path.name)
    t, O = read_timeseries_csv(path)
    T_end = float(np.max(t))

    tau = params["tau"]
    if (
        tau is None or not np.isfinite(tau) or tau <= 0
        or params["cc"] is None or params["cc"] <= 0
        or params["permult"] is None or not np.isfinite(params["permult"])
    ):
        raise ValueError("Filename must specify valid cc, permult, and positive tau.")
    J = 1.0 / tau

    observed = t >= TRANSIENT_CUT_TIME
    if observed.sum() < HOLD_POINTS:
        raise ValueError("Too few samples in the observation window for initial classification.")
    t_obs, O_obs = t[observed], O[observed]
    O_t0 = float(np.mean(O_obs[:HOLD_POINTS]))
    ordered_t0 = bool(O_t0 >= O_HIGH)

    t_sw = None
    if not ordered_t0:
        t_sw = first_sustained_crossing(t_obs, O_obs, O_HIGH, HOLD_POINTS)
    switched = (not ordered_t0) and (t_sw is not None)
    censored = (not ordered_t0) and (t_sw is None)

    return RunRecord(
        path=path,
        cc=int(params["cc"]) if params["cc"] is not None else None,
        rand=int(params["rand"]) if params["rand"] is not None else None,
        tau=float(tau) if tau is not None else None,
        permult=float(params["permult"]) if params["permult"] is not None else None,
        J=float(J) if J is not None else None,
        T_end=T_end,
        O_at_t0=O_t0,
        ordered_at_t0=ordered_t0,
        t_switch=t_sw,
        switched=switched,
        censored=censored,
    )


def main():
    if not np.isfinite(TIME_SCALE) or TIME_SCALE <= 0:
        raise SystemExit("TIME_SCALE must be positive and finite.")
    if not isinstance(HOLD_POINTS, int) or HOLD_POINTS < 1:
        raise SystemExit("HOLD_POINTS must be a positive integer.")
    if not np.isfinite(TRANSIENT_CUT_TIME) or TRANSIENT_CUT_TIME < 0:
        raise SystemExit("TRANSIENT_CUT_TIME must be non-negative and finite.")
    if MAX_SURVIVAL_CURVES < 1:
        raise SystemExit("MAX_SURVIVAL_CURVES must be positive.")

    base = Path(".").resolve()
    ts_dir = base / TIME_SERIES_DIRNAME if TIME_SERIES_DIRNAME else base
    if not ts_dir.exists():
        raise SystemExit(f"Could not find time-series directory: {ts_dir}")

    files = sorted([p for p in ts_dir.glob(GLOB_PATTERN) if p.is_file()])
    files = [p for p in files if ("cc_" in p.name and "tau_" in p.name and "permult_" in p.name)]
    if not files:
        raise SystemExit(f"No time-series files found in {ts_dir}.")

    out_dir = base / "survival_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": FONTSIZE})

    recs: List[RunRecord] = []
    for fp in files:
        try:
            recs.append(compute_run_record(fp))
        except Exception as e:
            print(f"[WARN] Skipping {fp.name}: {e}")
    if not recs:
        raise SystemExit("No valid runs parsed.")

    df = pd.DataFrame([r.__dict__ for r in recs])
    df.to_csv(out_dir / "runs_parsed.csv", index=False)

    # Group repeated runs at the same parameter point.
    agg = df.groupby(["permult", "cc", "tau", "J"], dropna=False).agg(
        n_runs=("tau", "size"),
        frac_ordered_at_t0=("ordered_at_t0", "mean"),
        frac_switched=("switched", "mean"),
        frac_censored=("censored", "mean"),
    ).reset_index().sort_values(["permult", "cc", "J"])

    # Empirical median among observed switches, not the Kaplan-Meier median.
    df_sw = df[df["switched"] & df["t_switch"].notna()].copy()
    med = df_sw.groupby(["permult", "cc", "tau", "J"], dropna=False)["t_switch"].median().rename("median_t_switch")
    agg = agg.merge(med.reset_index(), on=["permult", "cc", "tau", "J"], how="left")

    agg.to_csv(out_dir / "switching_summary.csv", index=False)

    # One pair of plots for each shape parameter and cell count.
    for (perm, cc), agg_g in agg.groupby(["permult", "cc"], dropna=False):
        df_g = df[(df["permult"] == perm) & (df["cc"] == cc)].copy()
        tag = f"permult{perm:g}__cc{int(cc)}" if pd.notna(cc) else f"permult{perm:g}"
        g = agg_g.dropna(subset=["J"]).sort_values("J")

        # Median absolute switching time among observed switches.
        fig = plt.figure(figsize=(4.4, 3.2))
        ax = fig.add_subplot(111)
        g2 = g.dropna(subset=["median_t_switch"])
        ax.plot(
            g2["J"], g2["median_t_switch"], marker="o", linewidth=LINEWIDTH,
            color="black", markerfacecolor="black", markeredgecolor="black",
        )
        ax.set_xlabel(r"$J$", fontsize=MEDIAN_LABEL_FONTSIZE, color="black")
        ax.set_ylabel(r"$t_{\mathrm{switch}}$", fontsize=MEDIAN_LABEL_FONTSIZE, color="black")
        ax.tick_params(axis="both", labelsize=MEDIAN_TICK_FONTSIZE, colors="black")
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
        for spine in ax.spines.values():
            spine.set_color("black")
        if g2.empty:
            ax.text(0.5, 0.5, "No observed switches", ha="center", va="center", color="black", transform=ax.transAxes)
            ax.set_ylim(TRANSIENT_CUT_TIME, float(df_g["T_end"].max()))
        else:
            ax.set_ylim(TRANSIENT_CUT_TIME, None)
        ax.grid(True, color="black", alpha=0.15)
        fig.tight_layout()
        fig.savefig(out_dir / f"median_tswitch_vs_J__{tag}.png", dpi=DPI, bbox_inches="tight")
        fig.savefig(out_dir / f"median_tswitch_vs_J__{tag}.pdf", bbox_inches="tight")
        plt.close(fig)

        # S(t) = initial unordered fraction * conditional Kaplan-Meier survival.
        levels = g.dropna(subset=["tau","J"]).sort_values("J")[["tau","J"]].to_records(index=False)
        levels = list(levels)
        if len(levels) > MAX_SURVIVAL_CURVES:
            levels = downsample(levels, MAX_SURVIVAL_CURVES)

        cmap = plt.get_cmap(CMAP_NAME)
        J_vals = [float(rec[1]) for rec in levels]
        norm = mpl.colors.Normalize(vmin=min(J_vals), vmax=max(J_vals))

        fig = plt.figure(figsize=(8.2, 5.2))
        ax = fig.add_subplot(111)
        T_end_group = float(np.nanmax(df_g["T_end"].to_numpy(dtype=float)))
        ax.set_xlim(TRANSIENT_CUT_TIME, T_end_group)

        for tau_val, J_val in levels:
            sub = df_g[df_g["tau"] == float(tau_val)].copy()
            if sub.empty:
                continue
            at_risk = sub[~sub["ordered_at_t0"]].copy()
            f0 = float(len(at_risk)/len(sub))
            if f0 == 0.0:
                ax.step(
                    [TRANSIENT_CUT_TIME, float(sub["T_end"].max())], [0.0, 0.0],
                    where="post", linewidth=LINEWIDTH, color=cmap(norm(J_val)),
                )
                continue
            t_event = at_risk["t_switch"].to_numpy(dtype=float)
            cens = ~np.isfinite(t_event)
            times = np.where(np.isfinite(t_event), t_event, at_risk["T_end"].to_numpy(dtype=float))
            t_km, S_cond = kaplan_meier(
                times, cens, end_time=float(at_risk["T_end"].max()),
            )
            S_uncond = f0 * S_cond
            ax.step(t_km, S_uncond, where="post", linewidth=LINEWIDTH, color=cmap(norm(J_val)))

        ax.set_xlabel("time", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(r"$S(t)$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=FONTSIZE)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$J$", rotation=90, fontsize=AXIS_LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=FONTSIZE)
        fig.tight_layout()
        fig.savefig(out_dir / f"survival__{tag}.png", dpi=DPI, bbox_inches="tight")
        fig.savefig(out_dir / f"survival__{tag}.pdf", bbox_inches="tight")
        plt.close(fig)

    print("[OK] Outputs:", out_dir)
    print("[INFO] TIME_SCALE:", TIME_SCALE, "TIME_UNIT:", TIME_UNIT)
    print("[INFO] Observation starts at:", TRANSIENT_CUT_TIME)
    print("[INFO] Medians include observed switches only.")


if __name__ == "__main__":
    main()
