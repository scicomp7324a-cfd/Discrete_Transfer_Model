
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------------------------------------------------------
# Plot styling
# -----------------------------------------------------------------------------

def set_report_style() -> None:
    """Set a clear, report-friendly Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.figsize": (9.5, 6.2),
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "lines.linewidth": 2.8,
            "lines.markersize": 8,
            "axes.grid": True,
            "grid.alpha": 0.32,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "0.8",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    """Save each figure as both PNG and PDF."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / f"{stem}.png"
    pdf = fig_dir / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


# -----------------------------------------------------------------------------
# Robust CSV reading helpers
# -----------------------------------------------------------------------------

def read_csv_required(path: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    """Read a CSV and fail with a useful message if required columns are absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find required CSV file:\n  {path}\n"
            "Check --output-root and whether the DTM diagnostics have finished writing."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV file {path} is missing required columns: {missing}\n"
            f"Available columns are: {list(df.columns)}"
        )

    return df


def case_csv(output_root: Path, case_name: str) -> Path:
    return output_root / case_name / "heat_flux_diagnostics.csv"


# -----------------------------------------------------------------------------
# PLOT 1: heatFluxVariationTest
# -----------------------------------------------------------------------------

def plot_heat_flux_variation(output_root: Path, fig_dir: Path) -> None:
    """
    Plot heat flux against east wall temperature for heatFluxVariationTest().

    main.cu output case:
        Output/heat_flux_test/heat_flux_diagnostics.csv
    """
    csv_path = case_csv(output_root, "heat_flux_test")
    df = read_csv_required(
        csv_path,
        [
            "east_wall_temperature",
            "heat_flux_A",
            "heat_flux_analytical_A",
            "west_wall_temperature",
        ],
    )

    df = df.sort_values("east_wall_temperature")
    tw = df["west_wall_temperature"].iloc[0]

    fig, ax = plt.subplots()
    ax.plot(
        df["east_wall_temperature"],
        df["heat_flux_A"],
        marker="None",
        linestyle = "--",
        label="DTM simulated heat flux on west plate",
    )
    ax.plot(
        df["east_wall_temperature"],
        df["heat_flux_analytical_A"],
        marker="s",
        linestyle="None",
        label="Analytical heat flux on west plate",
    )

    ax.set_title(f"DTM heat-flux variation with east-wall temperature\nWest wall fixed at {tw:g} K")
    ax.set_xlabel("East wall temperature, $T_E$ [K]")
    ax.set_ylabel("Net heat flux leaving west plate, $q_W$ [W m$^{-2}$]")
    ax.legend(loc="best")

    save_figure(fig, fig_dir, "dtm_01_heat_flux_vs_east_wall_temperature")


# -----------------------------------------------------------------------------
# PLOT 2 and 3: convergenceAndTriangleRuntimeTest
# -----------------------------------------------------------------------------

def plot_heat_flux_convergence(output_root: Path, fig_dir: Path) -> None:
    """
    Plot heat flux against number of triangles per surface for
    convergenceAndTriangleRuntimeTest().
    """
    csv_path = case_csv(output_root, "convergence_test_triangle_runtime_scaling")
    df = read_csv_required(
        csv_path,
        ["triangles_per_surface", "heat_flux_A", "heat_flux_B"],
    )

    df["triangles_per_surface"] = pd.to_numeric(df["triangles_per_surface"])
    df = df.sort_values("triangles_per_surface")

    final_a = df["heat_flux_A"].iloc[-1]
    #final_b = df["heat_flux_B"].iloc[-1]

    fig, ax = plt.subplots()
    ax.plot(
        df["triangles_per_surface"],
        df["heat_flux_A"],
        marker="s",
        markerfacecolor = "tab:red",
        color ="tab:green",
        label="West plate heat flux",
    )
    #ax.plot(
        #df["triangles_per_surface"],
        #df["heat_flux_B"],
        #marker="s",
        #linestyle="--",
        #label="East plate heat flux",
    #)
    ax.axhline(final_a, linewidth=1.8, alpha=0.75, color = "tab:red", label=f"Final west-plate value: {final_a:.3g} W m$^{{-2}}$")
    #ax.axhline(final_b, linewidth=1.8, alpha=0.75, linestyle=":", label=f"Final east-plate value: {final_b:.3g} W m$^{{-2}}$")

    ax.set_title("DTM heat-flux convergence with surface discretisation")
    ax.set_xlabel("Number of triangles per surface")
    ax.set_ylabel("Net heat flux leaving plate [W m$^{-2}$]")
    ax.set_xscale("log")
    ax.legend(loc="best")

    save_figure(fig, fig_dir, "dtm_02_heat_flux_convergence_vs_triangles")


def plot_triangle_runtime_scaling(output_root: Path, fig_dir: Path) -> None:
    """
    Plot runtime against number of triangles per surface for
    convergenceAndTriangleRuntimeTest().
    """
    csv_path = case_csv(output_root, "convergence_test_triangle_runtime_scaling")
    df = read_csv_required(csv_path, ["triangles_per_surface", "runtime"])

    df["triangles_per_surface"] = pd.to_numeric(df["triangles_per_surface"])
    df = df.sort_values("triangles_per_surface")

    fig, ax = plt.subplots()
    ax.plot(
        df["triangles_per_surface"],
        df["runtime"],
        marker="D",
        markersize = 10,
        markerfacecolor = "tab:red",
        markeredgecolor = "tab:green",
        linestyle="-",
        label="DTM total simulation runtime",
    )

    offsets = [(0, 15), (0, -16), (12, 10), (-12, 10)]

    for i, (x, y) in enumerate(zip(df["triangles_per_surface"], df["runtime"])):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            f"({x:.0f},{y:.2f})",
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            fontsize=9,
        )

    ax.set_title("DTM runtime scaling with surface discretisation")
    ax.set_xlabel("Number of triangles per surface")
    ax.set_ylabel("Runtime [s]")
    ax.set_xscale("log")
    ax.legend(loc="best")

    save_figure(fig, fig_dir, "dtm_03_runtime_vs_triangles")


# -----------------------------------------------------------------------------
# PLOT 4: rayRuntimeTest
# -----------------------------------------------------------------------------

def plot_ray_runtime_scaling(output_root: Path, fig_dir: Path) -> None:
    """
    Plot runtime against emitted rays per triangle for rayRuntimeTest().

    In main.cu, ray_count = thetanum * psinum, with psinum fixed at 100.
    """
    csv_path = case_csv(output_root, "rays_runtime_scaling")
    df = read_csv_required(csv_path, ["ray_count", "runtime"])

    df["ray_count"] = pd.to_numeric(df["ray_count"])
    df = df.sort_values("ray_count")

    fig, ax = plt.subplots()
    ax.plot(
        df["ray_count"],
        df["runtime"],
        marker="o",
        markersize = 10,
        markerfacecolor = "tab:red",
        markeredgecolor = "tab:green",
        label="DTM total simulation runtime",
    )

    offsets = [(0, 15), (10, -20), (12, 10), (-16, 10)]

    for i, (x, y) in enumerate(zip(df["ray_count"], df["runtime"])):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            f"({x:.0f},{y:.2f})",
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            fontsize=9,
        )

    ax.set_title("DTM runtime scaling with rays emitted per triangle")
    ax.set_xlabel("Number of rays emitted per triangle")
    ax.set_ylabel("Runtime [s]")
    ax.legend(loc="best")

    save_figure(fig, fig_dir, "dtm_04_runtime_vs_rays_per_triangle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate report-quality DTM plots from heat_flux_diagnostics.csv files."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Output"),
        help="Root directory containing DTM output case folders. Default: Output",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Directory where figures are saved. Default: <output-root>/Figures_DTM",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=["heatflux", "convergence", "tri-runtime", "ray-runtime"],
        default=None,
        help="Optionally generate only selected plot groups.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    fig_dir = args.fig_dir if args.fig_dir is not None else output_root / "Figures_DTM"

    set_report_style()

    selected = set(args.only or ["heatflux", "convergence", "tri-runtime", "ray-runtime"])

    if "heatflux" in selected:
        plot_heat_flux_variation(output_root, fig_dir)

    if "convergence" in selected:
        plot_heat_flux_convergence(output_root, fig_dir)

    if "tri-runtime" in selected:
        plot_triangle_runtime_scaling(output_root, fig_dir)

    if "ray-runtime" in selected:
        plot_ray_runtime_scaling(output_root, fig_dir)

    print(f"\nAll requested DTM figures saved in: {fig_dir}")


if __name__ == "__main__":
    main()
