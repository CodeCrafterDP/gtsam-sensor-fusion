#!/usr/bin/env python3

"""
Compare GPS trajectory with GTSAM fused trajectory.

Input:
    results/fused_results.csv
    results/gps_local.csv

Output:
    results/gps_vs_gtsam_trajectory.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

FUSED_FILE = RESULTS_DIR / "fused_results.csv"
GPS_FILE = RESULTS_DIR / "gps_local.csv"

OUTPUT_FILE = RESULTS_DIR / "gps_vs_gtsam_trajectory.png"


def main():

    # ---------------------------------------------------------------
    # Check input files
    # ---------------------------------------------------------------

    if not FUSED_FILE.exists():
        raise FileNotFoundError(
            f"Fused result file not found:\n{FUSED_FILE}"
        )

    if not GPS_FILE.exists():
        raise FileNotFoundError(
            f"GPS result file not found:\n{GPS_FILE}"
        )

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    fused = pd.read_csv(FUSED_FILE)
    gps = pd.read_csv(GPS_FILE)

    # ---------------------------------------------------------------
    # Create trajectory plot
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 8))

    # GPS trajectory
    plt.plot(
        gps["x"],
        gps["y"],
        label="GPS trajectory",
        linewidth=2,
    )

    # GTSAM trajectory
    plt.plot(
        fused["x"],
        fused["y"],
        label="GTSAM fused trajectory",
        linewidth=2,
    )

    # Start point
    plt.scatter(
        gps["x"].iloc[0],
        gps["y"].iloc[0],
        s=80,
        label="Start",
    )

    # GPS end point
    plt.scatter(
        gps["x"].iloc[-1],
        gps["y"].iloc[-1],
        s=80,
        label="GPS End",
    )

    # ---------------------------------------------------------------
    # Labels
    # ---------------------------------------------------------------

    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")

    plt.title(
        "GPS vs GTSAM Fused Trajectory"
    )

    plt.legend()
    plt.grid(True)

    # Equal scale so the trajectory is not distorted.
    plt.axis("equal")

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Save figure
    # ---------------------------------------------------------------

    plt.savefig(
        OUTPUT_FILE,
        dpi=200,
    )

    plt.show()

    print()
    print("Trajectory comparison complete.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()