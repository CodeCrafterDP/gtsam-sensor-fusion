#!/usr/bin/env python3

"""
Offline Smartphone IMU + GPS Sensor Fusion using GTSAM

This script:
1. Reads uncalibrated accelerometer data.
2. Reads uncalibrated gyroscope data.
3. Reads GPS latitude/longitude/altitude data.
4. Converts GPS coordinates to a local metric coordinate system.
5. Reorders Sensor Recorder axes from z,y,x to x,y,z.
6. Feeds IMU and GPS measurements into the GTSAM fusion core.
7. Estimates position, orientation, velocity and sensor biases.
8. Saves the fused results and plots.

Dataset:
    AccelerometerUncalibrated.csv
    GyroscopeUncalibrated.csv
    Location.csv

The GTSAM core is based on the original:
    https://github.com/PaulKemppi/gtsam_fusion

This project uses the core directly for offline CSV processing
instead of the original ROS 1 interface.
"""

from pathlib import Path
import sys
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utm


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Allow Python to find gtsam_fusion_core.py inside scripts/
sys.path.insert(0, str(ROOT / "scripts"))

from gtsam_fusion_core import GtsamFusionCore


# ---------------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------------

ACC_FILE = DATA_DIR / "AccelerometerUncalibrated.csv"
GYR_FILE = DATA_DIR / "GyroscopeUncalibrated.csv"
GPS_FILE = DATA_DIR / "Location.csv"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def check_file(path):
    """Check whether an input file exists."""
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
            f"Please place the required CSV files inside:\n{DATA_DIR}"
        )


def load_sensor_data():
    """Load accelerometer, gyroscope and GPS CSV files."""

    check_file(ACC_FILE)
    check_file(GYR_FILE)
    check_file(GPS_FILE)

    acc = pd.read_csv(ACC_FILE)
    gyr = pd.read_csv(GYR_FILE)
    gps = pd.read_csv(GPS_FILE)

    return acc, gyr, gps


def prepare_imu_data(acc, gyr):
    """
    Prepare accelerometer and gyroscope data.

    Sensor Recorder CSV axis order:
        z, y, x

    GTSAM expects:
        x, y, z
    """

    required_columns = [
        "seconds_elapsed",
        "x",
        "y",
        "z",
    ]

    for column in required_columns:
        if column not in acc.columns:
            raise ValueError(
                f"Accelerometer CSV is missing required column: {column}"
            )

        if column not in gyr.columns:
            raise ValueError(
                f"Gyroscope CSV is missing required column: {column}"
            )

    # Use accelerometer timestamps as the IMU timeline.
    imu = pd.DataFrame()

    imu["t"] = acc["seconds_elapsed"].astype(float)

    # Sensor Recorder stores columns as z,y,x.
    # Convert to standard x,y,z ordering.
    imu["ax"] = acc["x"].astype(float)
    imu["ay"] = acc["y"].astype(float)
    imu["az"] = acc["z"].astype(float)

    imu["gx"] = gyr["x"].astype(float)
    imu["gy"] = gyr["y"].astype(float)
    imu["gz"] = gyr["z"].astype(float)

    # Remove invalid values.
    imu = imu.replace([np.inf, -np.inf], np.nan)
    imu = imu.dropna()

    # Sort chronologically.
    imu = imu.sort_values("t")

    # Remove duplicate timestamps.
    imu = imu.drop_duplicates(subset="t")

    imu = imu.reset_index(drop=True)

    return imu


def gps_to_local_coordinates(gps):
    """
    Convert latitude/longitude GPS data into a local metric frame.

    The first GPS point is used as the origin.

    Returns:
        gps_local:
            DataFrame containing:
                t, x, y, z

        utm_zone_number
        utm_zone_letter
    """

    required_columns = [
        "seconds_elapsed",
        "latitude",
        "longitude",
        "altitude",
    ]

    for column in required_columns:
        if column not in gps.columns:
            raise ValueError(
                f"GPS CSV is missing required column: {column}"
            )

    gps = gps.copy()

    gps["t"] = gps["seconds_elapsed"].astype(float)
    gps["latitude"] = gps["latitude"].astype(float)
    gps["longitude"] = gps["longitude"].astype(float)
    gps["altitude"] = gps["altitude"].astype(float)

    gps = gps.replace([np.inf, -np.inf], np.nan)
    gps = gps.dropna(
        subset=[
            "t",
            "latitude",
            "longitude",
            "altitude",
        ]
    )

    gps = gps.sort_values("t")
    gps = gps.drop_duplicates(subset="t")
    gps = gps.reset_index(drop=True)

    if len(gps) == 0:
        raise ValueError("No valid GPS measurements were found.")

    # Convert first GPS point to UTM.
    first_lat = gps.loc[0, "latitude"]
    first_lon = gps.loc[0, "longitude"]

    first_easting, first_northing, zone_number, zone_letter = (
        utm.from_latlon(first_lat, first_lon)
    )

    first_altitude = gps.loc[0, "altitude"]

    local_x = []
    local_y = []
    local_z = []

    for _, row in gps.iterrows():

        easting, northing, _, _ = utm.from_latlon(
            row["latitude"],
            row["longitude"],
        )

        # Local coordinate system.
        x = easting - first_easting
        y = northing - first_northing
        z = row["altitude"] - first_altitude

        local_x.append(x)
        local_y.append(y)
        local_z.append(z)

    gps_local = pd.DataFrame(
        {
            "t": gps["t"].values,
            "x": local_x,
            "y": local_y,
            "z": local_z,
        }
    )

    return gps_local, zone_number, zone_letter


# ---------------------------------------------------------------------------
# GTSAM parameters
# ---------------------------------------------------------------------------

def create_gtsam_parameters():
    """
    Create the parameters used by the GTSAM fusion core.

    These values follow the parameter configuration used by
    the original gtsam_fusion project, adapted for the installed
    GTSAM 4.2.2 Python API.
    """

    params = {

        # Gravity
        "g": [0.0, 0.0, -9.81],

        # IMU noise
        "sigma_accelerometer": 0.05,
        "sigma_gyroscope": 0.005,
        "sigma_integration": 0.0,

        # Coriolis
        "use_2nd_order_coriolis": False,
        "omega_coriolis": [0.0, 0.0, 0.0],

        # Body-to-sensor transform
        "b2s_pos": [0.0, 0.0, 0.0],

        "b2s_ori": [
            -4.59837371e-02,
            -1.34857154e-03,
            -6.20780829e-05,
            9.98941276e-01,
        ],

        # Initial state
        "init_pos": [0.0, 0.0, 0.0],

        "init_ori": [
            0.0,
            0.0,
            0.0,
            1.0,
        ],

        "init_vel": [
            0.0,
            0.0,
            0.0,
        ],

        # Initial IMU biases
        "init_acc_bias": [
            0.0,
            0.0,
            0.0,
        ],

        "init_gyr_bias": [
            0.0,
            0.0,
            0.0,
        ],

        # Initial uncertainty
        "sigma_init_ori": 2.0 * math.pi,
        "sigma_init_pos": 10.0,
        "sigma_init_vel": 10.0,

        "sigma_acc_init_bias": 1.0,
        "sigma_gyr_init_bias": 0.1,

        # Pose measurement noise
        "sigma_pose_rot": [
            1.0,
            1.0,
            1.0,
        ],

        "sigma_pose_pos": [
            1.0,
            1.0,
            1.0,
        ],

        # GPS measurement noise
        "sigma_gps": [
            1.0,
            1.0,
            1.0,
        ],

        # Bias evolution
        "sigma_acc_bias_evol": [
            1e-3,
            1e-3,
            1e-3,
        ],

        "sigma_gyr_bias_evol": [
            1e-5,
            1e-5,
            1e-5,
        ],

        # Optimization
        "opt_meas_buffer_time": 0.3,
        "relinearize_th": 0.01,
        "relinearize_skip": 1,
        "factorization": "CHOLESKY",
    }

    return params


# ---------------------------------------------------------------------------
# Main fusion process
# ---------------------------------------------------------------------------

def run_fusion(imu, gps_local, params):
    """
    Run GTSAM sensor fusion.

    GPS measurements are inserted before the IMU measurement
    whose timestamp has reached the GPS timestamp.
    """

    fusion = GtsamFusionCore(params)

    gps_i = 0

    results = []

    last_t = None

    for row in imu.itertuples(index=False):

        t = float(row.t)

        # ---------------------------------------------------------------
        # Skip first IMU sample because there is no previous timestamp
        # from which a valid dt can be calculated.
        # ---------------------------------------------------------------

        if last_t is None:
            last_t = t
            continue

        dt = t - last_t

        # Protect GTSAM from duplicate or out-of-order timestamps.
        if dt <= 0:
            continue

        last_t = t

        # ---------------------------------------------------------------
        # Insert GPS measurements that are available up to current IMU time
        # ---------------------------------------------------------------

        while gps_i < len(gps_local):

            gps_time = float(gps_local.iloc[gps_i]["t"])

            if gps_time > t:
                break

            gps_position = np.array(
                [
                    gps_local.iloc[gps_i]["x"],
                    gps_local.iloc[gps_i]["y"],
                    gps_local.iloc[gps_i]["z"],
                ],
                dtype=float,
            )

            fusion.add_gps_measurement(
                gps_time,
                gps_position,
            )

            gps_i += 1

        # ---------------------------------------------------------------
        # IMU measurement
        # ---------------------------------------------------------------

        acceleration = np.array(
            [
                float(row.ax),
                float(row.ay),
                float(row.az),
            ],
            dtype=float,
        )

        angular_velocity = np.array(
            [
                float(row.gx),
                float(row.gy),
                float(row.gz),
            ],
            dtype=float,
        )

        state = fusion.add_imu_measurement(
            t,
            acceleration,
            angular_velocity,
            dt,
        )

        # ---------------------------------------------------------------
        # Save estimated state
        # ---------------------------------------------------------------

        if state is None:
            continue

        position = state[0]
        orientation = state[1]
        velocity = state[2]
        acc_bias = state[3]
        gyr_bias = state[4]

        # Convert quaternion xyzw to Euler angles.
        qx = float(orientation[0])
        qy = float(orientation[1])
        qz = float(orientation[2])
        qw = float(orientation[3])

        # Roll
        sinr_cosp = 2.0 * (
            qw * qx + qy * qz
        )

        cosr_cosp = 1.0 - 2.0 * (
            qx * qx + qy * qy
        )

        roll = math.atan2(
            sinr_cosp,
            cosr_cosp,
        )

        # Pitch
        sinp = 2.0 * (
            qw * qy - qz * qx
        )

        if abs(sinp) >= 1:
            pitch = math.copysign(
                math.pi / 2,
                sinp,
            )
        else:
            pitch = math.asin(sinp)

        # Yaw
        siny_cosp = 2.0 * (
            qw * qz + qx * qy
        )

        cosy_cosp = 1.0 - 2.0 * (
            qy * qy + qz * qz
        )

        yaw = math.atan2(
            siny_cosp,
            cosy_cosp,
        )

        results.append(
            {
                "time": t,

                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),

                "roll_deg": math.degrees(roll),
                "pitch_deg": math.degrees(pitch),
                "yaw_deg": math.degrees(yaw),

                "vx": float(velocity[0]),
                "vy": float(velocity[1]),
                "vz": float(velocity[2]),

                "acc_bias_x": float(acc_bias[0]),
                "acc_bias_y": float(acc_bias[1]),
                "acc_bias_z": float(acc_bias[2]),

                "gyr_bias_x": float(gyr_bias[0]),
                "gyr_bias_y": float(gyr_bias[1]),
                "gyr_bias_z": float(gyr_bias[2]),
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def create_plots(imu, fused, gps_local):
    """Create all result plots."""

    if len(fused) == 0:
        print("No fused results available for plotting.")
        return

    # ---------------------------------------------------------------
    # Accelerometer
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        imu["t"],
        imu["ax"],
        label="Accelerometer X",
    )

    plt.plot(
        imu["t"],
        imu["ay"],
        label="Accelerometer Y",
    )

    plt.plot(
        imu["t"],
        imu["az"],
        label="Accelerometer Z",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Uncalibrated Accelerometer Data")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "accelerometer.png",
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Gyroscope
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        imu["t"],
        imu["gx"],
        label="Gyroscope X",
    )

    plt.plot(
        imu["t"],
        imu["gy"],
        label="Gyroscope Y",
    )

    plt.plot(
        imu["t"],
        imu["gz"],
        label="Gyroscope Z",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Angular velocity (rad/s)")
    plt.title("Uncalibrated Gyroscope Data")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "gyroscope.png",
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Fused trajectory
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 8))

    plt.plot(
        fused["x"],
        fused["y"],
        linewidth=2,
        label="GTSAM fused trajectory",
    )

    plt.scatter(
        fused["x"].iloc[0],
        fused["y"].iloc[0],
        s=80,
        label="Start",
    )

    plt.scatter(
        fused["x"].iloc[-1],
        fused["y"].iloc[-1],
        s=80,
        label="End",
    )

    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("GTSAM Fused Trajectory")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "trajectory_xy.png",
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Orientation
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        fused["time"],
        fused["roll_deg"],
        label="Roll",
    )

    plt.plot(
        fused["time"],
        fused["pitch_deg"],
        label="Pitch",
    )

    plt.plot(
        fused["time"],
        fused["yaw_deg"],
        label="Yaw",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Angle (degrees)")
    plt.title("Estimated Orientation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "orientation.png",
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Accelerometer bias
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        fused["time"],
        fused["acc_bias_x"],
        label="Bias X",
    )

    plt.plot(
        fused["time"],
        fused["acc_bias_y"],
        label="Bias Y",
    )

    plt.plot(
        fused["time"],
        fused["acc_bias_z"],
        label="Bias Z",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Accelerometer bias")
    plt.title("Estimated Accelerometer Bias")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "acc_bias.png",
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Gyroscope bias
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        fused["time"],
        fused["gyr_bias_x"],
        label="Bias X",
    )

    plt.plot(
        fused["time"],
        fused["gyr_bias_y"],
        label="Bias Y",
    )

    plt.plot(
        fused["time"],
        fused["gyr_bias_z"],
        label="Bias Z",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Gyroscope bias")
    plt.title("Estimated Gyroscope Bias")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "gyr_bias.png",
        dpi=200,
    )

    plt.close()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def create_summary(imu, gps_local, zone_number, zone_letter):
    """Create a text summary of the experiment."""

    imu_duration = (
        imu["t"].iloc[-1] - imu["t"].iloc[0]
    )

    gps_duration = (
        gps_local["t"].iloc[-1] - gps_local["t"].iloc[0]
    )

    imu_rate = (
        (len(imu) - 1) / imu_duration
        if imu_duration > 0
        else 0
    )

    gps_rate = (
        (len(gps_local) - 1) / gps_duration
        if gps_duration > 0
        else 0
    )

    summary = f"""GTSAM Smartphone Sensor Fusion Experiment

IMU samples: {len(imu)}
GPS samples: {len(gps_local)}

Duration (s): {imu_duration:.3f}

Measured IMU rate (Hz): {imu_rate:.3f}
Measured GPS rate (Hz): {gps_rate:.3f}

UTM zone: {zone_number}{zone_letter}

Axes reordered from CSV z,y,x to GTSAM x,y,z.

GPS converted from latitude/longitude to local metric coordinates.

Sensors used:
- Uncalibrated accelerometer
- Uncalibrated gyroscope
- GPS

Fusion method:
- IMU preintegration
- GPS factor
- GTSAM ISAM2 optimization

GTSAM version:
4.2.2

Processing mode:
Offline CSV processing
"""

    with open(
        RESULTS_DIR / "run_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(summary)

    return imu_rate, gps_rate, imu_duration


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print()
    print("=" * 65)
    print("GTSAM SMARTPHONE IMU + GPS SENSOR FUSION")
    print("=" * 65)
    print()

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    print("Loading sensor data...")

    acc, gyr, gps = load_sensor_data()

    print(f"Accelerometer rows : {len(acc)}")
    print(f"Gyroscope rows     : {len(gyr)}")
    print(f"GPS rows            : {len(gps)}")
    print()

    # ---------------------------------------------------------------
    # Prepare IMU
    # ---------------------------------------------------------------

    print("Preparing IMU data...")

    imu = prepare_imu_data(
        acc,
        gyr,
    )

    print(f"Prepared IMU samples: {len(imu)}")
    print()

    # ---------------------------------------------------------------
    # Prepare GPS
    # ---------------------------------------------------------------

    print("Converting GPS latitude/longitude to local coordinates...")

    gps_local, zone_number, zone_letter = (
        gps_to_local_coordinates(gps)
    )

    print(
        f"UTM zone: {zone_number}{zone_letter}"
    )

    print(
        f"Prepared GPS samples: {len(gps_local)}"
    )

    print()

    # ---------------------------------------------------------------
    # Create parameters
    # ---------------------------------------------------------------

    print("Creating GTSAM parameters...")

    params = create_gtsam_parameters()

    print("Parameters ready.")
    print()

    # ---------------------------------------------------------------
    # Run fusion
    # ---------------------------------------------------------------

    print("Running GTSAM sensor fusion...")
    print("This may take some time for the full dataset.")
    print()

    fused = run_fusion(
        imu,
        gps_local,
        params,
    )

    if len(fused) == 0:
        raise RuntimeError(
            "GTSAM did not produce any fused results."
        )

    # ---------------------------------------------------------------
    # Save fused CSV
    # ---------------------------------------------------------------

    fused_file = RESULTS_DIR / "fused_results.csv"

    fused.to_csv(
        fused_file,
        index=False,
    )

    # ---------------------------------------------------------------
    # Save local GPS data
    # ---------------------------------------------------------------

    gps_file = RESULTS_DIR / "gps_local.csv"

    gps_local.to_csv(
        gps_file,
        index=False,
    )

    # ---------------------------------------------------------------
    # Generate plots
    # ---------------------------------------------------------------

    print("Generating plots...")

    create_plots(
        imu,
        fused,
        gps_local,
    )

    # ---------------------------------------------------------------
    # Generate summary
    # ---------------------------------------------------------------

    imu_rate, gps_rate, duration = create_summary(
        imu,
        gps_local,
        zone_number,
        zone_letter,
    )

    # ---------------------------------------------------------------
    # Final output
    # ---------------------------------------------------------------

    print()
    print("=" * 65)
    print("=== GTSAM CSV FUSION COMPLETE ===")
    print("=" * 65)

    print(
        f"IMU samples : {len(imu)}"
    )

    print(
        f"GPS samples : {len(gps_local)}"
    )

    print(
        f"IMU rate    : {imu_rate:.2f} Hz"
    )

    print(
        f"GPS rate    : {gps_rate:.2f} Hz"
    )

    print(
        f"Duration    : {duration:.2f} s"
    )

    print(
        f"Fused states: {len(fused)}"
    )

    print(
        f"Results     : {RESULTS_DIR}"
    )

    print()
    print("Created:")

    for file in sorted(RESULTS_DIR.iterdir()):

        if file.is_file():
            print(
                f"  {file.name}"
            )

    print()
    print("=" * 65)


if __name__ == "__main__":
    main()