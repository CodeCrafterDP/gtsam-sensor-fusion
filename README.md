# GTSAM Sensor Fusion: Smartphone IMU + GPS

## 1. Project Overview

This project performs **offline sensor fusion of smartphone IMU and GPS data using GTSAM** (Georgia Tech Smoothing and Mapping) and its incremental optimizer, **ISAM2**.

The objective is to combine high-rate inertial measurements from a smartphone with lower-rate GPS position measurements to estimate a smoother and more continuous vehicle/person trajectory, while also estimating orientation, velocity, and IMU bias.

The project is based on the sensor-fusion core of the open-source `gtsam_fusion` project by Paul Kemppi, adapted for the current Python GTSAM API and for offline CSV processing.

Upstream project:
https://github.com/PaulKemppi/gtsam_fusion

---

## 2. What I Created

The final repository is organized as:

```text
gtsam-sensor-fusion/
│
├── run_csv_fusion.py
├── compare_trajectory.py
│
├── scripts/
│   └── gtsam_fusion_core.py
│
├── results/
│   ├── trajectory_xy.png
│   ├── gps_vs_gtsam_trajectory.png
│   ├── accelerometer.png
│   ├── gyroscope.png
│   ├── orientation.png
│   ├── acc_bias.png
│   └── gyr_bias.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

The raw smartphone CSV files are intentionally not included in the public repository because they contain sensor/location data.

---

## 3. Sensor Data Used

The experiment uses three required sensor sources:

### Accelerometer
File:
`AccelerometerUncalibrated.csv`

Measured samples:
- 68,798 IMU samples

### Gyroscope
File:
`GyroscopeUncalibrated.csv`

Measured samples:
- 68,798 IMU samples

### GPS
File:
`Location.csv`

Measured samples:
- 933 GPS samples

### Metadata
`Metadata.csv` was used to inspect recording/device information but is not required by the fusion algorithm.

### Important measurement note

The original recording was intended to contain high-rate IMU and lower-rate GPS measurements. However, the actual CSV data was measured rather than assumed.

Measured rates from the provided recording:

| Measurement | Measured Rate |
|---|---:|
| IMU | 202.72 Hz |
| GPS | 2.66 Hz |
| Recording duration | 339.53 seconds |

Therefore, this project reports the **actual measured sampling rates** instead of claiming ideal 250 Hz IMU or 1 Hz GPS rates.

---

## 4. Processing Pipeline

The complete processing flow is:

```text
Smartphone Sensor Recorder CSV Files
                │
                ├── Uncalibrated Accelerometer
                ├── Uncalibrated Gyroscope
                └── GPS Location
                │
                ▼
        CSV Loading & Validation
                │
                ▼
        Axis / Timestamp Processing
                │
                ▼
     GPS Latitude/Longitude → UTM
                │
                ▼
          Local Metric Frame
                │
                ▼
       IMU + GPS Time Synchronization
                │
                ▼
        IMU Preintegration
                │
                ▼
       GTSAM IMU Factor + GPS Factor
                │
                ▼
             ISAM2
                │
                ▼
      Optimized State Estimation
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
     Position  Velocity  Orientation
                │
                ▼
          Bias Estimation
                │
                ▼
       CSV Results + Plots
```

---

## 5. Important Data Conversion

### 5.1 Sensor Axis Conversion

The smartphone CSV files store the sensor axes in this order:

```text
z, y, x
```

The fusion code converts them into the standard GTSAM processing order:

```text
x, y, z
```

This conversion is important because the sensor CSV column order must match the coordinate convention expected by the fusion pipeline.

---

## 6. GPS Coordinate Conversion

GPS data is initially provided as:

```text
latitude
longitude
altitude
```

Latitude and longitude are geographic coordinates and are not directly suitable for calculating trajectory distances in meters.

Therefore, the project converts GPS coordinates into a metric **UTM coordinate system**.

For this dataset:

```text
UTM Zone: 45Q
```

The first GPS point is used as the local origin, producing local metric coordinates:

```text
x = Easting - initial Easting
y = Northing - initial Northing
z = altitude - initial altitude
```

This allows the GPS trajectory and GTSAM trajectory to be plotted in meters.

---

## 7. Timestamp Synchronization

The sensor files contain a `seconds_elapsed` timestamp.

The fusion pipeline uses this timestamp to synchronize:

- Accelerometer measurements
- Gyroscope measurements
- GPS measurements

IMU measurements are processed sequentially, and GPS measurements are inserted when their timestamp is reached.

The code also calculates the time difference:

```text
dt = current_timestamp - previous_timestamp
```

Invalid, duplicate, or non-positive time intervals are skipped.

The first IMU sample is skipped because there is no previous timestamp from which a valid `dt` can be calculated.

---

## 8. GTSAM Processing

The main fusion engine is implemented in:

```text
scripts/gtsam_fusion_core.py
```

The core uses:

- GTSAM
- ISAM2
- IMU preintegration
- IMU factors
- GPS factors
- Bias evolution factors
- Pose, velocity and bias states

The original upstream core is designed around incremental smoothing and mapping using ISAM2. The implementation was adapted to work with the installed Python GTSAM 4.2.2 API.

---

## 9. State Estimated by the System

The fusion system estimates:

### Position

```text
x, y, z
```

### Orientation

```text
roll
pitch
yaw
```

### Velocity

```text
vx
vy
vz
```

### Accelerometer Bias

```text
acc_bias_x
acc_bias_y
acc_bias_z
```

### Gyroscope Bias

```text
gyr_bias_x
gyr_bias_y
gyr_bias_z
```

The final state is stored in:

```text
results/fused_results.csv
```

---

## 10. Main Python Files

### `run_csv_fusion.py`

This is the main execution script.

Its responsibilities are:

1. Load the accelerometer CSV.
2. Load the gyroscope CSV.
3. Load the GPS CSV.
4. Reorder sensor axes.
5. Read timestamps.
6. Convert GPS coordinates into local metric coordinates.
7. Synchronize GPS and IMU measurements.
8. Calculate IMU `dt`.
9. Feed IMU measurements into GTSAM.
10. Add GPS constraints.
11. Run ISAM2 optimization.
12. Save the estimated trajectory and state information.
13. Generate result plots.

---

### `scripts/gtsam_fusion_core.py`

This file contains the reusable GTSAM fusion engine.

It handles:

- ISAM2 initialization
- Initial pose/velocity/bias priors
- IMU preintegration
- IMU prediction
- GPS measurement insertion
- GPS factors
- IMU factors
- Bias evolution
- Incremental optimization
- Pose/velocity/bias state updates

The core follows the architecture of the upstream `gtsam_fusion` implementation while including compatibility changes required by the installed GTSAM Python API.

---

### `compare_trajectory.py`

This script compares:

```text
GPS trajectory
vs.
GTSAM fused trajectory
```

It produces:

```text
results/gps_vs_gtsam_trajectory.png
```

This provides a direct visual comparison between the raw GPS path and the fused estimate.

---

## 11. GTSAM Compatibility Changes

The original upstream implementation was written for an older Python GTSAM API.

Several compatibility changes were required for GTSAM 4.2.2.

Examples include:

### ISAM2 relinearization

Older style:

```python
setRelinearizeSkip(...)
```

Updated usage:

```python
relinearizeSkip = ...
```

### Symbol creation

Updated to:

```python
gtsam.symbol('x', 0)
gtsam.symbol('v', 0)
gtsam.symbol('b', 0)
```

### Relinearization threshold

Updated to the available GTSAM API:

```python
setRelinearizeThreshold(...)
```

### Noise models

Updated to:

```python
gtsam.noiseModel.Isotropic.Sigmas(...)
```

### Quaternion access

Updated to the current quaternion API:

```python
toQuaternion()
```

and the quaternion components are converted explicitly into:

```text
[x, y, z, w]
```

These changes allow the original fusion concept to work with the installed GTSAM version.

---

## 12. Installation

The project was developed and tested on:

```text
Ubuntu 24.04
Python 3
GTSAM 4.2.2
```

A Python virtual environment was used.

Install dependencies with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required Python packages:

```text
gtsam==4.2.2
numpy==1.26.4
pandas==3.0.6
matplotlib==3.11.2
utm==0.9.0
```

---

## 13. Running the Fusion

Place the required input CSV files in:

```text
data/
```

Required files:

```text
data/AccelerometerUncalibrated.csv
data/GyroscopeUncalibrated.csv
data/Location.csv
```

Then run:

```bash
python run_csv_fusion.py
```

The generated files are placed in:

```text
results/
```

---

## 14. Output Files

The fusion pipeline generated the following results:

### `fused_results.csv`

Contains the estimated state for each processed IMU timestamp.

Columns include:

```text
time
x
y
z
roll_deg
pitch_deg
yaw_deg
vx
vy
vz
acc_bias_x
acc_bias_y
acc_bias_z
gyr_bias_x
gyr_bias_y
gyr_bias_z
```

### `gps_local.csv`

Contains GPS measurements converted to local metric coordinates.

### `run_summary.txt`

Contains the experiment summary and measured sampling rates.

### Plots

The project generates:

```text
accelerometer.png
gyroscope.png
orientation.png
acc_bias.png
gyr_bias.png
trajectory_xy.png
gps_vs_gtsam_trajectory.png
```

---

## 15. Experimental Result

The fusion pipeline was successfully executed on the supplied smartphone dataset.

Final processing summary:

```text
IMU samples       : 68798
GPS samples       : 933
IMU rate          : 202.72 Hz
GPS rate          : 2.66 Hz
Duration          : 339.53 s
UTM zone          : 45Q
```

The system successfully generated:

- Fused position
- Orientation
- Velocity
- Accelerometer bias estimates
- Gyroscope bias estimates
- GPS local trajectory
- GTSAM fused trajectory
- Visualization plots

The trajectory result can be inspected using:

```text
results/trajectory_xy.png
```

and the direct GPS-vs-fused comparison using:

```text
results/gps_vs_gtsam_trajectory.png
```

---

## 16. Why Sensor Fusion Is Useful

GPS provides an absolute position reference but has a comparatively low update rate and can contain noise.

The IMU provides high-frequency motion information but suffers from:

- Sensor noise
- Bias
- Drift
- Accumulated integration error

GTSAM combines these complementary measurements in a probabilistic optimization framework.

Conceptually:

```text
High-rate IMU
     │
     ▼
Short-term motion estimation
     │
     ├──────────────┐
     │              │
     ▼              ▼
 IMU Factors     GPS Factors
     │              │
     └───────┬──────┘
             ▼
           ISAM2
             │
             ▼
       Optimized State
```

The GPS measurements help constrain long-term drift, while the IMU provides dense motion information between GPS updates.

---

## 17. Limitations

This experiment has several limitations:

1. The smartphone IMU is uncalibrated.
2. The dataset contains sensor noise and possible phone-motion artifacts.
3. The phone orientation during recording affects the interpretation of raw IMU measurements.
4. GPS measurements can contain multipath and positioning noise.
5. The experiment is offline rather than a real-time ROS pipeline.
6. The current implementation uses a simplified local coordinate representation.
7. No ground-truth trajectory is available, so absolute position/orientation accuracy cannot be claimed from this experiment alone.

Therefore, the generated trajectory should be interpreted as a sensor-fusion estimate rather than a ground-truth measurement.

---

## 18. Future Improvements

Possible future work includes:

- IMU calibration
- Better gravity initialization
- Magnetometer integration
- Wheel odometry integration
- Camera/stereo visual odometry
- ZED2i camera integration
- ROS 2 integration
- Real-time sensor streaming
- More advanced GPS noise modeling
- Ground-truth based accuracy evaluation
- RMSE / ATE / RPE trajectory evaluation
- Improved handling of coordinate frames
- Sensor extrinsic calibration

The eventual robotics pipeline can combine stereo-camera depth estimation with inertial and GPS information.

---

## 19. Repository Safety and Privacy

Raw smartphone sensor data is not included in the public repository.

In particular, GPS data can reveal the physical location associated with a recording. Therefore, raw CSV files should remain local unless there is explicit permission to publish them.

The `.gitignore` is configured to exclude:

```text
data/*.csv
```

as well as:

```text
.venv/
__pycache__/
```

and runtime CSV/text outputs.

---

## 20. Project Learning Outcome

This project demonstrates a complete offline sensor-fusion workflow:

```text
Raw Smartphone Data
        ↓
CSV Processing
        ↓
Timestamp Synchronization
        ↓
Coordinate Transformation
        ↓
IMU Preintegration
        ↓
GPS + IMU Factors
        ↓
ISAM2 Optimization
        ↓
Trajectory / Orientation / Velocity / Bias
        ↓
Plots and Analysis
```

It also demonstrates how an existing GTSAM-based research/robotics codebase can be adapted to a modern Python environment and a real smartphone dataset.

---

## 21. References

### GTSAM

GTSAM is used as the underlying factor-graph and optimization framework.

https://gtsam.org/

### Upstream GTSAM Fusion Project

Original fusion architecture:

https://github.com/PaulKemppi/gtsam_fusion

The upstream implementation provides the core concept for fusing IMU, GPS and pose measurements through GTSAM/ISAM2.

### Python

https://www.python.org/

### NumPy

https://numpy.org/

### Pandas

https://pandas.pydata.org/

### Matplotlib

https://matplotlib.org/

### UTM

https://pypi.org/project/utm/

---

## 22. Author

**Debraj Podder**

B.Tech CSE (AIML)  
Adamas University

This repository was created as part of a robotics/sensor-fusion learning and development project.

---

## 23. Final Summary

The project takes real smartphone accelerometer, gyroscope and GPS measurements and processes them through a GTSAM-based sensor-fusion pipeline.

The main contribution is the complete offline workflow that converts raw CSV sensor data into a synchronized, metric-coordinate dataset and then estimates a continuous trajectory using **IMU preintegration + GPS constraints + ISAM2 optimization**.

The implementation was successfully executed on the provided dataset and produced trajectory, orientation, velocity and IMU-bias outputs for further analysis.
