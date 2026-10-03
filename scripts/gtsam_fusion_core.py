#!/usr/bin/python3

"""
GTSAM Fusion Core

Provides functionality to fuse:
    - IMU
    - GPS
    - Absolute 6DOF pose
    - Relative 6DOF pose

using GTSAM ISAM2 incremental smoothing and mapping.

This version contains compatibility updates required for
GTSAM 4.2.2 Python bindings.
"""

from gtsam import gtsam

import numpy as np#!/usr/bin/python3

"""
GTSAM Fusion Core

Provides functionality to fuse:
    - IMU
    - GPS
    - Absolute 6DOF pose
    - Relative 6DOF pose

using GTSAM ISAM2 incremental smoothing and mapping.

This version contains compatibility updates required for
GTSAM 4.2.2 Python bindings.
"""

from gtsam import gtsam

import numpy as np
from collections import deque
import heapq


# ============================================================================
# GTSAM / NumPy conversion functions
# ============================================================================

def gtsam_pose_to_numpy(gtsam_pose):
    """
    Convert GTSAM Pose3 to NumPy arrays.

    Returns:
        position:
            np.array([x, y, z])

        orientation:
            np.array([qx, qy, qz, qw])
            Quaternion in xyzw order.
    """

    position = np.array([
        gtsam_pose.x(),
        gtsam_pose.y(),
        gtsam_pose.z()
    ])

    # GTSAM 4.2.2 uses toQuaternion()
    quat = gtsam_pose.rotation().toQuaternion()

    orientation = np.array([
        quat.x(),
        quat.y(),
        quat.z(),
        quat.w()
    ])

    return position, orientation


def numpy_pose_to_gtsam(position, orientation):
    """
    Convert NumPy position and quaternion to GTSAM Pose3.

    Input:
        position:
            [x, y, z]

        orientation:
            [qx, qy, qz, qw]
    """

    return gtsam.Pose3(
        gtsam.Rot3.Quaternion(
            orientation[3],
            orientation[0],
            orientation[1],
            orientation[2]
        ),
        np.array([
            position[0],
            position[1],
            position[2]
        ])
    )


# ============================================================================
# GTSAM Fusion Core
# ============================================================================

class GtsamFusionCore():

    """
    Core functions for ISAM2-based sensor fusion.
    """

    def __init__(self, params):

        """
        Initialize:
            - ISAM2
            - IMU preintegration
            - initial state
            - prior factors
            - measurement queues
        """

        # --------------------------------------------------------------------
        # Measurement queues
        # --------------------------------------------------------------------

        self.__imu_measurements_predict = []

        self.__imu_measurements_optimize = []

        self.__imu_samples = []

        self.__opt_measurements = []

        self.__opt_meas_buffer_time = (
            params['opt_meas_buffer_time']
        )

        # --------------------------------------------------------------------
        # ISAM2 initialization
        # --------------------------------------------------------------------

        isam2_params = gtsam.ISAM2Params()

        # GTSAM 4.2.2 compatible API
        isam2_params.setRelinearizeThreshold(
            params['relinearize_th']
        )

        # GTSAM 4.2.2 compatible API
        isam2_params.relinearizeSkip = (
            params['relinearize_skip']
        )

        isam2_params.setFactorization(
            params['factorization']
        )

        self.__isam2 = gtsam.ISAM2(
            isam2_params
        )

        self.__graph = gtsam.NonlinearFactorGraph()

        self.__initial_estimate = gtsam.Values()

        # --------------------------------------------------------------------
        # Relative pose state
        # --------------------------------------------------------------------

        self.__prev_relative_pose = None

        self.__prev_pose_key = None

        self.__min_imu_sample_count_for_integration = 2

        # --------------------------------------------------------------------
        # ISAM2 keys
        # --------------------------------------------------------------------
        #
        # GTSAM 4.2.2 compatible symbol API
        #

        self.__pose_key = gtsam.symbol('x', 0)

        self.__vel_key = gtsam.symbol('v', 0)

        self.__bias_key = gtsam.symbol('b', 0)

        # --------------------------------------------------------------------
        # IMU preintegration parameters
        # --------------------------------------------------------------------

        self.__pre_integration_params = (
            gtsam.PreintegrationParams(
                np.asarray(params['g'])
            )
        )

        self.__pre_integration_params.setAccelerometerCovariance(
            np.eye(3) *
            np.power(
                params['sigma_accelerometer'],
                2
            )
        )

        self.__pre_integration_params.setGyroscopeCovariance(
            np.eye(3) *
            np.power(
                params['sigma_gyroscope'],
                2
            )
        )

        self.__pre_integration_params.setIntegrationCovariance(
            np.eye(3) *
            params['sigma_integration'] ** 2
        )

        self.__pre_integration_params.setUse2ndOrderCoriolis(
            params['use_2nd_order_coriolis']
        )

        self.__pre_integration_params.setOmegaCoriolis(
            np.array(
                params['omega_coriolis']
            )
        )

        self.__pre_integration_params.setBodyPSensor(
            numpy_pose_to_gtsam(
                params['b2s_pos'],
                params['b2s_ori']
            )
        )

        # --------------------------------------------------------------------
        # Initial state
        # --------------------------------------------------------------------

        self.__current_time = 0

        self.__current_pose = numpy_pose_to_gtsam(
            params['init_pos'],
            params['init_ori']
        )

        self.__predicted_pose = self.__current_pose

        self.__current_vel = np.asarray(
            params['init_vel']
        )

        self.__current_bias = (
            gtsam.imuBias_ConstantBias(
                np.asarray(
                    params['init_acc_bias']
                ),
                np.asarray(
                    params['init_gyr_bias']
                )
            )
        )

        # --------------------------------------------------------------------
        # State stored for prediction
        # --------------------------------------------------------------------

        self.__last_opt_time = (
            self.__current_time
        )

        self.__last_opt_pose = (
            self.__current_pose
        )

        self.__last_opt_vel = (
            self.__current_vel
        )

        self.__last_opt_bias = (
            self.__current_bias
        )

        # --------------------------------------------------------------------
        # IMU accumulation
        # --------------------------------------------------------------------

        self.__imu_accum = (
            gtsam.PreintegratedImuMeasurements(
                self.__pre_integration_params
            )
        )

        # --------------------------------------------------------------------
        # Initial uncertainty
        #
        # GTSAM 4.2.2 compatible noise model API:
        # gtsam.noiseModel.Isotropic.Sigmas()
        # --------------------------------------------------------------------

        self.__sigma_init_pose = (
            gtsam.noiseModel.Isotropic.Sigmas(
                np.array([
                    params['sigma_init_ori'],
                    params['sigma_init_ori'],
                    params['sigma_init_ori'],

                    params['sigma_init_pos'],
                    params['sigma_init_pos'],
                    params['sigma_init_pos']
                ])
            )
        )

        self.__sigma_init_vel = (
            gtsam.noiseModel.Isotropic.Sigmas(
                np.array([
                    params['sigma_init_vel'],
                    params['sigma_init_vel'],
                    params['sigma_init_vel']
                ])
            )
        )

        self.__sigma_init_bias = (
            gtsam.noiseModel.Isotropic.Sigmas(
                np.array([
                    params['sigma_acc_init_bias'],
                    params['sigma_acc_init_bias'],
                    params['sigma_acc_init_bias'],

                    params['sigma_gyr_init_bias'],
                    params['sigma_gyr_init_bias'],
                    params['sigma_gyr_init_bias']
                ])
            )
        )

        # --------------------------------------------------------------------
        # Measurement noise
        # --------------------------------------------------------------------

        self.__pose_noise = (
            gtsam.noiseModel.Isotropic.Sigmas(
                np.concatenate((
                    params['sigma_pose_rot'],
                    params['sigma_pose_pos']
                ))
            )
        )

        self.__gps_noise = (
            gtsam.noiseModel.Isotropic.Sigmas(
                np.asarray(
                    params['sigma_gps']
                )
            )
        )

        self.__bias_noise = (
            gtsam.noiseModel.Isotropic.Sigmas(
                np.concatenate((
                    params['sigma_acc_bias_evol'],
                    params['sigma_gyr_bias_evol']
                ))
            )
        )

        # --------------------------------------------------------------------
        # Prior factors
        # --------------------------------------------------------------------

        prior_pose_factor = (
            gtsam.PriorFactorPose3(
                self.__pose_key,
                self.__current_pose,
                self.__sigma_init_pose
            )
        )

        self.__graph.add(
            prior_pose_factor
        )

        prior_vel_factor = (
            gtsam.PriorFactorVector(
                self.__vel_key,
                self.__current_vel,
                self.__sigma_init_vel
            )
        )

        self.__graph.add(
            prior_vel_factor
        )

        prior_bias_factor = (
            gtsam.PriorFactorConstantBias(
                self.__bias_key,
                self.__current_bias,
                self.__sigma_init_bias
            )
        )

        self.__graph.add(
            prior_bias_factor
        )

        # --------------------------------------------------------------------
        # Initial estimates
        # --------------------------------------------------------------------

        self.__initial_estimate.insert(
            self.__pose_key,
            self.__current_pose
        )

        self.__initial_estimate.insert(
            self.__vel_key,
            self.__current_vel
        )

        self.__initial_estimate.insert(
            self.__bias_key,
            self.__current_bias
        )

    # =========================================================================
    # Relative pose measurement
    # =========================================================================

    def add_relative_pose_measurement(
        self,
        time,
        position,
        orientation
    ):

        """
        Add relative 6DOF pose measurement.

        Input:
            time: float
            position: np.array([x, y, z])
            orientation: np.array([qx, qy, qz, qw])
        """

        heapq.heappush(
            self.__opt_measurements,
            (
                time,
                'relative_pose',
                position,
                orientation
            )
        )

        buffer_time = (
            heapq.nlargest(
                1,
                self.__opt_measurements
            )[0][0]
            -
            heapq.nsmallest(
                1,
                self.__opt_measurements
            )[0][0]
        )

        if buffer_time >= self.__opt_meas_buffer_time:

            self.__trigger_update(
                heapq.heappop(
                    self.__opt_measurements
                )
            )

    # =========================================================================
    # Absolute pose measurement
    # =========================================================================

    def add_absolute_pose_measurement(
        self,
        time,
        position,
        orientation
    ):

        """
        Add absolute 6DOF pose measurement.

        Input:
            time: float
            position: np.array([x, y, z])
            orientation: np.array([qx, qy, qz, qw])
        """

        heapq.heappush(
            self.__opt_measurements,
            (
                time,
                'absolute_pose',
                position,
                orientation
            )
        )

        buffer_time = (
            heapq.nlargest(
                1,
                self.__opt_measurements
            )[0][0]
            -
            heapq.nsmallest(
                1,
                self.__opt_measurements
            )[0][0]
        )

        if buffer_time >= self.__opt_meas_buffer_time:

            self.__trigger_update(
                heapq.heappop(
                    self.__opt_measurements
                )
            )

    # =========================================================================
    # GPS measurement
    # =========================================================================

    def add_gps_measurement(
        self,
        time,
        position
    ):

        """
        Add GPS measurement.

        GPS must already be converted into metric planar coordinates.

        Input:
            time:
                timestamp in seconds

            position:
                np.array([x, y, z])
        """

        heapq.heappush(
            self.__opt_measurements,
            (
                time,
                'gps',
                position
            )
        )

        buffer_time = (
            heapq.nlargest(
                1,
                self.__opt_measurements
            )[0][0]
            -
            heapq.nsmallest(
                1,
                self.__opt_measurements
            )[0][0]
        )

        if buffer_time >= self.__opt_meas_buffer_time:

            self.__trigger_update(
                heapq.heappop(
                    self.__opt_measurements
                )
            )

    # =========================================================================
    # IMU measurement
    # =========================================================================

    def add_imu_measurement(
        self,
        time,
        linear_acceleration,
        angular_velocity,
        dt
    ):

        """
        Add IMU measurement.

        Input:
            linear_acceleration:
                np.array([x, y, z])

            angular_velocity:
                np.array([x, y, z])

            dt:
                time difference between consecutive samples

        Output:
            position
            orientation
            velocity
            accelerometer bias
            gyroscope bias
        """

        # Add measurement to prediction queue.
        heapq.heappush(
            self.__imu_measurements_predict,
            (
                time,
                linear_acceleration,
                angular_velocity,
                dt
            )
        )

        # Add measurement to optimization queue.
        heapq.heappush(
            self.__imu_measurements_optimize,
            (
                time,
                linear_acceleration,
                angular_velocity,
                dt
            )
        )

        # Process oldest sample.
        return self.__imu_predict()

    # =========================================================================
    # Trigger optimization
    # =========================================================================

    def __trigger_update(
        self,
        measurement
    ):

        """
        Trigger optimization based on GPS/pose measurement.
        """

        meas_time = measurement[0]

        meas_type = measurement[1]

        imu_samples = []

        # ---------------------------------------------------------------
        # Collect IMU measurements before measurement time
        # ---------------------------------------------------------------

        while True:

            if not self.__imu_measurements_optimize:
                break

            imu_sample = heapq.heappop(
                self.__imu_measurements_optimize
            )

            if imu_sample[0] < meas_time:

                imu_samples.append(
                    imu_sample
                )

            else:

                # Put the sample back because it belongs
                # after the current measurement.
                heapq.heappush(
                    self.__imu_measurements_optimize,
                    imu_sample
                )

                break

        # ---------------------------------------------------------------
        # Need at least two IMU measurements
        # ---------------------------------------------------------------

        if (
            len(imu_samples)
            <
            self.__min_imu_sample_count_for_integration
        ):

            for imu_sample in imu_samples:

                heapq.heappush(
                    self.__imu_measurements_optimize,
                    imu_sample
                )

            print(
                "Ignoring measurement at: {}".format(
                    meas_time
                )
            )

            return

        # ---------------------------------------------------------------
        # Handle relative pose
        # ---------------------------------------------------------------

        if (
            meas_type == "relative_pose"
            and
            not self.__prev_relative_pose
        ):

            pose = numpy_pose_to_gtsam(
                measurement[2],
                measurement[3]
            )

            self.__prev_relative_pose = pose

            self.__prev_pose_key = (
                self.__pose_key
            )

        else:

            # -----------------------------------------------------------
            # Create new state keys
            # -----------------------------------------------------------

            self.__pose_key += 1

            self.__vel_key += 1

            self.__bias_key += 1

            # -----------------------------------------------------------
            # GPS measurement
            # -----------------------------------------------------------

            if meas_type == "gps":

                position = measurement[2]

                # GTSAM 4.2.2 accepts the position as
                # a NumPy vector for GPSFactor.
                point = np.array([
                    position[0],
                    position[1],
                    position[2]
                ])

                gps_factor = (
                    gtsam.GPSFactor(
                        self.__pose_key,
                        point,
                        self.__gps_noise
                    )
                )

                self.__graph.add(
                    gps_factor
                )

            # -----------------------------------------------------------
            # Relative pose measurement
            # -----------------------------------------------------------

            elif meas_type == "relative_pose":

                pose = numpy_pose_to_gtsam(
                    measurement[2],
                    measurement[3]
                )

                between_pose_factor = (
                    gtsam.BetweenFactorPose3(
                        self.__prev_pose_key,
                        self.__pose_key,
                        self.__prev_relative_pose.between(
                            pose
                        ),
                        self.__pose_noise
                    )
                )

                self.__graph.add(
                    between_pose_factor
                )

                self.__prev_relative_pose = pose

                self.__prev_pose_key = (
                    self.__pose_key
                )

            # -----------------------------------------------------------
            # Absolute pose measurement
            # -----------------------------------------------------------

            elif meas_type == "absolute_pose":

                pose = numpy_pose_to_gtsam(
                    measurement[2],
                    measurement[3]
                )

                pose_factor = (
                    gtsam.PriorFactorPose3(
                        self.__pose_key,
                        pose,
                        self.__pose_noise
                    )
                )

                self.__graph.add(
                    pose_factor
                )

            # -----------------------------------------------------------
            # Optimize
            # -----------------------------------------------------------

            self.__optimize(
                meas_time,
                imu_samples
            )

    # =========================================================================
    # IMU prediction
    # =========================================================================

    def __imu_predict(self):

        """
        Perform IMU prediction.
        """

        # ---------------------------------------------------------------
        # Check if a new optimized pose is available
        # ---------------------------------------------------------------

        if (
            self.__current_time
            >
            self.__last_opt_time
        ):

            # Store optimized state.
            self.__last_opt_time = (
                self.__current_time
            )

            self.__last_opt_pose = (
                self.__current_pose
            )

            self.__last_opt_vel = (
                self.__current_vel
            )

            self.__last_opt_bias = (
                self.__current_bias
            )

            # Start a new integration.
            self.__imu_accum.resetIntegration()

            # Re-integrate remaining IMU samples.
            new_imu_samples = []

            for sample in self.__imu_samples:

                if (
                    sample[0]
                    >
                    self.__last_opt_time
                ):

                    self.__imu_accum.integrateMeasurement(
                        sample[1],
                        sample[2],
                        sample[3]
                    )

                    new_imu_samples.append(
                        sample
                    )

            self.__imu_samples = (
                new_imu_samples
            )

        # ---------------------------------------------------------------
        # Get newest/oldest sample from queue
        # ---------------------------------------------------------------

        (
            time,
            linear_acceleration,
            angular_velocity,
            dt
        ) = heapq.heappop(
            self.__imu_measurements_predict
        )

        # Store sample for re-integration.
        self.__imu_samples.append(
            (
                time,
                linear_acceleration,
                angular_velocity,
                dt
            )
        )

        # Integrate IMU measurement.
        self.__imu_accum.integrateMeasurement(
            linear_acceleration,
            angular_velocity,
            dt
        )

        # ---------------------------------------------------------------
        # Predict navigation state
        # ---------------------------------------------------------------

        predicted_nav_state = (
            self.__imu_accum.predict(
                gtsam.NavState(
                    self.__last_opt_pose,
                    self.__last_opt_vel
                ),
                self.__last_opt_bias
            )
        )

        pos, ori = gtsam_pose_to_numpy(
            predicted_nav_state.pose()
        )

        return (
            pos,
            ori,
            self.__last_opt_vel,
            self.__last_opt_bias.accelerometer(),
            self.__last_opt_bias.gyroscope()
        )

    # =========================================================================
    # IMU optimization
    # =========================================================================

    def __imu_update(
        self,
        imu_samples
    ):

        """
        Create IMU factor and perform bias evolution.
        """

        # ---------------------------------------------------------------
        # New IMU preintegration object
        # ---------------------------------------------------------------

        imu_accum = (
            gtsam.PreintegratedImuMeasurements(
                self.__pre_integration_params
            )
        )

        # ---------------------------------------------------------------
        # Preintegrate all IMU measurements
        # ---------------------------------------------------------------

        for imu_sample in imu_samples:

            imu_accum.integrateMeasurement(
                imu_sample[1],
                imu_sample[2],
                imu_sample[3]
            )

        # ---------------------------------------------------------------
        # Current optimized state
        # ---------------------------------------------------------------

        last_opt_pose = (
            self.__current_pose
        )

        last_opt_vel = (
            self.__current_vel
        )

        last_opt_bias = (
            self.__current_bias
        )

        # ---------------------------------------------------------------
        # Predict state
        # ---------------------------------------------------------------

        predicted_nav_state = (
            imu_accum.predict(
                gtsam.NavState(
                    last_opt_pose,
                    last_opt_vel
                ),
                last_opt_bias
            )
        )

        # ---------------------------------------------------------------
        # Create IMU factor
        # ---------------------------------------------------------------

        imu_factor = (
            gtsam.ImuFactor(
                self.__pose_key - 1,
                self.__vel_key - 1,
                self.__pose_key,
                self.__vel_key,
                self.__bias_key,
                imu_accum
            )
        )

        self.__graph.add(
            imu_factor
        )

        return predicted_nav_state

    # =========================================================================
    # Optimization
    # =========================================================================

    def __optimize(
        self,
        meas_time,
        imu_samples
    ):

        """
        Perform GTSAM optimization.
        """

        # ---------------------------------------------------------------
        # IMU preintegration
        # ---------------------------------------------------------------

        predicted_nav_state = (
            self.__imu_update(
                imu_samples
            )
        )

        # ---------------------------------------------------------------
        # Add predicted state to initial estimates
        # ---------------------------------------------------------------

        self.__initial_estimate.insert(
            self.__pose_key,
            predicted_nav_state.pose()
        )

        self.__initial_estimate.insert(
            self.__vel_key,
            predicted_nav_state.velocity()
        )

        self.__initial_estimate.insert(
            self.__bias_key,
            self.__current_bias
        )

        # ---------------------------------------------------------------
        # Bias evolution factor
        # ---------------------------------------------------------------

        bias_factor = (
            gtsam.BetweenFactorConstantBias(
                self.__bias_key - 1,
                self.__bias_key,
                gtsam.imuBias_ConstantBias(),
                self.__bias_noise
            )
        )

        self.__graph.add(
            bias_factor
        )

        # ---------------------------------------------------------------
        # ISAM2 update
        # ---------------------------------------------------------------

        result = self.__isam2_update()

        if result:

            self.__current_time = (
                meas_time
            )

            self.__current_pose = (
                result.atPose3(
                    self.__pose_key
                )
            )

            self.__current_vel = (
                result.atVector(
                    self.__vel_key
                )
            )

            self.__current_bias = (
                result.atimuBias_ConstantBias(
                    self.__bias_key
                )
            )

    # =========================================================================
    # ISAM2 update
    # =========================================================================

    def __isam2_update(self):

        """
        Perform ISAM2 update and calculate estimate.
        """

        result = None

        try:

            # -----------------------------------------------------------
            # Update ISAM2
            # -----------------------------------------------------------

            self.__isam2.update(
                self.__graph,
                self.__initial_estimate
            )

            result = (
                self.__isam2.calculateEstimate()
            )

        except RuntimeError as e:

            print(
                "Runtime error in optimization: {}".format(
                    e
                )
            )

        except IndexError as e:

            print(
                "Index error in optimization: {}".format(
                    e
                )
            )

        except TypeError as e:

            print(
                "Type error in optimization: {}".format(
                    e
                )
            )

        # ---------------------------------------------------------------
        # Clear graph and initial estimates
        # ---------------------------------------------------------------

        self.__graph.resize(0)

        self.__initial_estimate.clear()

        return result
from collections import deque
import heapq


# ============================================================================
# GTSAM / NumPy conversion functions
# ============================================================================

def gtsam_pose_to_numpy(gtsam_pose):
    """
    Convert GTSAM Pose3 to NumPy arrays.

    Returns:
        position:
            np.array([x, y, z])

        orientation:
            np.array([qx, qy, qz, qw])
            Quaternion in xyzw order.
    """

    position = np.array([
        gtsam_pose.x(),
        gtsam_pose.y(),
        gtsam_pose.z()
    ])

    # GTSAM 4.2.2 uses toQuaternion()
    quat = gtsam_pose.rotation().toQuaternion()

    orientation = np.array([
        quat.x(),
        quat.y(),
        quat.z(),
        quat.w()
    ])

    return position, orientation


def numpy_pose_to_gtsam(position, orientation):
    """
    Convert NumPy position and quaternion to GTSAM Pose3.

    Input:
        position:
            [x, y, z]

        orientation:
            [qx, qy, qz, qw]
    """

    return gtsam.Pose3(
        gtsam.Rot3.Quaternion(
            orientation[3],
            orientation[0],
            orientation[1],
            orientation[2]
        ),
        np.array([
            position[0],
            position[1],
            position[2]
        ])
    )


# ============================================================================
# GTSAM Fusion Core
# ============================================================================

class GtsamFusionCore():

    """
    Core functions for ISAM2-based sensor fusion.
    """

    def __init__(self, params):

        """
        Initialize:
            - ISAM2
            - IMU preintegration
            - initial state
            - prior factors
            - measurement queues
        """

        # --------------------------------------------------------------------
        # Measurement queues
        # --------------------------------------------------------------------

        self.__imu_measurements_predict = []

        self.__imu_measurements_optimize = []

        self.__imu_samples = []

        self.__opt_measurements = []

        self.__opt_meas_buffer_time = (
            params['opt_meas_buffer_time']
        )

        # --------------------------------------------------------------------
        # ISAM2 initialization
        # --------------------------------------------------------------------

        isam2_params = gtsam.ISAM2Params()

        # GTSAM 4.2.2 compatible API
        isam2_params.setRelinearizeThreshold(
            params['relinearize_th']
        )

        # GTSAM 4.2.2 compatible API
        isam2_params.relinearizeSkip = (
            params['relinearize_skip']
        )

        isam2_params.setFactorization(
            params['factorization']
        )

        self.__isam2 = gtsam.ISAM2(
            isam2_params
        )

        self.__graph = gtsam.NonlinearFactorGraph()

        self.__initial_estimate = gtsam.Values()

        # --------------------------------------------------------------------
        # Relative pose state
        # --------------------------------------------------------------------

        self.__prev_relative_pose = None

        self.__prev_pose_key = None

        self.__min_imu_sample_count_for_integration = 2

        # --------------------------------------------------------------------
        # ISAM2 keys
        # --------------------------------------------------------------------
        #
        # GTSAM 4.2.2 compatible symbol API
        #

        self.__pose_key = gtsam.symbol('x', 0)

        self.__vel_key = gtsam.symbol('v', 0)

        self.__bias_key = gtsam.symbol('b', 0)

        # --------------------------------------------------------------------
        # IMU preintegration parameters
        # --------------------------------------------------------------------

        self.__pre_integration_params = (
            gtsam.PreintegrationParams(
                np.asarray(params['g'])
            )
        )

        self.__pre_integration_params.setAccelerometerCovariance(
            np.eye(3) *
            np.power(
                params['sigma_accelerometer'],
                2
            )
        )

        self.__pre_integration_params.setGyroscopeCovariance(
            np.eye(3) *
            np.power(
                params['sigma_gyroscope'],
                2
            )
        )

        self.__pre_integration_params.setIntegrationCovariance(
            np.eye(3) *
            params['sigma_integration'] ** 2
        )

        self.__pre_integration_params.setUse2ndOrderCoriolis(
            params['use_2nd_order_coriolis']
        )

        self.__pre_integration_params.setOmegaCoriolis(
            np.array(
                params['omega_coriolis']
            )
        )

        self.__pre_integration_params.setBodyPSensor(
            numpy_pose_to_gtsam(
                params['b2s_pos'],
                params['b2s_ori']
            )
        )

        # --------------------------------------------------------------------
        # Initial state
        # --------------------------------------------------------------------

        self.__current_time = 0

        self.__current_pose = numpy_pose_to_gtsam(
            params['init_pos'],
            params['init_ori']
        )

        self.__predicted_pose = self.__current_pose

        self.__current_vel = np.asarray(
            params['init_vel']
        )

        self.__current_bias = (
            gtsam.imuBias_ConstantBias(
                np.asarray(
                    params['init_acc_bias']
                ),
                np.asarray(
                    params['init_gyr_bias']
                )
            )
        )

        # --------------------------------------------------------------------
        # State stored for prediction
        # --------------------------------------------------------------------

        self.__last_opt_time = (
            self.__current_time
        )

        self.__last_opt_pose = (
            self.__current_pose
        )

        self.__last_opt_vel = (
            self.__current_vel
        )

        self.__last_opt_bias = (
            self.__current_bias
        )

        # --------------------------------------------------------------------
        # IMU accumulation
        # --------------------------------------------------------------------

        self.__imu_accum = (
            gtsam.PreintegratedImuMeasurements(
                self.__pre_integration_params
            )
        )

        # --------------------------------------------------------------------
        # Initial uncertainty
        #
        # GTSAM 4.2.2 compatible noise model API:
        # gtsam.noiseModel.Isotropic.Sigmas()
        # --------------------------------------------------------------------

        self.__sigma_init_pose = (
            gtsam.noiseModel.Isotropic.Sigmas(
                np.array([
                    params['sigma_init_ori'],
                    params['sigma_init_ori'],
                    params['sigma_init_ori'],

                    params['sigma_init_pos'],
                    params['sigma_init_pos'],
                    params['sigma_init_pos']
                ])
            )
        )

        self.__sigma_init_vel = (
            gtsam.noiseModel.Isotropic.Sigmas(
                np.array([
                    params['sigma_init_vel'],
                    params['sigma_init_vel'],
                    params['sigma_init_vel']
                ])
            )
        )

        self.__sigma_init_bias = (
            gtsam.noiseModel.Isotropic.Sigmas(
                np.array([
                    params['sigma_acc_init_bias'],
                    params['sigma_acc_init_bias'],
                    params['sigma_acc_init_bias'],

                    params['sigma_gyr_init_bias'],
                    params['sigma_gyr_init_bias'],
                    params['sigma_gyr_init_bias']
                ])
            )
        )

        # --------------------------------------------------------------------
        # Measurement noise
        # --------------------------------------------------------------------

        self.__pose_noise = (
            gtsam.noiseModel.Isotropic.Sigmas(
                np.concatenate((
                    params['sigma_pose_rot'],
                    params['sigma_pose_pos']
                ))
            )
        )

        self.__gps_noise = (
            gtsam.noiseModel.Isotropic.Sigmas(
                np.asarray(
                    params['sigma_gps']
                )
            )
        )

        self.__bias_noise = (
            gtsam.noiseModel.Isotropic.Sigmas(
                np.concatenate((
                    params['sigma_acc_bias_evol'],
                    params['sigma_gyr_bias_evol']
                ))
            )
        )

        # --------------------------------------------------------------------
        # Prior factors
        # --------------------------------------------------------------------

        prior_pose_factor = (
            gtsam.PriorFactorPose3(
                self.__pose_key,
                self.__current_pose,
                self.__sigma_init_pose
            )
        )

        self.__graph.add(
            prior_pose_factor
        )

        prior_vel_factor = (
            gtsam.PriorFactorVector(
                self.__vel_key,
                self.__current_vel,
                self.__sigma_init_vel
            )
        )

        self.__graph.add(
            prior_vel_factor
        )

        prior_bias_factor = (
            gtsam.PriorFactorConstantBias(
                self.__bias_key,
                self.__current_bias,
                self.__sigma_init_bias
            )
        )

        self.__graph.add(
            prior_bias_factor
        )

        # --------------------------------------------------------------------
        # Initial estimates
        # --------------------------------------------------------------------

        self.__initial_estimate.insert(
            self.__pose_key,
            self.__current_pose
        )

        self.__initial_estimate.insert(
            self.__vel_key,
            self.__current_vel
        )

        self.__initial_estimate.insert(
            self.__bias_key,
            self.__current_bias
        )

    # =========================================================================
    # Relative pose measurement
    # =========================================================================

    def add_relative_pose_measurement(
        self,
        time,
        position,
        orientation
    ):

        """
        Add relative 6DOF pose measurement.

        Input:
            time: float
            position: np.array([x, y, z])
            orientation: np.array([qx, qy, qz, qw])
        """

        heapq.heappush(
            self.__opt_measurements,
            (
                time,
                'relative_pose',
                position,
                orientation
            )
        )

        buffer_time = (
            heapq.nlargest(
                1,
                self.__opt_measurements
            )[0][0]
            -
            heapq.nsmallest(
                1,
                self.__opt_measurements
            )[0][0]
        )

        if buffer_time >= self.__opt_meas_buffer_time:

            self.__trigger_update(
                heapq.heappop(
                    self.__opt_measurements
                )
            )

    # =========================================================================
    # Absolute pose measurement
    # =========================================================================

    def add_absolute_pose_measurement(
        self,
        time,
        position,
        orientation
    ):

        """
        Add absolute 6DOF pose measurement.

        Input:
            time: float
            position: np.array([x, y, z])
            orientation: np.array([qx, qy, qz, qw])
        """

        heapq.heappush(
            self.__opt_measurements,
            (
                time,
                'absolute_pose',
                position,
                orientation
            )
        )

        buffer_time = (
            heapq.nlargest(
                1,
                self.__opt_measurements
            )[0][0]
            -
            heapq.nsmallest(
                1,
                self.__opt_measurements
            )[0][0]
        )

        if buffer_time >= self.__opt_meas_buffer_time:

            self.__trigger_update(
                heapq.heappop(
                    self.__opt_measurements
                )
            )

    # =========================================================================
    # GPS measurement
    # =========================================================================

    def add_gps_measurement(
        self,
        time,
        position
    ):

        """
        Add GPS measurement.

        GPS must already be converted into metric planar coordinates.

        Input:
            time:
                timestamp in seconds

            position:
                np.array([x, y, z])
        """

        heapq.heappush(
            self.__opt_measurements,
            (
                time,
                'gps',
                position
            )
        )

        buffer_time = (
            heapq.nlargest(
                1,
                self.__opt_measurements
            )[0][0]
            -
            heapq.nsmallest(
                1,
                self.__opt_measurements
            )[0][0]
        )

        if buffer_time >= self.__opt_meas_buffer_time:

            self.__trigger_update(
                heapq.heappop(
                    self.__opt_measurements
                )
            )

    # =========================================================================
    # IMU measurement
    # =========================================================================

    def add_imu_measurement(
        self,
        time,
        linear_acceleration,
        angular_velocity,
        dt
    ):

        """
        Add IMU measurement.

        Input:
            linear_acceleration:
                np.array([x, y, z])

            angular_velocity:
                np.array([x, y, z])

            dt:
                time difference between consecutive samples

        Output:
            position
            orientation
            velocity
            accelerometer bias
            gyroscope bias
        """

        # Add measurement to prediction queue.
        heapq.heappush(
            self.__imu_measurements_predict,
            (
                time,
                linear_acceleration,
                angular_velocity,
                dt
            )
        )

        # Add measurement to optimization queue.
        heapq.heappush(
            self.__imu_measurements_optimize,
            (
                time,
                linear_acceleration,
                angular_velocity,
                dt
            )
        )

        # Process oldest sample.
        return self.__imu_predict()

    # =========================================================================
    # Trigger optimization
    # =========================================================================

    def __trigger_update(
        self,
        measurement
    ):

        """
        Trigger optimization based on GPS/pose measurement.
        """

        meas_time = measurement[0]

        meas_type = measurement[1]

        imu_samples = []

        # ---------------------------------------------------------------
        # Collect IMU measurements before measurement time
        # ---------------------------------------------------------------

        while True:

            if not self.__imu_measurements_optimize:
                break

            imu_sample = heapq.heappop(
                self.__imu_measurements_optimize
            )

            if imu_sample[0] < meas_time:

                imu_samples.append(
                    imu_sample
                )

            else:

                # Put the sample back because it belongs
                # after the current measurement.
                heapq.heappush(
                    self.__imu_measurements_optimize,
                    imu_sample
                )

                break

        # ---------------------------------------------------------------
        # Need at least two IMU measurements
        # ---------------------------------------------------------------

        if (
            len(imu_samples)
            <
            self.__min_imu_sample_count_for_integration
        ):

            for imu_sample in imu_samples:

                heapq.heappush(
                    self.__imu_measurements_optimize,
                    imu_sample
                )

            print(
                "Ignoring measurement at: {}".format(
                    meas_time
                )
            )

            return

        # ---------------------------------------------------------------
        # Handle relative pose
        # ---------------------------------------------------------------

        if (
            meas_type == "relative_pose"
            and
            not self.__prev_relative_pose
        ):

            pose = numpy_pose_to_gtsam(
                measurement[2],
                measurement[3]
            )

            self.__prev_relative_pose = pose

            self.__prev_pose_key = (
                self.__pose_key
            )

        else:

            # -----------------------------------------------------------
            # Create new state keys
            # -----------------------------------------------------------

            self.__pose_key += 1

            self.__vel_key += 1

            self.__bias_key += 1

            # -----------------------------------------------------------
            # GPS measurement
            # -----------------------------------------------------------

            if meas_type == "gps":

                position = measurement[2]

                # GTSAM 4.2.2 accepts the position as
                # a NumPy vector for GPSFactor.
                point = np.array([
                    position[0],
                    position[1],
                    position[2]
                ])

                gps_factor = (
                    gtsam.GPSFactor(
                        self.__pose_key,
                        point,
                        self.__gps_noise
                    )
                )

                self.__graph.add(
                    gps_factor
                )

            # -----------------------------------------------------------
            # Relative pose measurement
            # -----------------------------------------------------------

            elif meas_type == "relative_pose":

                pose = numpy_pose_to_gtsam(
                    measurement[2],
                    measurement[3]
                )

                between_pose_factor = (
                    gtsam.BetweenFactorPose3(
                        self.__prev_pose_key,
                        self.__pose_key,
                        self.__prev_relative_pose.between(
                            pose
                        ),
                        self.__pose_noise
                    )
                )

                self.__graph.add(
                    between_pose_factor
                )

                self.__prev_relative_pose = pose

                self.__prev_pose_key = (
                    self.__pose_key
                )

            # -----------------------------------------------------------
            # Absolute pose measurement
            # -----------------------------------------------------------

            elif meas_type == "absolute_pose":

                pose = numpy_pose_to_gtsam(
                    measurement[2],
                    measurement[3]
                )

                pose_factor = (
                    gtsam.PriorFactorPose3(
                        self.__pose_key,
                        pose,
                        self.__pose_noise
                    )
                )

                self.__graph.add(
                    pose_factor
                )

            # -----------------------------------------------------------
            # Optimize
            # -----------------------------------------------------------

            self.__optimize(
                meas_time,
                imu_samples
            )

    # =========================================================================
    # IMU prediction
    # =========================================================================

    def __imu_predict(self):

        """
        Perform IMU prediction.
        """

        # ---------------------------------------------------------------
        # Check if a new optimized pose is available
        # ---------------------------------------------------------------

        if (
            self.__current_time
            >
            self.__last_opt_time
        ):

            # Store optimized state.
            self.__last_opt_time = (
                self.__current_time
            )

            self.__last_opt_pose = (
                self.__current_pose
            )

            self.__last_opt_vel = (
                self.__current_vel
            )

            self.__last_opt_bias = (
                self.__current_bias
            )

            # Start a new integration.
            self.__imu_accum.resetIntegration()

            # Re-integrate remaining IMU samples.
            new_imu_samples = []

            for sample in self.__imu_samples:

                if (
                    sample[0]
                    >
                    self.__last_opt_time
                ):

                    self.__imu_accum.integrateMeasurement(
                        sample[1],
                        sample[2],
                        sample[3]
                    )

                    new_imu_samples.append(
                        sample
                    )

            self.__imu_samples = (
                new_imu_samples
            )

        # ---------------------------------------------------------------
        # Get newest/oldest sample from queue
        # ---------------------------------------------------------------

        (
            time,
            linear_acceleration,
            angular_velocity,
            dt
        ) = heapq.heappop(
            self.__imu_measurements_predict
        )

        # Store sample for re-integration.
        self.__imu_samples.append(
            (
                time,
                linear_acceleration,
                angular_velocity,
                dt
            )
        )

        # Integrate IMU measurement.
        self.__imu_accum.integrateMeasurement(
            linear_acceleration,
            angular_velocity,
            dt
        )

        # ---------------------------------------------------------------
        # Predict navigation state
        # ---------------------------------------------------------------

        predicted_nav_state = (
            self.__imu_accum.predict(
                gtsam.NavState(
                    self.__last_opt_pose,
                    self.__last_opt_vel
                ),
                self.__last_opt_bias
            )
        )

        pos, ori = gtsam_pose_to_numpy(
            predicted_nav_state.pose()
        )

        return (
            pos,
            ori,
            self.__last_opt_vel,
            self.__last_opt_bias.accelerometer(),
            self.__last_opt_bias.gyroscope()
        )

    # =========================================================================
    # IMU optimization
    # =========================================================================

    def __imu_update(
        self,
        imu_samples
    ):

        """
        Create IMU factor and perform bias evolution.
        """

        # ---------------------------------------------------------------
        # New IMU preintegration object
        # ---------------------------------------------------------------

        imu_accum = (
            gtsam.PreintegratedImuMeasurements(
                self.__pre_integration_params
            )
        )

        # ---------------------------------------------------------------
        # Preintegrate all IMU measurements
        # ---------------------------------------------------------------

        for imu_sample in imu_samples:

            imu_accum.integrateMeasurement(
                imu_sample[1],
                imu_sample[2],
                imu_sample[3]
            )

        # ---------------------------------------------------------------
        # Current optimized state
        # ---------------------------------------------------------------

        last_opt_pose = (
            self.__current_pose
        )

        last_opt_vel = (
            self.__current_vel
        )

        last_opt_bias = (
            self.__current_bias
        )

        # ---------------------------------------------------------------
        # Predict state
        # ---------------------------------------------------------------

        predicted_nav_state = (
            imu_accum.predict(
                gtsam.NavState(
                    last_opt_pose,
                    last_opt_vel
                ),
                last_opt_bias
            )
        )

        # ---------------------------------------------------------------
        # Create IMU factor
        # ---------------------------------------------------------------

        imu_factor = (
            gtsam.ImuFactor(
                self.__pose_key - 1,
                self.__vel_key - 1,
                self.__pose_key,
                self.__vel_key,
                self.__bias_key,
                imu_accum
            )
        )

        self.__graph.add(
            imu_factor
        )

        return predicted_nav_state

    # =========================================================================
    # Optimization
    # =========================================================================

    def __optimize(
        self,
        meas_time,
        imu_samples
    ):

        """
        Perform GTSAM optimization.
        """

        # ---------------------------------------------------------------
        # IMU preintegration
        # ---------------------------------------------------------------

        predicted_nav_state = (
            self.__imu_update(
                imu_samples
            )
        )

        # ---------------------------------------------------------------
        # Add predicted state to initial estimates
        # ---------------------------------------------------------------

        self.__initial_estimate.insert(
            self.__pose_key,
            predicted_nav_state.pose()
        )

        self.__initial_estimate.insert(
            self.__vel_key,
            predicted_nav_state.velocity()
        )

        self.__initial_estimate.insert(
            self.__bias_key,
            self.__current_bias
        )

        # ---------------------------------------------------------------
        # Bias evolution factor
        # ---------------------------------------------------------------

        bias_factor = (
            gtsam.BetweenFactorConstantBias(
                self.__bias_key - 1,
                self.__bias_key,
                gtsam.imuBias_ConstantBias(),
                self.__bias_noise
            )
        )

        self.__graph.add(
            bias_factor
        )

        # ---------------------------------------------------------------
        # ISAM2 update
        # ---------------------------------------------------------------

        result = self.__isam2_update()

        if result:

            self.__current_time = (
                meas_time
            )

            self.__current_pose = (
                result.atPose3(
                    self.__pose_key
                )
            )

            self.__current_vel = (
                result.atVector(
                    self.__vel_key
                )
            )

            self.__current_bias = (
                result.atimuBias_ConstantBias(
                    self.__bias_key
                )
            )

    # =========================================================================
    # ISAM2 update
    # =========================================================================

    def __isam2_update(self):

        """
        Perform ISAM2 update and calculate estimate.
        """

        result = None

        try:

            # -----------------------------------------------------------
            # Update ISAM2
            # -----------------------------------------------------------

            self.__isam2.update(
                self.__graph,
                self.__initial_estimate
            )

            result = (
                self.__isam2.calculateEstimate()
            )

        except RuntimeError as e:

            print(
                "Runtime error in optimization: {}".format(
                    e
                )
            )

        except IndexError as e:

            print(
                "Index error in optimization: {}".format(
                    e
                )
            )

        except TypeError as e:

            print(
                "Type error in optimization: {}".format(
                    e
                )
            )

        # ---------------------------------------------------------------
        # Clear graph and initial estimates
        # ---------------------------------------------------------------

        self.__graph.resize(0)

        self.__initial_estimate.clear()

        return result