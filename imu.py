from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class IMU:
    """
    Inertial Measurement Unit (IMU) data processing class.

    This class handles accelerometer and gyroscope data with optional
    calibration parameters.

    Parameters
    ----------
    ax, ay, az : array_like
        Accelerometer measurements along x, y, and z axes. assumes m/s^2
    gx, gy, gz : array_like
        Gyroscope measurements along x, y, and z axes. assumes deg/s
    correction_matrix : ndarray, optional
        3x3 matrix for correcting sensor measurements. Default is identity matrix.
    accel_bias : ndarray, optional
        Accelerometer bias vector (3x1). Default is zero vector.
    gyro_bias : ndarray, optional
        Gyroscope bias vector (3x1). Default is zero vector.
    sample_rate : float, optional
        Data sampling frequency in Hz. Default is 100.0 Hz.
    sensor2body_matrix : ndarray, optional
        Transformation matrix from sensor frame to body frame. Default is identity matrix.

    Notes
    -----
    All array inputs are converted to numpy arrays during initialization.
    The class performs two main transformations during initialization:
    1. Applies the correction data to adjust accelerometer and gyroscope data
    2. applies a sensor to body transformation

    Methods
    -------
    to_numpy()
        Convert data to a numpy array with shape (n, 6) where n is
        the number of samples and columns are [ax, ay, az, gx, gy, gz]
    """

    # Required fields
    ax: Union[List, np.ndarray, pd.Series, float]
    ay: Union[List, np.ndarray, pd.Series, float]
    az: Union[List, np.ndarray, pd.Series, float]
    gx: Union[List, np.ndarray, pd.Series, float]
    gy: Union[List, np.ndarray, pd.Series, float]
    gz: Union[List, np.ndarray, pd.Series, float]

    # Optional fields with defaults
    correction_matrix: Optional[np.ndarray] = field(default_factory=lambda: np.eye(3))
    accel_bias: Optional[np.ndarray] = field(default_factory=lambda: np.zeros(3))
    gyro_bias: Optional[np.ndarray] = field(default_factory=lambda: np.zeros(3))
    sample_rate: float = 100.0  # in Hz
    sensor2body_matrix: Optional[np.ndarray] = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        # Convert list inputs to numpy arrays
        self.ax = np.asarray(self.ax)
        self.ay = np.asarray(self.ay)
        self.az = np.asarray(self.az)
        self.gx = np.asarray(self.gx)
        self.gy = np.asarray(self.gy)
        self.gz = np.asarray(self.gz)

        # correct data
        self.ax, self.ay, self.az = self.get_corrected_accel().T
        self.gx, self.gy, self.gz = self.get_corrected_gyro().T

    def get_corrected_accel(self):
        """Return bias-corrected and transformed accelerometer data"""
        data = np.column_stack([self.ax, self.ay, self.az])
        return np.dot(self.correction_matrix.T, data.T).T + self.accel_bias

    def get_corrected_gyro(self):
        """Return bias-corrected and transformed gyroscope data"""
        data = np.column_stack([self.gx, self.gy, self.gz])
        return np.dot(self.correction_matrix.T, data.T).T - self.gyro_bias

    def get_imu_in_body_frame(self, apply=True):
        """Return data transformed to body frame"""

        acc = np.column_stack([self.ax, self.ay, self.az])
        acc_s2b = (self.sensor2body_matrix.T @ acc.T).T
        gyr = np.column_stack([self.gx, self.gy, self.gz])
        gyro_s2b = (self.sensor2body_matrix.T @ gyr.T).T
        if apply:
            self.ax, self.ay, self.az = acc_s2b.T
            self.gx, self.gy, self.gz = gyro_s2b.T
        return acc_s2b, gyro_s2b

    def to_numpy(self):
        """Compile individual components into numpy arrays"""
        data = np.column_stack([np.atleast_1d(self.ax), np.atleast_1d(self.ay), np.atleast_1d(self.az), np.atleast_1d(self.gx), np.atleast_1d(self.gy), np.atleast_1d(self.gz)])

        return data

    def plot(self, signalnames=None):
        """Plot the imu data. if no signalnames are provided, defaults to ['ax', 'ay', 'az', 'gx', 'gy', 'gz']"""
        import plotly.graph_objects as go
        import plotly.subplots as sp

        if signalnames is None:
            signalnames = ["ax", "ay", "az", "gx", "gy", "gz"]

        data = self.to_numpy()

        # Create subplots
        fig = sp.make_subplots(rows=len(signalnames), cols=1, subplot_titles=signalnames, shared_xaxes=True)

        for i, name in enumerate(signalnames):
            fig.add_trace(go.Scatter(y=data[:, i], name=name), row=i + 1, col=1)
            fig.update_yaxes(title_text=name, row=i + 1, col=1)

        fig.update_layout(height=200 * len(signalnames), width=900, showlegend=False, title_text="IMU Data")
        fig.update_xaxes(title_text="Sample", row=len(signalnames), col=1)

        fig.show()
