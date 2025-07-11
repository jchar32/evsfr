from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class forcePlate:
    """Force Plate Data Processing Class
    This class handles the processing and transformation of force plate data, including calibration
    and offset adjustments for forces and moments.

    Parameters
    ----------
    fx, fy, fz : Union[List, np.ndarray, pd.Series, float]
        Force components in the x, y, and z directions
    mx, my, mz : Union[List, np.ndarray, pd.Series, float]
        Moment components around the x, y, and z axes
    v0 : float, optional
        Voltage excitation level, defaults to 10.0
    gain : float, optional
        Amplifier gain, defaults to 4000.0
    calmatrix : np.ndarray, optional
        Calibration matrix for converting raw signals to forces and moments.
        Defaults to a 6x6 identity matrix
    offsets : np.ndarray, optional
        Offset distances [x, y, z] to adjust moment calculation origin.
        Defaults to [1, 1, 1]

    Notes
    -----
    The class performs two main transformations during initialization:
    1. Applies the calibration matrix to convert raw signals to physical units
    2. Adjusts moments to represent values around the user-defined coordinate system

    Methods
    -------
    to_numpy()
        Convert data to a numpy array with shape (n, 6) where n is
        the number of samples and columns are [fx, fy, fz, mx, my, mz]
    """

    fx: Union[List, np.ndarray, pd.Series, float]
    fy: Union[List, np.ndarray, pd.Series, float]
    fz: Union[List, np.ndarray, pd.Series, float]
    mx: Union[List, np.ndarray, pd.Series, float]
    my: Union[List, np.ndarray, pd.Series, float]
    mz: Union[List, np.ndarray, pd.Series, float]
    v0: float = 10.0
    gain: float = 4000.0
    calmatrix: Optional[np.ndarray] = field(default_factory=lambda: np.eye(6))
    offsets: Optional[np.ndarray] = field(default_factory=lambda: np.ones(3))

    def __post_init__(self):
        if self.calmatrix is not None:
            """ appl sensitivity matrix to forces and moments"""
            calibrated_data = (self.calmatrix @ np.array([self.fx, self.fy, self.fz, self.mx, self.my, self.mz])).T / (1e-6 * self.v0 * self.gain)

            self.fx, self.fy, self.fz, self.mx, self.my, self.mz = calibrated_data.T

        if self.offsets is not None:
            """Place moments about user coordinate system."""
            self.mx = self.mx - (self.fy * self.offsets[2]) - (self.fz * self.offsets[1])
            self.my = self.my + (self.fx * self.offsets[2]) + (self.fz * self.offsets[0])
            self.mz = self.mz - (self.fx * self.offsets[1]) - (self.fy * self.offsets[0])
        self.cop_x, self.cop_y, self.cop_z, self.mzfree = self._cop()

    def _cop(self) -> np.ndarray:
        """Calculate the center of pressure (CoP) based on force plate data."""

        self.cop_x = (-(self.my + self.fx * self.offsets[-1]) / self.fz) + self.offsets[0]
        self.cop_y = ((self.mx + self.fy * self.offsets[-1]) / self.fz) + self.offsets[1]
        self.cop_z = np.zeros(self.fx.shape)
        self.mzfree = self.mz + self.fx * self.cop_y + self.fy * self.cop_x
        return np.array([self.cop_x, self.cop_y, self.cop_z, self.mzfree])

    def to_numpy(self) -> np.ndarray:
        return np.array([self.fx, self.fy, self.fz, self.mx, self.my, self.mz]).T

    def __repr__(self):
        return f"forcePlate(fx={self.fx}, fy={self.fy}, fz={self.fz}, mx={self.mx}, my={self.my}, mz={self.mz})"

    def __str__(self):
        return (
            f"Force Plate Data:\n"
            f"  Forces: fx={self.fx}, fy={self.fy}, fz={self.fz}\n"
            f"  Moments: mx={self.mx}, my={self.my}, mz={self.mz}\n"
            f"  Calibration Matrix:\n{self.calmatrix}\n"
            f"  Offsets: {self.offsets}"
        )
        return self.__str__()

    def plot(self, signalnames=None):
        """Plot the force plate data. if no signalnames are provided, defaults to ['fx', 'fy', 'fz', 'mx', 'my', 'mz']"""
        import plotly.graph_objects as go
        import plotly.subplots as sp

        if signalnames is None:
            signalnames = ["fx", "fy", "fz", "mx", "my", "mz"]

        data = self.to_numpy()

        # Create subplots
        fig = sp.make_subplots(rows=len(signalnames), cols=1, subplot_titles=signalnames, shared_xaxes=True)

        for i, name in enumerate(signalnames):
            fig.add_trace(go.Scatter(y=data[:, i], name=name), row=i + 1, col=1)
            fig.update_yaxes(title_text=name, row=i + 1, col=1)

        fig.update_layout(height=200 * len(signalnames), width=900, showlegend=False, title_text="Force Plate Data")
        fig.update_xaxes(title_text="Sample", row=len(signalnames), col=1)

        fig.show()
        return fig
