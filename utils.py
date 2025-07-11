from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import signal


def plot_signals(df: pd.DataFrame | np.ndarray, signalnames: list[str], frame_range: List[int] | None = None):
    """Plot specified signals from a DataFrame or numpy array.

    Parameters
    ----------
    df : pd.DataFrame | np.ndarray
        The data frame or numpy array containing the signals.
    signalnames : List[str]
        The names of the signals to plot.

    Returns
    -------
    go.Figure
        A plotly figure object containing the plotted signals. call `fig.show()` to display the figure.
    """
    fig = go.Figure()
    if frame_range is not None:
        start, end = frame_range
        df = df[start:end]
    if isinstance(df, pd.DataFrame):
        for signal in signalnames:
            fig.add_trace(
                go.Scatter(
                    y=df[signal],
                    mode="lines",
                    name=signal,
                )
            )
    elif isinstance(df, np.ndarray):
        for i, signal in enumerate(signalnames):
            if i == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    y=df[:, i],
                    mode="lines",
                    name=signal,
                )
            )
        return fig


def _get_filter_coefs(type: str, cutoff: float | List[float], fs: int, order: int = 2):
    """Get the coefficients for a Butterworth filter.

    Parameters
    ----------
    type : str
        The type of the filter. Can be "low", "high", "band", or "stop".
    cutoff : float or List[float]
        The cutoff frequency or frequencies.
    fs : int
        The sampling rate.
    order : int, optional
        The order of the filter. Default is 2.

    Returns
    -------
    tuple
        The numerator (b) and denominator (a) polynomials of the IIR filter.
    """

    b, a = signal.butter(
        order,
        cutoff,
        type,
        fs=fs,
    )
    return b, a


def filter_signal(
    data: pd.Series | pd.DataFrame | np.ndarray,
    cutoff: float | List[float],
    fs: int = 100,
    type: str = "low",
    order: int = 2,
    return_as: str = "same",
    new_col_names: List[str] | None = None,
):
    """
    Apply a digital filter to the data.

    Parameters
    ----------
    data : pd.Series | pd.DataFrame | np.ndarray
        The data to filter.
    cutoff : float or List[float]
        The cutoff frequency or frequencies.
    fs : int, optional
        The sampling rate. Default is 100.
    type : str, optional
        The type of the filter. Can be "low", "high", "band", or "stop". Default is "low".
    order : int, optional
        The order of the filter. Default is 2.
    return_as : str, optional
        The type of object to return. Can be "same", "ndarray", "pd.dataframe", or "pd.series". Default is "same".
    new_col_names : List[str] or None, optional
        Column names when returning as pd.DataFrame. Default is None.

    Returns
    -------
    np.ndarray or pd.DataFrame or pd.Series
        The filtered data in the format specified by return_as.
    """
    b, a = _get_filter_coefs(type=type, cutoff=cutoff, fs=fs, order=order)

    # assumes time axis is 0
    filtered_signal = signal.filtfilt(b, a, data, axis=0)  # returns as np.ndarray by default

    if return_as == "same":
        if isinstance(data, pd.DataFrame):
            filtered_signal = pd.DataFrame(filtered_signal, index=data.index, columns=data.columns)
        elif isinstance(data, pd.Series):
            filtered_signal = pd.Series(filtered_signal, index=data.index)
    elif return_as == "pd.dataframe":
        non_time_axis = np.argmin(filtered_signal.shape)
        col_name_list = [f"s{i}" for i in (range(filtered_signal.shape[non_time_axis]) if filtered_signal.ndim > 1 else range(1))]
        index = np.arange(0, filtered_signal.shape[0], 1)
        filtered_signal = pd.DataFrame(
            filtered_signal,
            index=index,
            columns=col_name_list if new_col_names is None else new_col_names,
        )

    return filtered_signal


def start_end_sync_idx(sync_signal: np.ndarray | pd.Series, threshold: float = None):
    """Find the start and end indices of a relevant time period in a sync signal.

    This function assumes the sync signal is a multisine with a square wave pulse
    with magnitude > 1V  at the start and end of the relevant time period.

    Parameters
    ----------
    sync_signal : np.ndarray
        The synchronization signal containing square wave pulses at start/end points.
    threshold : float, optional
        Threshold multiplier for standard deviation to detect edges.
        The actual threshold is calculated as threshold * std(diff_signal).
        By default 1, but internally overridden.

    Returns
    -------
    tuple
        (start_index, end_index).
    """
    diff_signal = np.abs(np.diff(sync_signal))
    threshold = float(np.std(diff_signal) * 50.0) if threshold is None else 1.0
    square_wave_edges = np.where(diff_signal > threshold)[0]
    # add 1 for due to diff function
    start = square_wave_edges[0] + 1
    end = square_wave_edges[-1] + 1
    return start, end
