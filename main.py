# %% Imports

import os
from pathlib import Path

import numpy as np
import pandas as pd

import utils
from fp import forcePlate
from imu import IMU

SHARED_DATA_PATH = os.path.join(Path(os.environ.get("OneDrive")), "EVSFR")


# %% Load Data

data_path = os.path.join(SHARED_DATA_PATH, "pilotdata", "lf", "EVSFR_PilotLF_Trial2_ec_1ma_stim.txt")

# read in force plate calibrations (SN: 5571)
# // force plate a calibration data 
fpa_cal = pd.read_csv("fp_cal/fpa.csv", header=None).to_numpy()
offsets_a = np.array([0.001024, -0.000411, -0.042919])

# // force plate b calibration data (SN: 8032)
fpb_cal = pd.read_csv("fp_cal/fpb.csv", header=None).to_numpy()
offsets_b = np.array([0.0004318, -0.0006096, -0.0413766])

V0 = 10
gain = 4000

# read in experimental data
headernames = pd.read_csv("resources/headernames.csv", sep=",", header=0).columns.tolist()
df = pd.read_csv(data_path, sep=",", header=0, names=headernames)
# %%
fpa = forcePlate(
    fx=df["fpa_fx"].to_numpy(),
    fy=df["fpa_fy"].to_numpy(),
    fz=df["fpa_fz"].to_numpy(),
    mx=df["fpa_mx"].to_numpy(),
    my=df["fpa_my"].to_numpy(),
    mz=df["fpa_mz"].to_numpy(),
    v0=V0,
    gain=gain,
    calmatrix=np.linalg.inv(fpa_cal),
    offsets=offsets_a,
)

fpb = forcePlate(
    fx=df["fpb_fx"].to_numpy(),
    fy=df["fpb_fy"].to_numpy(),
    fz=df["fpb_fz"].to_numpy(),
    mx=df["fpb_mx"].to_numpy(),
    my=df["fpb_my"].to_numpy(),
    mz=df["fpb_mz"].to_numpy(),
    v0=V0,
    gain=gain,
    calmatrix=np.linalg.inv(fpb_cal),
    offsets=offsets_b,
)

imu = IMU(
    ax=df["imu_ax"].to_numpy(),
    ay=df["imu_ay"].to_numpy(),
    az=df["imu_az"].to_numpy(),
    gx=df["imu_gx"].to_numpy(),
    gy=df["imu_gy"].to_numpy(),
    gz=df["imu_gz"].to_numpy(),
    # correction_matrix=np.eye(3),
    # accel_bias=np.zeros(3),
    # gyro_bias=np.zeros(3),
    sample_rate=500.0,
    # sensor2body_matrix=np.eye(3),
)

start, end = utils.start_end_sync_idx(df["sync"].to_numpy())
# %% test data filtering
fpb_filt = utils.filter_signal(
    data=fpb.to_numpy(),
    cutoff=0.5,
    fs=100,
    type="low",
    order=2,
    # return_as="np.ndarray",
)
