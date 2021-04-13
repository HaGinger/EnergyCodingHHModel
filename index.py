import numpy as np
from config import dt, E_K, E_Na, E_l


def NER(i_l, i_Na, i_K):
    Power = i_l * abs(E_l) - abs(i_Na) * abs(E_Na) + i_K * abs(E_K)
    Power *= 0.001
    Power = np.sum(Power, 0)
    Power_negative = np.float32(np.signbit(Power)) * Power
    Energy_negative = -dt * np.sum(Power_negative)
    Power_positive = np.float32(~np.signbit(Power)) * Power
    Energy_positive = dt * np.sum(Power_positive)

    return 100 * Energy_negative / (Energy_negative + Energy_positive)


def mNER(V_m):
    dV_m = (V_m[:, 0:-1] - V_m[:, 1:]) / dt
    dV_m = np.c_[dV_m, dV_m[:, -1]]
    Power = dV_m * .8 + (V_m + 68)
    Power = np.sum(Power, 0)
    Power_negative = np.float32(np.signbit(Power)) * Power
    Energy_negative = -dt * np.sum(Power_negative)
    Power_positive = np.float32(~np.signbit(Power)) * Power
    Energy_positive = dt * np.sum(Power_positive)

    return 100 * Energy_negative / (Energy_negative + Energy_positive)


def MCC(V_m):
    corr = np.corrcoef(V_m)
    row, col = np.diag_indices_from(corr)
    corr[row, col] = -np.ones((len(V_m),))
    mcc = corr.max(1).mean()

    return mcc