import hhmodel
import numpy as np
from config import N, dt, E_K, E_Na, E_l, i_0, time_delay
import matplotlib.pyplot as plt


def NER_1(V_m):
    dV_m = (V_m[:, 0:-1] - V_m[:, 1:]) / dt
    dV_m = np.c_[dV_m, dV_m[:, -1]]
    Power = dV_m * .8 + (V_m + 68)
    Power = np.sum(Power, 0)
    Power_negative = np.float32(np.signbit(Power)) * Power
    Energy_negative = -dt * np.sum(Power_negative)
    Power_positive = np.float32(~np.signbit(Power)) * Power
    Energy_positive = dt * np.sum(Power_positive)

    return 100 * Energy_negative / (Energy_negative + Energy_positive)


def NER(i_l, i_Na, i_K):
    Power = i_l * abs(E_l) - abs(i_Na) * abs(E_Na) + i_K * abs(E_K)
    Power *= 0.001
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


if __name__ == "__main__":
    t = 100
    i_t = 100
    test_t0 = 0.01
    test_t1 = 1
    n_test = int(test_t1 / test_t0)
    coupling_strength = np.linspace(test_t0, test_t1, n_test)
    tau = np.random.uniform(time_delay[0], time_delay[1], (N, N))

    n_stimulate = int(0.1 * N)
    i_ext = np.zeros((N, 1))
    i_ext[0:n_stimulate] = i_0
    I_ext = np.tile(i_ext, (1, int(i_t / dt) + 1))

    alpha = np.zeros((n_test,))
    beta = np.zeros((n_test,))
    rho = np.zeros((n_test,))

    for i in range(n_test):
        tmp_ner = 0
        tmp_beta = 0
        tmp_mcc = 0
        task_percent = int(100 * i / n_test)
        if int(20 * (i + 1) / n_test) > task_percent / 5:
            print('task completed ' + str(task_percent) + '%')
        for j in range(3):
            w = np.random.uniform(0, coupling_strength[i], (N, N))
            V_m, i_l, i_Na, i_K, spike_count = hhmodel.network(t, w, tau, I_ext)
            tmp_ner += NER(i_l[n_stimulate:, :], i_Na[n_stimulate:, :], i_K[n_stimulate:, :])
            tmp_beta += NER_1(V_m[n_stimulate:, :])
            tmp_mcc += MCC(V_m[n_stimulate:, :])

        alpha[i] = tmp_ner / 3
        beta[i] = tmp_beta / 3
        rho[i] = tmp_mcc / 3

    fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True)
    fig.suptitle('correlation between coupling strength and index')
    plt.xlabel('coupling strength')
    ax1.plot(coupling_strength, alpha, '.')
    ax1.set_ylabel('NER')
    ax2.plot(coupling_strength, beta, '.')
    ax2.set_ylabel('NER_1')
    ax3.plot(coupling_strength, rho, '.')
    ax3.set_ylabel('MCC')

    fig.show()
