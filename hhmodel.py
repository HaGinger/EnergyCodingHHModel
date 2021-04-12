from config import dt, C_m, g_l, g_Na, g_K, E_l, E_Na, E_K, V_r, V_threshold, N, i_0, time_delay
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def alpha_n(v_0):
    return 0.01 * (10 - v_0) / (np.exp((10 - v_0) / 10) - 1)


def beta_n(v_0):
    return 0.125 * np.exp(-v_0 / 80)


def alpha_m(v_0):
    return 0.1 * (25 - v_0) / (np.exp((25 - v_0) / 10) - 1)


def beta_m(v_0):
    return 4 * np.exp(-v_0 / 18)


def alpha_h(v_0):
    return 0.07 * np.exp(-v_0 / 20)


def beta_h(v_0):
    return 1 / (np.exp((30 - v_0) / 10) + 1)


def membrane_potential(V_m, n, m, h, I):
    i_l = -g_l * (E_l - V_m)
    i_Na = -g_Na * (m ** 3) * h * (E_Na - V_m)
    i_K = -g_K * (n ** 4) * (E_K - V_m)

    V_1 = V_m + (dt / C_m) * (-i_l - i_Na - i_K + I)

    v_0 = V_m - V_r

    n_1 = n + dt * (alpha_n(v_0) * (1 - n) - beta_n(v_0) * n)

    m_1 = m + dt * (alpha_m(v_0) * (1 - m) - beta_m(v_0) * m)

    h_1 = h + dt * (alpha_h(v_0) * (1 - h) - beta_h(v_0) * h)

    return V_1, n_1, m_1, h_1, i_l, i_Na, i_K


def network(t, weight, tau, I_ext=None):
    tn = int(t / dt)

    I_tmp = np.zeros((N, tn + 1))
    if I_ext.any():
        I_tmp[:, :I_ext.shape[1]] = I_ext
    I_ext = I_tmp

    V_m = V_r * np.ones((N, tn + 1))
    V_0 = V_m[:, 0] - V_r
    n = alpha_n(V_0) / (alpha_n(V_0) + beta_n(V_0))
    m = alpha_m(V_0) / (alpha_m(V_0) + beta_m(V_0))
    h = alpha_h(V_0) / (alpha_h(V_0) + beta_h(V_0))

    spike_count = np.zeros((N, tn + 1))
    i_l = np.zeros((N, tn + 1))
    i_Na = np.zeros((N, tn + 1))
    i_K = np.zeros((N, tn + 1))

    Q_0 = np.zeros((N,))
    tau_int = np.int32(tau / dt)
    tau_max = np.max(tau_int)
    I_in = np.zeros(N, )
    idx_row = np.tile(np.arange(N), (N, 1))
    for i in range(tn):
        idx = i - tau_int
        if i <= tau_max:
            idx1 = np.where(idx < 0)
            idx[idx1] = 0
            Q = np.float64(~np.signbit(V_m[idx_row, idx] - V_threshold))
            Q[idx1] = 0
        else:
            Q = np.float64(~np.signbit(V_m[idx_row, idx] - V_threshold))
        for j in range(N):
            I_in[j] = np.dot(weight[j, :], Q[j, :])
        # I_in = np.diag(np.dot(weight, np.transpose(Q)))

        Q_1 = np.float64(~np.signbit(V_m[np.arange(N), i] - V_threshold))
        dQ = Q_1 * (Q_1 - Q_0)
        # tau[np.where(dQ == 1)] = np.random.uniform(time_delay[0], time_delay[1], (int(np.sum(dQ)), ))
        spike_count[:, i] = dQ

        Q_0 = Q_1

        # I_in = I_in / N - 1e-5
        # I_in = I_in / (1 - I_in)
        # I_in = (sigmoid(I_in) - .5) * 50
        I_all = 450 * I_in / N + I_ext[:, i]
        # if i % 800 == 0:
        #     print('time ', int(i / 40))
        V_m[:, i + 1], n, m, h, i_l[:, i], i_Na[:, i], i_K[:, i] = membrane_potential(V_m[:, i], n, m, h, I_all)

    i_l[:, -1] = -g_l * (E_l - V_m[:, -1])
    i_Na[:, -1] = -g_Na * (m ** 3) * h * (E_Na - V_m[:, -1])
    i_K[:, -1] = -g_K * (n ** 4) * (E_K - V_m[:, -1])

    return V_m, i_l, i_Na, i_K, spike_count
