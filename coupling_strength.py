import hhmodel
import numpy as np
from config import N, dt, i_0, td
from index import NER, mNER, MCC
import matplotlib.pyplot as plt


if __name__ == "__main__":
    t = 200
    i_t = 200
    test_w0 = 0.01
    test_w1 = 1
    n_test = int(test_w1 / test_w0)
    n_rep = 5
    coupling_strength = np.linspace(test_w0, test_w1, n_test)
    tau = np.random.uniform(td[0], td[1], (N, ))

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
        for j in range(n_rep):
            w = np.random.uniform(0, coupling_strength[i], (N, N))
            V_m, i_l, i_Na, i_K, spike_count = hhmodel.network(N, t, w, tau, I_ext)
            tmp_ner += NER(i_l[n_stimulate:, :], i_Na[n_stimulate:, :], i_K[n_stimulate:, :])
            tmp_beta += mNER(V_m[n_stimulate:, :])
            tmp_mcc += MCC(V_m[n_stimulate:, :])

        alpha[i] = tmp_ner / n_rep
        beta[i] = tmp_beta / n_rep
        rho[i] = tmp_mcc / n_rep

    fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex='all')
    fig.suptitle('correlation between coupling strength and index')
    plt.xlabel('coupling strength')
    ax1.plot(coupling_strength, alpha, '.')
    ax1.set_ylabel('NER')
    ax2.plot(coupling_strength, beta, '.')
    ax2.set_ylabel('mNER')
    ax3.plot(coupling_strength, rho, '.')
    ax3.set_ylabel('MCC')

    fig.show()
