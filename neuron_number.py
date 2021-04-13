import hhmodel
import numpy as np
from config import dt, i_0, td, weight
from index import NER, mNER, MCC
import matplotlib.pyplot as plt


if __name__ == "__main__":
    time_span = 200
    i_t = 200
    test_num0 = 20
    test_num1 = 500
    test_interval = 10
    n_test = int((test_num1 - test_num0) / test_interval) + 1
    n_rep = 5
    neuron_number = np.linspace(test_num0, test_num1, n_test, dtype=np.int)
    totaltime = (test_num0 + test_num1) * n_test / 2
    testtime = test_num0

    alpha = np.zeros((n_test,))
    beta = np.zeros((n_test,))
    rho = np.zeros((n_test,))

    for i in range(n_test):
        tmp_ner = 0
        tmp_beta = 0
        tmp_mcc = 0

        N = neuron_number[i]
        n_stimulate = int(0.1 * N)
        i_ext = np.zeros((N, 1))
        i_ext[0:n_stimulate] = i_0
        I_ext = np.tile(i_ext, (1, int(i_t / dt) + 1))
        for j in range(n_rep):
            tau = np.random.uniform(td[0], td[1], (N,))
            w = np.random.uniform(0, weight, (N, N))
            V_m, i_l, i_Na, i_K, spike_count = hhmodel.network(N, time_span, w, tau, I_ext)
            tmp_ner += NER(i_l[n_stimulate:, :], i_Na[n_stimulate:, :], i_K[n_stimulate:, :])
            tmp_beta += mNER(V_m[n_stimulate:, :])
            tmp_mcc += MCC(V_m[n_stimulate:, :])

        alpha[i] = tmp_ner / n_rep
        beta[i] = tmp_beta / n_rep
        rho[i] = tmp_mcc / n_rep

        task_percent = int(100 * testtime / totaltime)
        testtime_1 = testtime + N + test_interval
        if int(20 * testtime_1 / totaltime) > task_percent / 5:
            print('task completed ' + str(task_percent) + '%')
        testtime = testtime_1

    fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex='all')
    fig.suptitle('correlation between neuron number and index')
    plt.xlabel('neuron number')
    ax1.plot(neuron_number, alpha, '.')
    ax1.set_ylabel('NER')
    ax2.plot(neuron_number, beta, '.')
    ax2.set_ylabel('mNER')
    ax3.plot(neuron_number, rho, '.')
    ax3.set_ylabel('MCC')

    fig.show()