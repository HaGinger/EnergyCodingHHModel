import hhmodel
import numpy as np
from config import N, dt, i_0, E_K, E_Na, E_l, td, weight
import matplotlib.pyplot as plt

if __name__ == "__main__":
    time_span = 450
    i_t = 450
    timeline = np.linspace(0, time_span, int(time_span / dt) + 1)
    n_stimulate = int(0.1 * N)

    w = np.random.uniform(0, weight, (N, N))
    tau = np.random.uniform(td[0], td[1], (N, N))
    i_ext = np.zeros((N, 1))
    i_ext[0:n_stimulate] = i_0
    I_ext = np.tile(i_ext, (1, int(i_t / dt) + 1))

    V_m, i_l, i_Na, i_K, spike_count = hhmodel.network(time_span, w, tau, I_ext)

    Power = (i_l * abs(E_l) - abs(i_Na) * abs(E_Na) + i_K * abs(E_K)) * 0.001
    Power = np.sum(Power, 0)
    Power_negative = np.float32(np.signbit(Power)) * Power
    Energy_negative = -dt * np.sum(Power_negative)
    Power_positive = np.float32(~np.signbit(Power)) * Power
    Energy_positive = dt * np.sum(Power_positive)
    NER = Energy_negative / (Energy_positive + Energy_negative)

    idx = np.where(spike_count == 1)

    fig = plt.figure()
    fig1 = fig.add_subplot(121)
    fig1.scatter(dt * idx[1], idx[0] + 1, s=1)
    fig1.set_title('spike')
    fig1.set_ylabel('neuron')

    fig2 = fig.add_subplot(122)
    fig2.plot(timeline, Power)
    fig2.set_title('power')
    fig2.set_ylabel('power')
    plt.show()
