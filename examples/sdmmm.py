import numpy as np
import numpy.random as npr
import scipy.stats as sps
import matplotlib.pyplot as plt
import dynmix as dm


k = 2
T = 10
n = 5

sim_eta = np.array(
    [[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.],
      [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.]],
     [[1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.],
      [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.]],
     [[1., .0], [.9, .1], [.8, .2], [.7, .3], [.6, .4],
      [.5, .5], [.4, .6], [.3, .7], [.2, .8], [.1, .9]],
     [[0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.],
      [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.]],
     [[0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.],
      [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.]]],
)

sim_var = np.array([.8, .6])
sim_var_w = np.array([.4, .3])

sim_theta = np.empty((k, T))
sim_theta[:,0] = [3., -5.]

npr.seed(10)
for t in range(1, T):
    sim_theta[:,t] = sim_theta[:,t-1] + npr.normal(0, np.sqrt(sim_var_w))

npr.seed(20)
y = np.empty((n, T))
for i in range(n):
    for t in range(T):
        y[i,t] = np.dot(sim_eta[i,t], npr.normal(sim_theta[:,t], np.sqrt(sim_var)))

plt.plot(sim_theta.T, 'k')
for i in range(n):
    plt.plot(y[i], 'g--')
plt.show()

eta, theta, phi, phi_w, U = dm.sdmmm.estimator(y, 2, np.array([5, -5]), 0.8)

plt.plot(sim_theta.T, 'k')
plt.plot(theta.T, 'r')
plt.show()

for i in range(n):
    plt.subplot(3, 2, i+1)
    plt.plot(sim_eta[i], 'k')
    plt.plot(eta[i], 'r')
plt.show()
