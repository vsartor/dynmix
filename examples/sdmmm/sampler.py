import numpy as np
import numpy.random as npr
import scipy.stats as sps
import matplotlib.pyplot as plt
import dynmix as dm


k = 2
T = 10
n = 9

sim_eta = np.array(
    [[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.],
      [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.]],
     [[1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.],
      [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.]],
     [[1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.],
      [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.]],
     [[1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.],
      [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.]],
     [[1., .0], [.9, .1], [.8, .2], [.7, .3], [.6, .4],
      [.5, .5], [.4, .6], [.3, .7], [.2, .8], [.1, .9]],
     [[0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.],
      [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.]],
     [[0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.],
      [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.]],
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

plt.figure(figsize = (8, 2.5))
#plt.plot(sim_theta.T, 'k')
for i in range(n):
    plt.plot(y[i], 'g--')
plt.show()

eta, Z, theta, phi, phi_w = dm.sdmmm.sampler(y, 2, 0.2, np.array([5., -5.]))

plt.figure(figsize = (8, 2.5))
eta_est = eta.mean(axis=0)
for i in range(n):
    if i != 4:
        plt.plot(y[i], color = (eta_est[i,t,0],0,1-eta_est[i,t,0]))
for t in range(T):
    plt.scatter(t, y[4,t], color = (eta_est[4,t,0],0,1-eta_est[4,t,0]))
plt.show()

plt.figure(figsize = (8, 2.5))
plt.plot(sim_theta.T, 'k')
plt.plot(theta.mean(axis=0).T, 'r')
plt.plot(np.quantile(theta, 0.95, axis=0).T, 'r', alpha = 0.4)
plt.plot(np.quantile(theta, 0.05, axis=0).T, 'r', alpha = 0.4)
plt.show()

eta_gibbs = eta
plt.figure(figsize = (8, 2.5))
plt.plot(sim_eta[4,:,0], 'k')
plt.plot(np.quantile(eta_gibbs, 0.5, axis=0)[4,:,0], 'r')
plt.plot(np.quantile(eta_gibbs, 0.95, axis=0)[4,:,0], 'r', alpha=0.4)
plt.plot(np.quantile(eta_gibbs, 0.05, axis=0)[4,:,0], 'r', alpha=0.4)
plt.show()
