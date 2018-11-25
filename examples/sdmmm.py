import numpy as np
import numpy.random as npr
import scipy.stats as sps
import matplotlib.pyplot as plt
from dynmix import dlm_multi_filter, dlm_smoother, dirichlet_backwards_estimator, dirichlet_forward_filter

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

#plt.plot(sim_theta.T, 'k')
#for i in range(n):
#    plt.plot(y[i], 'g--')
#plt.show()


def sdmmm_estimator(y, k, init_level, delta = 0.9, numit = 100):
    '''
    Simple Dynamic Membership Mixture Model. A level-based mixture
    model with a first-order polynomial DLM for each cluster. This
    variation of the function attempts to perform point estimates.

    Args:
        y: An array with each row being the time-series from one
            observational unit.
        k: Number of clusters.
        init_level: The initial level of each cluster.
        delta: The universal discount factor to be used for all the
            units.
        numit: Number of iterations for the algorithm to run.

    Returns:
        The evolution of the estimates for each parameter.
    '''

    n, T = y.shape

    #-- Initialize the parameters
    #TODO: Allow more user initializations

    # DLM parameters
    phi = np.ones(k)
    phi_w = np.ones(k)
    theta = np.tile(init_level, (T, 1)).T

    # Dirichlet Process parameters
    eta = np.tile(npr.dirichlet(np.ones(k)), (n, T, 1))
    Z = np.tile(npr.multinomial(1, np.ones(k)/k), (n, T, 1))

    # Likelihood
    U = np.empty(numit)

    #-- Constants

    F = G = np.array([[1]])

    #-- Iterative updates of parameter estimates based on means

    for l in range(numit):
        print(f'sdmmm_estimator: iteration {l} out of {numit}')

        # Update membership dummy parameters for each unit
        for i in range(n):
            for t in range(T):
                probs = sps.norm.pdf(y[i,t], theta[:,t], 1. / np.sqrt(phi))
                params = eta[i,t] * probs
                Z[i,t] = np.zeros(k)
                Z[i,t,params.argmax()] = 1

        # Update Dirichlet states for each unit
        for i in range(n):
            c = dirichlet_forward_filter(Z[i], delta, np.ones(k) * 0.1)
            eta[i] = dirichlet_backwards_estimator(c, delta)

        # Update DLM states and parameters for each cluster
        for j in range(k):
            # Create observation list for multi_dlm
            YJ = []
            for t in range(T):
                mask = Z[:,t,j] == 1
                YJ.append(y[mask,t])

            # Update states
            V = np.array([[1 / np.sqrt(phi[j])]])
            W = np.array([[1 / np.sqrt(phi_w[j])]])
            filters = dlm_multi_filter(YJ, F, G, V, W)
            theta[j] = dlm_smoother(G, *filters)[0][:,0]

            # Update parameters
            observation_ssq = 0
            for t in range(T):
                observation_ssq += np.sum((YJ[t] - theta[j,t])**2)
            phi[j] = n / observation_ssq
            phi_w[j] = np.mean((theta[j,:-1] - theta[j,1:])**2)

        # Update likelihood
        # U[l] = sdmmm_likelihood(y, Z, eta, delta, theta, phi, phi_w)
        U[l] = 0.

    return eta, theta, phi, phi_w, U

eta, theta, phi, phi_w, U = sdmmm_estimator(y, 2, np.array([5, -5]), delta = 0.8)

plt.plot(sim_theta.T, 'k')
plt.plot(theta.T, 'r')
plt.show()

for i in range(n):
    plt.subplot(3, 2, i+1)
    plt.plot(sim_eta[i], 'k')
    plt.plot(eta[i], 'r')
plt.show()
