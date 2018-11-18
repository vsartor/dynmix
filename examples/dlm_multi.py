import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import dynmix as dm

npr.seed(123456)
n = [4, 6, 2, 5, 7, 2, 4, 2, 3, 5, 7, 2, 9, 4, 2]
T = len(n)
F = np.array([1., 0.])
G = np.array([[1., 1.], [0., 1.]])
V = 1.
W = [[.1, 0.], [0, .05]]
Y = []
theta = np.empty((T, 2))
theta[0] = np.array([50, 0.7])
Y.append(npr.randn(n[0]) * V + theta[0,0])

for t in range(1, T):
    theta[t] = np.dot(G, theta[t-1]) + npr.multivariate_normal(np.zeros(2), W)
    Y.append(npr.randn(n[t]) * V + theta[t,0])

a, R, m, C = dm.dlm_multi_filter(Y, F, G, V, W, np.array([40, 0]), np.eye(2))
s, S = dm.dlm_smoother(G, a, R, m, C)

plt.plot(range(T), a[:,0], 'b')
plt.plot(range(T), a[:,0] + 1.96 * R[:,0,0], 'b--')
plt.plot(range(T), a[:,0] - 1.96 * R[:,0,0], 'b--')
plt.plot(range(T), m[:,0], 'r')
plt.plot(range(T), m[:,0] + 1.96 * C[:,0,0], 'r--')
plt.plot(range(T), m[:,0] - 1.96 * C[:,0,0], 'r--')
plt.plot(range(T), s[:,0], 'g')
plt.plot(range(T), s[:,0] + 1.96 * S[:,0,0], 'g--')
plt.plot(range(T), s[:,0] - 1.96 * S[:,0,0], 'g--')
for t in range(1, T):
    plt.scatter(np.repeat(t, n[t]), Y[t], c='k')
plt.show()
