import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import dynmix as dm


T = 36
y = np.array([np.cumsum(npr.randn(36)) + npr.randn(T) * 0.1]).T + 20
F = np.array([[1]])
G = np.array([[1]])
V = np.array([[.1]])
W = np.array([[1]])
a, R, f, Q, m, C = dm.dlm_filter(y, F, G, V, W)

plt.scatter(range(T), y)
plt.plot(range(T), f)
plt.plot(range(T), f + 1.96 * Q[:,0])
plt.plot(range(T), f - 1.96 * Q[:,0])
plt.show()
