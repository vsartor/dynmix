import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import dynmix as dm


T = 36
y = np.array([np.cumsum(npr.randn(36)) + npr.randn(T) * 0.1]).T + 20
F = np.array([[1]])
G = np.array([[1]])
V = np.array([[1]])
W = np.array([[.1]])
a, R, f, Q, m, C = dm.dlm.filter(y, F, G, V, W)
s, S = dm.dlm.smoother(G, a, R, m, C)

plt.scatter(range(T), y)
plt.plot(range(T), f, 'r')
plt.plot(range(T), f + 1.96 * Q[:,0], 'r--')
plt.plot(range(T), f - 1.96 * Q[:,0], 'r--')
plt.show()

plt.scatter(range(T), y)
plt.plot(range(T), a, 'b')
plt.plot(range(T), a + 1.96 * R[:,0], 'b--')
plt.plot(range(T), a - 1.96 * R[:,0], 'b--')
plt.plot(range(T), m, 'r')
plt.plot(range(T), m + 1.96 * C[:,0], 'r--')
plt.plot(range(T), m - 1.96 * C[:,0], 'r--')
plt.plot(range(T), s, 'g')
plt.plot(range(T), s + 1.96 * S[:,0], 'g--')
plt.plot(range(T), s - 1.96 * S[:,0], 'g--')
plt.ylim((-15, 40))
plt.show()
