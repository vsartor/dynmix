import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import dynmix as dm


npr.seed(123)
y = npr.multinomial(1, [.1, .15, .15, .2, .4], 40)
c = dm.dirichlet_forward_filter(y, 0.9, 1.1 * np.ones(5))
estimators = dm.dirichlet_backwards_estimator(c, 0.95)
samples = []
for i in range(100):
    samples.append(dm.dirichlet_backwards_sampler(c, 0.9))


cols = ['r', 'g', 'g', 'b', 'k']
for i in range(5):
    plt.subplot(3, 2, i+1)
    plt.plot(estimators[:,i], cols[i])
    for sample in samples:
        plt.plot(sample[:,i], cols[i], alpha = 0.05)


plt.show()
