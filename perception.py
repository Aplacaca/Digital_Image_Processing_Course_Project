import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
num_observations = 500

x1 = np.random.multivariate_normal([0, 0],[[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4],[[1, .75],[.75, 1]], num_observations)

X = np.vstack((x1, x2)).astype(np.float32)
Y = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

class Perception():
    def __init__(self,x, num_neuron) -> None:
        weights = np.zeros(num_neuron)

    def f_x(self,x):
        np.sign(x*self.weights)




f1 = plt.figure(1)
plt.scatter(x1,x2)
plt.savefig('./dots.png')
