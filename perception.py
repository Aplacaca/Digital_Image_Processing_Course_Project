import numpy as np
import matplotlib.pyplot as plt
import ipdb

np.random.seed(729)
num_observations = 500
lr1 = 1e-4
MAX_STEP = 200


class perception():
    def __init__(self, lr, dataset):
        self.lr = lr
        self.X, self.y = dataset
        self.X_train = np.hstack((np.ones((num_observations*2, 1)), self.X))
        self.w = np.random.rand(self.X_train.shape[1]).reshape(
            (1, self.X_train.shape[1]))
        self.loss_history = []

    def fit(self):
        count = 0
        while count < MAX_STEP:
            for i in range(self.X_train.shape[0]):
                pred = self.w.dot(self.X_train[i].T)
                self.w = self.w+self.lr*(self.y[0][i]-pred)*self.X_train[i]
            count = count + 1
            self.loss_history.append(
                self.loss(self.w.dot(self.X_train[i].T)))

    def mertix(self):
        """P、R
        Args:
            pred: output label of model
        """
        pred = self.w.dot(self.X_train.T) > 0.5

        # P = TP / (TP + FP)
        index_ = pred == 1
        TP = (self.y[index_] == 1).sum()
        P = TP / index_.sum()

        # R = TP / (TP + FN)
        R = TP / self.y.sum()

        # acc = (TP + TN) / (TP + FP + TN + FN)
        acc = ((pred == self.y).sum())/pred.shape[1]

        print("Precise: %.2f%%, Recall: %.2f%%, acc: %.2f%%" %
              (100*P, 100*R, 100*acc))
        return P, R, acc

    def loss(self, pred):
        return 0.5*np.average(pow(pred-self.y, 2))


def dataset():
    # 生成两个二元正态分布矩阵作为两类样本，各500个
    x1 = np.random.multivariate_normal(
        [0, 0], [[1, .75], [.75, 1]], num_observations)
    x2 = np.random.multivariate_normal(
        [1, 4], [[1, .75], [.75, 1]], num_observations)

    X = np.vstack((x1, x2)).astype(np.float32)
    Y = np.hstack((np.zeros(num_observations), np.ones(num_observations)))
    return X, Y.reshape((1, Y.shape[0]))


def show_clf(x, y, w):
    fig, ax = plt.subplots()
    ax.scatter(x[:, 1], x[:, 2], c=y, alpha=0.9, edgecolors='black')

    xmin, xmax = x[:, 1].min()-1, x[:, 1].max()+1
    xx = np.linspace(xmin, xmax, num_observations)
    yy = (0.5 - w[:, 0]-xx*w[:, 1]) / w[:, 2]
    ax.plot(xx, yy)
    plt.show()


def show_loss(loss):
    xx = range(MAX_STEP)
    yy = loss
    plt.plot(xx, yy)
    plt.show()


if __name__ == "__main__":
    clf = perception(lr1, dataset())
    clf.fit()
    clf.mertix()

    show_clf(clf.X_train, clf.y, clf.w)
    show_loss(clf.loss_history)
