import utils
import mytorch
from mytorch import my_tensor
from mytorch.myglobal import graph

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


utils.setup_seed(729)
lr = 1e-2
momentum = 0.9
epoch_num = 2000
batch_size = 16
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "mytorch_model.ckpt"


class Net(mytorch.Model):
    def __init__(self, input_size=2):
        super(Net, self).__init__()

        # layers
        self.fc1 = mytorch.Linear(in_features=input_size, out_features=6)
        self.fc2 = mytorch.Linear(in_features=6, out_features=6)
        self.fc3 = mytorch.Linear(in_features=6, out_features=1)
        self.relu1 = mytorch.Functional.ReLU()
        self.relu2 = mytorch.Functional.ReLU()

        self.parameters = [self.fc1.w, self.fc2.w, self.fc3.w]

    def forward(self, x):
        # import pdb;pdb.set_trace()
        super().forward(x)
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = out.squeeze(-1)
        return out


def generate_data():
    X, y = datasets.make_moons(n_samples=1000, shuffle=True, noise=0.2)
    return X, y


def show_data(X, y):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, marker=".")
    plt.show()


def test(model, x_test, y_test):
    """
    Return the precise and recall on the test dataset
    """
    predict = []
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        x = my_tensor.from_array(x)
        y = my_tensor.from_array(np.array(y))
        predict.append(model.forward(x))
    P, R = utils.mertix(predict, y_test)

    graph.flush()  # 清除算子图

    return P, R


def get_batches(inputs, targets, batch_size=16, shuffle=True):
    # inputs相当于X，targets相当于Y
    length = len(inputs)
    if shuffle:  # shuffle 随机打乱
        index = np.arange(length)
        np.random.shuffle(index)
    else:
        index = np.arange(length)
    start_idx = 0
    while (start_idx < length):
        end_idx = min(length, start_idx + batch_size)
        excerpt = index[start_idx:end_idx]
        X = inputs[excerpt]
        Y = targets[excerpt]
        Y = np.expand_dims(Y, -1)
        yield my_tensor.from_array(X), my_tensor.from_array(Y)
        start_idx += batch_size


def main():

    X, y = generate_data()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=True)
    # show_data(x_test, y_test)

    model = Net(X.shape[1])
    # optimizer = mytorch.Optim.SGD(
    #     module_params=model.parameters, lr=lr, momentum=momentum)
    # optimizer = mytorch.Optim.Adagrad(model.parameters, lr=lr)
    # optimizer = mytorch.Optim.RMSProp(model.parameters, lr=lr)
    optimizer = mytorch.Optim.Adam(model.parameters, lr=lr)
    criterion = mytorch.Functional.MSELoss(n_classes=2)

    # Visualize
    # Start the server by: `python -m visdom.server`
    vis = utils.Visualizer(env='HalfMoon_MyTorch')

    # Train
    for epoch in range(epoch_num):

        for x, y in get_batches(x_train, y_train, batch_size=batch_size):

            x = my_tensor.from_array(x)
            y = my_tensor.from_array(np.array(y))

            # Forward Pass
            output = model.forward(x)
            loss = criterion(output, y)

            # Backward and Optimize
            model.backward(loss.backward())
            optimizer.step()

        if epoch % 20 == 1:
            P, R = test(model, x_test, y_test)
            vis.plot('loss', loss.loss)
            vis.plot('Precise', P)
            vis.plot('Recall', R)
            print("Precise: %.2f%%   Recall: %.2f%%" % (P*100, R*100))

    # Test
    P, R = test(model, x_test, y_test)
    print(" Precise: %.2f%%   Recall: %.2f%%" % (P*100, R*100))


if __name__ == "__main__":
    # sft = mytorch.Functional.Softmax()
    # tsft = mytorch.Functional.Softmax()
    # x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    # h = 0.0001*np.ones_like(x)
    # y_plus_h = tsft.forward(x+h)
    # y_minus_h = tsft.forward(x-h)
    # dydx_2h = (y_plus_h - y_minus_h)/2*h
    # y = sft.forward(x)
    # dydx = sft.backward(x)
    main()
