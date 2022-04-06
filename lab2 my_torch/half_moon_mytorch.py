import utils
import mytorch
from mytorch import my_tensor
from mytorch.myglobal import graph

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import argparse


utils.setup_seed(729)


class Net(mytorch.Module):
    def __init__(self, input_size=2):
        super(Net, self).__init__()

        # layers
        self.fc1 = mytorch.Linear(in_features=input_size, out_features=6)
        self.fc2 = mytorch.Linear(in_features=6, out_features=6)
        self.fc3 = mytorch.Linear(in_features=6, out_features=1)
        self.relu = mytorch.Functional.ReLU()

        self.parameters = [self.fc1.w, self.fc2.w, self.fc3.w]

    def forward(self, x):
        super().forward(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
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

ii = 0
def plot_predictions(X, y, model, axes):
    global ii
    ii=ii+1
    p=100
    plt.scatter(X[:, 0], X[:, 1], c=y)
    x0s = np.linspace(axes[0], axes[1], p)
    x1s = np.linspace(axes[2], axes[3], p)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = np.array(model.forward(X)>0.5,dtype=int).reshape(p,p)
    graph.flush()
    # import pdb;pdb.set_trace()
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.title("Classifier")
    plt.savefig(f"./gif/{ii}.png")
    # return plt

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
    # Hyper-parameters
    parser = argparse.ArgumentParser(
        description="Opional arguments for training")
    parser.add_argument('-lr', '--learning-rate', default=1e-2, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr_decay', default=0.8, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_epochs', default=2000, type=int)
    parser.add_argument('--hidden_size', default=500, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--input_size', default=784, type=int)
    parser.add_argument('--optim', default='Adam', type=str)
    args = parser.parse_args()

    X, y = generate_data()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=True)
    show_data(x_test, y_test)

    model = Net(X.shape[1])
    if args.optim == 'Adam':
        optimizer = mytorch.Optim.Adam(
            module_params=model.parameters, lr=args.learning_rate)
    elif args.optim == 'SGD':
        optimizer = mytorch.Optim.SGD(
            module_params=model.parameters, lr=args.learning_rate, momentum=0.9)
    elif args.optim == 'Adagrad':
        optimizer = mytorch.Optim.Adagrad(
            module_params=model.parameters, lr=args.learning_rate)
    elif args.optim == 'RMSProp':
        optimizer = mytorch.Optim.RMSProp(
            module_params=model.parameters, lr=args.learning_rate)
    criterion = mytorch.Functional.MSELoss(n_classes=2)

    # Visualize
    # Start the server by: `python -m visdom.server`
    vis_env = 'HalfMoon_' + optimizer.__class__.__name__
    vis = utils.Visualizer(env=vis_env)

    # Train
    for epoch in range(args.num_epochs):

        for x, y in get_batches(x_train, y_train, batch_size=args.batch_size):

            x = my_tensor.from_array(x)
            y = my_tensor.from_array(np.array(y))

            # Forward Pass
            output = model.forward(x)
            loss = criterion(np.expand_dims(output, -1), y)

            # Backward and Optimize
            model.backward(loss.backward())
            optimizer.step()

        if epoch%100 == 0:
            plot_predictions(x_train,y_train,model,[-3,3,-3,3])
        if epoch % 20 == 1:
            # vis.matplot(ax)
            P, R = test(model, x_test, y_test)
            vis.plot('loss', loss.loss)
            vis.plot('Precise', P)
            vis.plot('Recall', R)
            print("Precise: %.2f%%   Recall: %.2f%%" % (P*100, R*100))

    # Test
    P, R = test(model, x_test, y_test)
    print(" Precise: %.2f%%   Recall: %.2f%%" % (P*100, R*100))


if __name__ == "__main__":
    main()
