import mytorch
from mytorch import my_tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from utils import setup_seed
from sklearn.model_selection import train_test_split


setup_seed(729)
lr = 0.001
momentum = 0.9
epoch_num = 500
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "mytorch_model.ckpt"


class Net(mytorch.Model):
    def __init__(self, input_size=2):
        super(Net, self).__init__()

        # layers
        self.fc1 = mytorch.Linear(in_features=input_size, out_features=6)
        self.fc2 = mytorch.Linear(in_features=6, out_features=6)
        self.fc3 = mytorch.Linear(in_features=6, out_features=1)
        self.relu = mytorch.Functional.ReLU()

        self.parameters = [self.fc1.w, self.fc2.w, self.fc3.w]

    def forward(self, x):
        # import pdb;pdb.set_trace()
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


def mertix(predict, y):
    index = (np.array(predict) > 0.5).reshape(-1)
    if index.sum() != 0:
        P = y[index].sum() / index.sum()
        R = y[index].sum() / y.sum()
    else:
        P = 0
        R = 0

    return P, R


def test(model, x_test, y_test):
    """
    Return the precise and recall on the test dataset
    """

    predict = []
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        x = my_tensor.from_array(x)
        y = my_tensor.from_array(np.array(y))
        predict.append(model.forward(x))
    P, R = mertix(predict, y_test)
    return P, R


def main():

    X, y = generate_data()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=True)
    show_data(x_test, y_test)

    model = Net(X.shape[1])
    optimizer = mytorch.Optim.SGD(lr=lr, momentum=momentum)
    criterion = mytorch.Functional.MSELoss(n_classes=2)

    # Train
    for epoch in range(epoch_num):
        for _, (x, y) in enumerate(zip(x_train, y_train)):
            x = my_tensor.from_array(x)
            y = my_tensor.from_array(np.array(y))

            # Forward Pass
            output = model.forward(x)
            loss = criterion(output, y)

            # Backward and Optimize
            model.backward(loss.backward())

            optimizer.step(model.parameters)

        if epoch % 20 == 1:
            P, R = test(model, x_test, y_test)
            print("Precise: %.2f%%   Recall: %.2f%%" % (P*100, R*100))

    # Test
    P, R = test(model, x_test, y_test)
    print(" Precise: %.2f%%   Recall: %.2f%%" % (P*100, R*100))


if __name__ == "__main__":
    main()
