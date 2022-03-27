import ipdb
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn import datasets
from tqdm import tqdm
from sklearn.model_selection import train_test_split

np.random.seed(729)
lr = 0.01
epoch_num = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "torch_model.ckpt"


class Net(nn.Module):
    def __init__(self, input_size=2):
        super(Net, self).__init__()

        # layers
        self.fc1 = torch.nn.Linear(in_features=input_size, out_features=5)
        self.fc2 = torch.nn.Linear(in_features=5, out_features=5)
        self.fc3 = torch.nn.Linear(in_features=5, out_features=1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        # init weights
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.normal_(self.fc3.weight, std=0.01)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = self.fc3(out)
        out = out.squeeze(-1)

        return out


def generate_data():
    X, y = datasets.make_moons(n_samples=500, shuffle=True, noise=0.2)
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
    with torch.no_grad():
        predict = []
        for i, (x, y) in enumerate(zip(x_test, y_test)):
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)

            predict.append(model.forward(x).cpu().numpy().tolist())

        P, R = mertix(predict, y_test)
        return P, R


def main():

    X, y = generate_data()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=True)
    # show_data(x_test, y_test)

    model = Net(X.shape[1]).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr, momentum=0.9, weight_decay=0.01)
    criterion = nn.MSELoss().to(device)

    # Train
    P_, R_ = 0, 0
    for epoch in tqdm(range(epoch_num)):
        # print("Epoch[%d]" % epoch)
        P, R = test(model, x_test, y_test)

        for _, (x, y) in enumerate(zip(x_train, y_train)):
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)

            # Forward Pass
            output = model.forward(x)
            loss = criterion(output, y)

            # Backward and Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # if 0.5*P+0.5*R > 0.5*P_+0.5*R_:
        #     torch.save(model.state_dict(), model_path)
        #     P_ = P
        #     R_ = R

    # Test
    P, R = test(model, x_test, y_test)
    print(" Precise: %.2f%%   Recall: %.2f%%" % (P*100, R*100))


if __name__ == "__main__":
    main()
