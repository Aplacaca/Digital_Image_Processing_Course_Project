import utils
import mytorch
import numpy as np
from mytorch import my_tensor
from mytorch.myglobal import graph

import torch
import torchvision
import torchvision.transforms as transforms

# Random Seed
utils.setup_seed(729)

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocess
preprocess = False
img_size = 12 if preprocess else 28
transform = transforms.Compose([
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
]) if preprocess else transforms.ToTensor()

# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 1e-4

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transform)

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, shuffle=False)


def test(model):
    """
    Return the precise and recall on the test dataset
    """
    total = 0
    correct = 0
    for images, labels in test_loader:
        images = my_tensor.from_array(
            images.reshape(-1, img_size**2).numpy())
        labels = my_tensor.from_array(labels.reshape(-1, 1).numpy())

        predicted = model(images)
        graph.flush()
        # import pdb;pdb.set_trace()
        predicted = np.expand_dims(np.argmax(predicted, axis = -1), -1)
        total += labels.shape[0]
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the %d test images: %.4f%%' %
          (len(test_loader), 100*correct/total))

    graph.flush()  # 清除算子图

    return 100*correct/total


# Fully connected neural network with two hidden layers
class NeuralNet(mytorch.Model):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        # layers
        self.fc1 = mytorch.Linear(input_size, hidden_size)
        self.fc2 = mytorch.Linear(hidden_size, num_classes)
        self.relu = mytorch.Functional.ReLU()
        self.softmax = mytorch.Functional.Softmax()

        self.parameters = [self.fc1.w, self.fc2.w]

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)

        return out


model = NeuralNet(input_size, hidden_size, num_classes)

# #Loss and Optimizer
criterion = mytorch.Functional.CrossEntropy(n_classes=10)
# Loss_fn and Optimizer
# criterion = mytorch.Functional.MSELoss(n_classes=10)
# optimizer = mytorch.Optim.SGD(module_params=model.parameters, lr=learning_rate)
optimizer = mytorch.Optim.Adam(
    module_params=model.parameters, lr=learning_rate)

# Visualize
# Start the server by: `python -m visdom.server`
vis = utils.Visualizer(env='MNIST_MyTorch')

# Train
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = my_tensor.from_array(
            images.reshape(-1, img_size**2).numpy())
        labels = my_tensor.from_array(labels.reshape(-1, 1).numpy())

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and Optimize
        model.backward(loss.backward())
        optimizer.step()

        if (i+1) % 100 == 0:
            # vis.plot('loss', loss.loss)
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %
                  (epoch + 1, num_epochs, i+1, total_step, loss.loss))

    test_accuracy = test(model)
    vis.plot('test_accuracy', test_accuracy)

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
