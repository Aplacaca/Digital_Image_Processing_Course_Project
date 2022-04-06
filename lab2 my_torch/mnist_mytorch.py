import utils
import mytorch
import numpy as np
from mytorch import my_tensor
from mytorch.myglobal import graph

import torch
import torchvision
import torchvision.transforms as transforms
import argparse

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
parser = argparse.ArgumentParser(description="Opional arguments for training")
parser.add_argument('-lr', '--learning-rate', default=5e-3, type=float)
parser.add_argument('--lr_decay', default=0.8, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--hidden_size', default=500, type=int)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--input_size', default=784, type=int)
parser.add_argument('--optim', default='Adam', type=str)
args = parser.parse_args()

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transform)

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size, shuffle=False)


def test(model):
    """
    Return the accuracy on the test dataset
    """
    total = 0
    correct = 0
    for images, labels in test_loader:
        images = my_tensor.from_array(
            images.reshape(-1, img_size**2).numpy())
        labels = my_tensor.from_array(labels.reshape(-1, 1).numpy())

        predicted = model(images)
        graph.flush()

        predicted = np.expand_dims(np.argmax(predicted, axis=-1), -1)
        total += labels.shape[0]
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the %d test images: %.4f%%' %
          (len(test_loader)*len(images), 100*correct/total))

    graph.flush()  # 清除算子图，避免计入梯度

    return 100*correct/total


# Fully connected neural network with two hidden layers
class NeuralNet(mytorch.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        # layers
        self.fc1 = mytorch.Linear(input_size, hidden_size)
        self.fc2 = mytorch.Linear(hidden_size, num_classes)
        self.relu = mytorch.Functional.ReLU()
        self.softmax = mytorch.Functional.Softmax()

        self.parameters = [self.fc1.w, self.fc2.w]

    def __call__(self, x: np.ndarray) -> np.ndarray:

        return self.forward(x)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)

        return out


# Model
model = NeuralNet(args.input_size, args.hidden_size, args.num_classes)

# Loss_fn and Optimizer
criterion = mytorch.Functional.CrossEntropy(n_classes=10)
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

# Visualize
# Start the server by: `python -m visdom.server`
vis_env = 'MNIST_MyTorch_' + optimizer.__class__.__name__ + \
    "_lr-%.1e" % args.learning_rate
vis = utils.Visualizer(env=vis_env)

# Train
total_step = len(train_loader)
for epoch in range(args.num_epochs):
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
            vis.plot('loss', loss.loss)
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %
                  (epoch + 1, args.num_epochs, i+1, total_step, loss.loss))

    # lr decay
    optimizer.lr *= args.lr_decay
    print('learning rate decay to %.4e' % optimizer.lr)

    # Test
    test_accuracy = test(model)
    vis.plot('test_accuracy', test_accuracy)

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
