import utils
import torch
import mytorch
from mytorch import my_tensor
import torchvision
import torchvision.transforms as transforms


# Random Seed
utils.setup_seed(729)

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 1e-4

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor())

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
        images = my_tensor.from_array(images.reshape(-1, 28*28).numpy())
        labels = my_tensor.from_array(labels.reshape(-1, 1).numpy())

        predicted = model(images)
        total += labels.shape[0]
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the %d test images: %.4f%%' %
          (len(test_loader), 100*correct/total))


# Fully connected neural network with two hidden layers
class NeuralNet(mytorch.Model):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        # layers
        self.fc1 = mytorch.Linear(input_size, hidden_size)
        self.fc2 = mytorch.Linear(hidden_size, hidden_size)
        self.fc3 = mytorch.Linear(hidden_size, num_classes)

        self.parameters = [self.fc1.w, self.fc2.w, self.fc3.w]

    def forward(self, x):
        out = self.fc1(x)
        out = mytorch.Functional.ReLU()(out)
        out = self.fc2(out)
        out = mytorch.Functional.ReLU()(out)
        out = self.fc3(out)
        out = mytorch.Functional.argmax()(out)

        return out


model = NeuralNet(input_size, hidden_size, num_classes)

#Loss and Optimizer
criterion = mytorch.Functional.MSELoss(n_classes=10)
# optimizer = mytorch.Optim.SGD(module_params=model.parameters, lr=learning_rate)
optimizer = mytorch.Optim.Adam(
    module_params=model.parameters, lr=learning_rate)

# Train
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = my_tensor.from_array(images.reshape(-1, 28*28).numpy())
        labels = my_tensor.from_array(labels.reshape(-1, 1).numpy())

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and Optimize
        model.backward(loss.backward())
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %
                  (epoch + 1, num_epochs, i+1, total_step, loss.loss))

    test(model)

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
