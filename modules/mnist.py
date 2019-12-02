
from torch import nn, Tensor
import torch.utils.data as data_utils
import torch as t
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms.transforms as transforms
import torchvision


# Hyperparameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

DATA_PATH = './MNISTData'
MODEL_STORE_PATH = './pytorch_models'

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def train():
    
    # transforms to apply to the data
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
    test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = ConvNet()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            if i == 0 and epoch == 0:
                print(images.shape)

            # Run the forward pass
            outputs = model(images)
            
            if i == 0 and epoch == 0:
                print(outputs.shape, labels.shape)

        #     loss = criterion(outputs, labels)
        #     loss_list.append(loss.item())

        #     # Backprop and perform Adam optimisation
        #     optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # # Track the accuracy
        # total = labels.size(0)
        # _, predicted = t.max(outputs.data, 1)
        # correct = (predicted == labels).sum().item()
        # acc_list.append(correct / total)

        # if (i + 1) % 100 == 0:
        #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
        #           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
        #                   (correct / total) * 100))