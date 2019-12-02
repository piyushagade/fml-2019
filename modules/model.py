from modules import imports as _
import torch as t
import torchvision
import torchvision.transforms.transforms as transforms

# 3-layer convolutional network
class ConvNet_1(t.nn.Module):
    def __init__(self):
        super(ConvNet_1, self).__init__()
        # Layer 1
        # The first step is to create some sequential layer objects within the class _init_ function. 
        # First, we create layer 1 (self.layer1) by creating a nn.Sequential object. 
        # This method allows us to create sequentially ordered layers in our network and is a handy way of 
        # creating a convolution + ReLU + pooling sequence. As can be observed, the first element in the 
        # sequential definition is the Conv2d nn.Module method – this method creates a set of convolutional filters. 
        # The first argument is the number of input channels – in this case, it is our single channel 
        # grayscale MNIST images, so the argument is 1. The second argument to Conv2d is the number of 
        # output channels – as shown in the model architecture diagram above, the first convolutional 
        # filter layer comprises of 32 channels, so this is the value of our second argument.
        self.layer1 = _.nn.Sequential(
            _.nn.Conv2d(1, 32, kernel_size=_.g.KERNEL_SIZE, stride=_.g.STRIDE, padding=2),
            _.nn.ReLU(),
            _.nn.MaxPool2d(kernel_size=2, stride=2))

        # Layer 2
        # Next, the second layer, self.layer2, is defined in the same way as the first layer. 
        # The only difference is that the input into the Conv2d function is now 32 channels, with an 
        # output of 64 channels. Using the same logic, and given the pooling down-sampling, the output 
        # from self.layer2 is 64 channels of 7 x 7 images.
        self.layer2 = _.nn.Sequential(
            _.nn.Conv2d(32, 64, kernel_size=_.g.KERNEL_SIZE, stride=_.g.STRIDE, padding=2),
            _.nn.ReLU(),
            _.nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = _.nn.Sequential(
            _.nn.Conv2d(64, 128, kernel_size=_.g.KERNEL_SIZE, stride=_.g.STRIDE, padding=2),
            _.nn.ReLU(),
            _.nn.MaxPool2d(kernel_size=2, stride=2))

        # Next, we specify a drop-out layer to avoid over-fitting in the model. Finally, 
        # two two fully connected layers are created. The first layer will be of size 8 x 8 x 64 nodes 
        # and will connect to the second layer of 1000 nodes. To create a fully connected layer in PyTorch, 
        # we use the nn.Linear method. The first argument to this method is the number of nodes in the 
        # layer, and the second argument is the number of nodes in the following layer.
        self.drop_out1 = _.nn.Dropout(p=_.g.DROPOUT_PROB)
        self.fc1 = _.nn.Linear((98 if _.g.KERNEL_SIZE == 5 else 162) * 64, 1000)
        self.drop_out2 = _.nn.Dropout(p=0.2)
        self.bn1 = _.nn.BatchNorm1d(1000)
        self.fc2 = _.nn.Linear(1000, _.g.NUM_CLASSES)

        
    # With this _init_ definition, the layer definitions have now been created. 
    # The next step is to define how the data flows through these layers when performing 
    # the forward pass through the network.
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out1(out)
        out = self.fc1(out)
        out = self.bn1(out)
        out = _.F.relu(out)
        out = self.drop_out2(out)
        out = self.fc2(out)
        return _.F.log_softmax(out, dim=1)

# Le-Net based architeture
class ConvNet_2(t.nn.Module):

    def __init__(self):
        super(ConvNet_2, self).__init__()

        self.layer1 = _.nn.Sequential(
            _.nn.Conv2d(1, 6, kernel_size=5),
            _.nn.ReLU(),
            _.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = _.nn.Sequential(
            _.nn.Conv2d(6, 16, kernel_size=5),
            _.nn.ReLU(),
            _.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = _.nn.Sequential(
            _.nn.Conv2d(16, 120, kernel_size=5),
            _.nn.ReLU())

        self.fc1 = _.nn.Linear(7680, 84)
        self.relu1 = _.nn.ReLU()
        self.fc2 = _.nn.Linear(84, _.g.NUM_CLASSES)
        
        print("Model initialized")

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.reshape(out.size(0), -1)
        out = _.F.relu(self.fc1(out))
        out = self.relu1(out)
        out = _.F.relu(self.fc2(out))
        return _.F.log_softmax(out, dim=-1)

# 2-layer convolutional network        
class ConvNet_3(t.nn.Module):
    def __init__(self):
        super(ConvNet_3, self).__init__()
        
        self.layer1 = _.nn.Sequential(
            _.nn.Conv2d(1, 32, kernel_size=_.g.KERNEL_SIZE, stride=_.g.STRIDE, padding=2),
            _.nn.ReLU(),
            _.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = _.nn.Sequential(
            _.nn.Conv2d(32, 64, kernel_size=_.g.KERNEL_SIZE, stride=_.g.STRIDE, padding=2),
            _.nn.ReLU(),
            _.nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.drop_out1 = _.nn.Dropout(p=_.g.DROPOUT_PROB)
        self.fc1 = _.nn.Linear((225 if _.g.KERNEL_SIZE == 5 else 256) * 64, 1000)
        self.drop_out2 = _.nn.Dropout(p=0.2)
        self.bn1 = _.nn.BatchNorm1d(1000)
        self.fc2 = _.nn.Linear(1000, _.g.NUM_CLASSES)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out1(out)
        out = self.fc1(out)
        out = self.bn1(out)
        out = _.F.relu(out)
        out = self.drop_out2(out)
        out = self.fc2(out)
        return _.F.log_softmax(out, dim=1)