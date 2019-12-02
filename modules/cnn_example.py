#https://algorithmia.com/blog/convolutional-neural-nets-in-pytorch

from torch.autograd import Variable
import torch.nn.functional as F
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as data_utils
import numpy as np
import torch.optim as optim
import time
from . import data as d
from . import preprocess

class SimpleCNN(torch.nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        #Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
        
        #64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(64, 10)
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))
        
        #Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)
        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 16, 16) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 18 * 16 *16)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return(x)

def create_loss_and_optimizer(net, learning_rate=0.001):
    #Loss function
    loss = torch.nn.CrossEntropyLoss()
    
    #Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    return(loss, optimizer)

def get_train_loader(batch_size):
    raw_data = d.load("./data/train_data.pkl")
    lables_data = d.load("./data/labels_data.npy")

    # Convert bool image data to int
    for i, item in enumerate(raw_data):
        raw_data[i] = preprocess.image(np.matrix(raw_data[i]))
        raw_data[i] = np.array(raw_data[i], dtype=np.int64)

    n_training_samples = len(raw_data)   

    train_labels_set = data_utils.TensorDataset(raw_data, lables_data)
    train_sampler = SubsetRandomSampler(np.arange(raw_data, dtype=np.int64))
    train_loader = torch.utils.data.DataLoader(train_labels_set, batch_size=batch_size, sampler=train_sampler)
    return train_loader

# def get_test_loader(batch_size):
    # 4
    # n_test_samples = 5000
    # test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, num_workers=2)

# def get_validation_loader(batch_size):
    # 128
    # n_val_samples = 5000
    # val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))
    # val_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=val_sampler, num_workers=2)

def train(net, batch_size, n_epochs, learning_rate):
    
    #Print all of the hyperparameters of the training iteration:
    print("Hyperparameters:")
    print("\tbatch_size=", batch_size)
    print("\tepochs=", n_epochs)
    print("\tlearning_rate=", learning_rate)
    print("\n")
    
    #Get training data
    train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)
    print("No. of batches: ", n_batches)
    
    #Create our loss and optimizer functions
    loss, optimizer = create_loss_and_optimizer(net, learning_rate)

    #Time for printing
    training_start_time = time.time()
    
    #Loop for n_epochs
    for epoch in range(1):
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        
        for i, data in enumerate(train_loader):
            
            # continue

            #Get inputs
            inputs, labels = data
            
            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            
            #Print statistics
            running_loss += loss_size.data[0]
            total_train_loss += loss_size.data[0]
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        # #At the end of the epoch, do a pass on the validation set
        # total_val_loss = 0
        # for inputs, labels in val_loader:
            
        #     #Wrap tensors in Variables
        #     inputs, labels = Variable(inputs), Variable(labels)
            
        #     #Forward pass
        #     val_outputs = net(inputs)
        #     val_loss_size = loss(val_outputs, labels)
        #     total_val_loss += val_loss_size.data[0]
            
        # print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        
    # print("Training finished, took {:.2f}s".format(time.time() - training_start_time))