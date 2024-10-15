import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class CNN(nn.Module):
    """Convolutional Neural Network.
    
    We provide a simple network with a Conv layer, followed by pooling,
    and a fully connected layer. Modify this to test different architectures,
    and hyperparameters, i.e. different number of layers, kernel size, feature
    dimensions etc.

    See https://pytorch.org/docs/stable/nn.html for a list of different layers
    in PyTorch.
    """

    def __init__(self):
        """Initialize layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 3) #output 48*48*12
        self.pool1 = nn.MaxPool2d(2, 2) #output 24*24*12
        self.conv2 = nn.Conv2d(12, 12, 3) #output 22*22*12
        self.pool2 = nn.MaxPool2d(2, 2) #output 11*11*12
        self.conv3 = nn.Conv2d(12, 12, 3) #output 9*9*12
        self.pool3 = nn.MaxPool2d(2, 2) #output 4*4*12
        self.fc1 = nn.Linear(12 * 4 * 4, 100) #output 100
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, 6)

        ''' # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 =
         nn.Linear(128 * 64 * 64, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 6)  # Assuming you want 6 output classes'''

        '''self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 9, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(9 * 3 * 3, 100)
        self.fc2 = nn.Linear(100, 30)'''

        '''self.conv1 = nn.Conv2d(3, 23, 3)
        self.conv2 = nn.Conv2d(23, 43, 3)
        self.conv3 = nn.Conv2d(43, 63, 4)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(63 * 4 * 4, 200)
        self.fc2 = nn.Linear(200, 30)
        self.fc3 = nn.Linear(30, 6)'''

    def forward(self, x):
        """Forward pass of network."""
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)

        return x

        '''x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = x.view(-1, 128 * 64 * 64)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x'''

        '''x = self.conv1(x)
        x = self.pool(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = F.relu(x)
        #x = self.conv3(x)
        #x = self.pool(x)
        #x = F.relu(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #x = F.relu(x)
        #x = self.fc3(x)
        return x'''
        
    def write_weights(self, fname):
        """ Store learned weights of CNN """
        torch.save(self.state_dict(), fname)

    def load_weights(self, fname):
        """
        Load weights from file in fname.
        The evaluation server will look for a file called checkpoint.pt
        """
        ckpt = torch.load(fname)
        self.load_state_dict(ckpt)


def get_loss_function():
    """Return the loss function to use during training. We use
       the Cross-Entropy loss for now.
    
    See https://pytorch.org/docs/stable/nn.html#loss-functions.
    """
    return nn.CrossEntropyLoss()


def get_optimizer(network, lr=0.001, momentum=0.9):
    """Return the optimizer to use during training.
    
    network specifies the PyTorch model.

    See https://pytorch.org/docs/stable/optim.html#how-to-use-an-optimizer.
    """

    # The fist parameter here specifies the list of parameters that are
    # learnt. In our case, we want to learn all the parameters in the network
    return optim.SGD(network.parameters(), lr=lr, momentum=momentum)
