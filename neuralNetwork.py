from turtle import forward
import torch as T
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        self.convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(720)))
        self.convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(720)))

        self.fc1 = nn.Linear(self.convh * self.convw * 32, 120)
        self.fc1 = nn.Linear(120,64)
        self.fc2 = nn.Linear(64,48)
        self.fc3 = nn.Linear(48,24)
        self.fc4 = nn.Linear(24,9)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,self.convw*self.convh*32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x 

