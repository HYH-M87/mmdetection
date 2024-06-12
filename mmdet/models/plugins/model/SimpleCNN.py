from torch import nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(245, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

from torch import nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一层卷积层，输入通道为1，输出通道为5，卷积核大小为3x3，填充为1
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=1)
        # 第二层卷积层，输入通道为5，输出通道为5，卷积核大小为3x3，填充为1
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1)
        # 最大池化层，池化核大小为2x2
        self.maxpool = nn.MaxPool2d(kernel_size=2) 
        # 将特征展平为一维向量
        self.flatten = nn.Flatten()
        # 全连接层，输入特征维度为245，输出维度为10
        self.fc = nn.Linear(245, 10)

    def forward(self, x):
        # 第一层卷积，使用ReLU激活函数
        x = F.relu(self.conv1(x)) # 28*28 5
        # 最大池化
        x = self.maxpool(x) # 14*14 5
        # 第二层卷积，使用ReLU激活函数
        x = F.relu(self.conv2(x)) # 14*14 5
        # 再次最大池化
        x = self.maxpool(x) # 7*7 5
        # 将特征展平
        x = self.flatten(x) # 7*7*5=245
        # 全连接层
        x = self.fc(x)
        # 使用对数softmax作为输出激活函数
        x = F.log_softmax(x, dim=1)
        return x

