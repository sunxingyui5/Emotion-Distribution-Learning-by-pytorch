import torch
from torch import nn


# 卷积神经网络
class CNN(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=10,
                stride=1,
                padding=0
            ),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),

            nn.Conv2d(6, 16, kernel_size=15, stride=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),

            nn.Conv2d(16, 24, kernel_size=15, stride=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(24 * 16 * 16, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
            nn.Softmax(1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)  # 扁平化
        return self.classifier(x)


class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 自定损失函数
class WeightedCombinationLoss(nn.Module):
    def __init__(self, lmd=0.8):
        super(WeightedCombinationLoss, self).__init__()
        self.sdl_loss_function = nn.KLDivLoss(reduction='batchmean')
        self.cls_loss_function = nn.CrossEntropyLoss()
        self.lmd = lmd

    def forward(self, input, target):
        return self.lmd * self.sdl_loss_function.forward(input, target) + \
               (1 - self.lmd) * self.cls_loss_function.forward(input, torch.argmax(target, dim=1))
