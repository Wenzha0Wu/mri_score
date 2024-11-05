import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes, num_regions):
        super(LeNet5, self).__init__()
        # enc
        self.layer1 = nn.Sequential(
            nn.Conv2d(5, 8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(),)
        self.subsampel1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),)
        self.subsampel2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # enc dense
        self.L1 = nn.Linear(576, 256)
        self.relu = nn.ReLU()

        # cls dense
        self.L2 = nn.Linear(256, 64)
        self.relu1 = nn.ReLU()
        self.L3 = nn.Linear(64, num_classes)
        
        # reg dense
        self.L5 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU()
        self.L6 = nn.Linear(64, num_regions)

        # dec dense
        self.L4 = nn.Linear(256, 576)
        self.relu3 = nn.ReLU()

        # dec 
        self.layer3 = nn.Sequential(
              nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
              nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=3),
              nn.BatchNorm2d(8),
              nn.ReLU(),
            )

        self.layer4 = nn.Sequential(
              nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
              nn.Conv2d(8, 5, kernel_size=3, stride=1, padding=1),
              nn.BatchNorm2d(5),
              nn.ReLU(),
            )

    def forward(self, x):
        out = self.layer1(x)
        out = self.subsampel1(out)
        #print(out.shape)
        out = self.layer2(out)
        out = self.subsampel2(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        bottleneck = self.L1(out)
        bottleneck = self.relu(bottleneck)
        
        img = self.L4(bottleneck)
        img = self.relu3(img)
        img = img.view(-1, 16, 6, 6)
        img = self.layer3(img)
        #print(img.shape)
        img = self.layer4(img)
        #print(img.shape)

        cls = self.L2(bottleneck)
        cls = self.relu1(cls)
        cls = self.L3(cls)

        reg = self.L5(bottleneck)
        reg = self.relu2(reg)
        reg = self.L6(reg)

        return cls, reg, img
