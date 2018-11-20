import math

import torch
import torch.nn as nn
from torchvision import models

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BCNN(nn.Module):
    """B-CNN for CUB200.

    The B-CNN model is illustrated as follows.
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> bilinear pooling
    -> sqrt-normalize -> L2-normalize -> fc (200).
    The network accepts a 3*448*448 input, and the pool5 activation has shape
    512*28*28 since we down-sample 5 times.

    Attributes:
        features, torch.nn.Module: Convolution and pooling layers.
        fc, torch.nn.Module: 200.
    """
    def __init__(self):
        """Declare all needed layers."""
        nn.Module.__init__(self)

        # Convolution and pooling layers of ResNet-101
        # self.features = models.resnet101(pretrained=False)
        self.features = resnet101(pretrained=False)
        # self.features = nn.Sequential(*list(self.features.children())[:-2])  # Remove fc avg.
        # self.conv = nn.Conv2d(2048, 512, kernel_size=3, padding=1)
        # print(self.features)

    def forward(self, X):
        """Forward pass of the network.

        Args:
            X, torch.autograd.Variable of shape N*3*448*448.

        Returns:
            Score, torch.autograd.Variable of shape N*200.
        """
        N = X.size()[0]
        assert X.size() == (N, 3, 224, 224)

        X = self.features(X)
        # X = self.conv(X)
        assert X.size() == (N, 1024, 1, 1)

        X = X.view(N, 1024, 1**2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (1**2)  # Bilinear
        assert X.size() == (N, 1024, 1024)
        X = X.view(N, 1024**2)
        X = torch.sqrt(X + 1e-5)                # L2
        X = nn.functional.normalize(X)

        return X

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.fc1_1 = nn.Linear(1024 ** 2, 163)
        # self.fc1_2 = nn.Linear(4096, 163)
        # self.fc1_3 = nn.Linear(2048, 163)

        self.fc2_1 = nn.Linear(1024 ** 2, 5)
        # self.fc2_2 = nn.Linear(4096, 2048)
        # self.fc2_3 = nn.Linear(512, 5)

        self.fc3_1 = nn.Linear(1024 ** 2, 12)
        # self.fc3_2 = nn.Linear(4096, 2048)
        # self.fc3_3 = nn.Linear(512, 12)

        self.fc4_1 = nn.Linear(1024 ** 2, 10)
        # self.fc4_2 = nn.Linear(4096, 2048)
        # self.fc4_3 = nn.Linear(512, 10)

        self.fc5_1 = nn.Linear(1024 ** 2, 15)
        # self.fc5_2 = nn.Linear(4096, 2048)
        # self.fc5_3 = nn.Linear(512, 15)

        self.fc6_1 = nn.Linear(1024 ** 2, 6)
        # self.fc6_2 = nn.Linear(4096, 2048)
        # self.fc6_3 = nn.Linear(512, 6)

    def forward(self, x):
        x1 = self.fc1_1(x)
        # x1 = self.relu(x1)
        # x1 = self.dropout(x1)
        # x1 = self.fc1_2(x1)
        # x1 = self.relu(x1)
        # x1 = self.dropout(x1)
        # x1 = self.fc1_3(x1)

        x2 = self.fc2_1(x)
        # x2 = self.relu(x2)
        # x2 = self.dropout(x2)
        # x2 = self.fc2_2(x2)
        # x2 = self.relu(x2)
        # x2 = self.dropout(x2)
        # x2 = self.fc2_3(x2)

        x3 = self.fc3_1(x)
        # x3 = self.relu(x3)
        # x3 = self.dropout(x3)
        # x3 = self.fc3_2(x3)
        # x3 = self.relu(x3)
        # x3 = self.dropout(x3)
        # x3 = self.fc3_3(x3)

        x4 = self.fc4_1(x)
        # x4 = self.relu(x4)
        # x4 = self.dropout(x4)
        # x4 = self.fc4_2(x4)
        # x4 = self.relu(x4)
        # x4 = self.dropout(x4)
        # x4 = self.fc4_3(x4)

        x5 = self.fc5_1(x)
        # x5 = self.relu(x5)
        # x5 = self.dropout(x5)
        # x5 = self.fc5_2(x5)
        # x5 = self.relu(x5)
        # x5 = self.dropout(x5)
        # x5 = self.fc5_3(x5)

        x6 = self.fc6_1(x)
        # x6 = self.relu(x6)
        # x6 = self.dropout(x6)
        # x6 = self.fc6_2(x6)
        # x6 = self.relu(x6)
        # x6 = self.dropout(x6)
        # x6 = self.fc6_3(x6)

        return x1, x2, x3, x4, x5, x6

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = BCNN()
        self.classifier = Classifier()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x1, x2, x3, x4, x5, x6 = self.classifier(x)
        return x1, x2, x3, x4, x5, x6
