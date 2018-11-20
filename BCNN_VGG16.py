import torch
import torch.nn as nn
from torchvision import models

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

        # Convolution and pooling layers of VGG-16.
        self.features = models.vgg16(pretrained=False).features

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
        assert X.size() == (N, 512, 7, 7)

        X = X.view(N, 512, 7**2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (7**2)  # Bilinear
        assert X.size() == (N, 512, 512)
        X = X.view(N, 512**2)
        X = torch.sqrt(X + 1e-5)                # L2
        X = nn.functional.normalize(X)

        return X

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.fc1_1 = nn.Linear(262144, 163)
        # self.fc1_2 = nn.Linear(4096, 2048)
        # self.fc1_3 = nn.Linear(2048, 163)

        self.fc2_1 = nn.Linear(262144, 5)
        # self.fc2_2 = nn.Linear(4096, 2048)
        # self.fc2_3 = nn.Linear(512, 5)

        self.fc3_1 = nn.Linear(262144, 12)
        # self.fc3_2 = nn.Linear(4096, 2048)
        # self.fc3_3 = nn.Linear(512, 12)

        self.fc4_1 = nn.Linear(262144, 10)
        # self.fc4_2 = nn.Linear(4096, 2048)
        # self.fc4_3 = nn.Linear(512, 10)

        self.fc5_1 = nn.Linear(262144, 15)
        # self.fc5_2 = nn.Linear(4096, 2048)
        # self.fc5_3 = nn.Linear(512, 15)

        self.fc6_1 = nn.Linear(262144, 6)
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
