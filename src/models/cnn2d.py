"""B2: space x time DAS context with anisotropic physical-axis kernels."""

from torch import nn


class CNN2D(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 7), stride=(1, 1), padding=(1, 3)),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d((2, 2)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, patch):
        return self.classifier(self.features(patch))
