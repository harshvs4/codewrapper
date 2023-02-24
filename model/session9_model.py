import torch
import torch.nn as nn
import torch.nn.functional as F

dropout = 0.05


class Model9(nn.Module):
    def __init__(self, num_classes=10, n_features=48):
        super(Model9, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.ultimus = nn.Sequential(
            ultimus(),
            ultimus(),
            ultimus(),
            ultimus(),
        )

        self.final_fc = nn.Linear(
            in_features=n_features, out_features=num_classes, bias=False
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.gap(x)
        x = x.view(-1, 48)

        x = self.ultimus(x)

        out = self.final_fc(x)

        out = out.view(out.size(0), -1)
        return out


class ultimus(nn.Module):
    def __init__(self, in_features=48, out_features=8):
        super(ultimus, self).__init__()

        self.d_k = out_features
        self.sqrt_d_k = self.d_k**0.5

        self.fc_k = nn.Linear(in_features=in_features, out_features=out_features)
        self.fc_q = nn.Linear(in_features=in_features, out_features=out_features)
        self.fc_v = nn.Linear(in_features=in_features, out_features=out_features)

        self.fc_out = nn.Linear(in_features=out_features, out_features=in_features)

    def forward(self, x):
        x = self.fc_k(x)
        x = x.view(x.size(0), 1, -1)
        y = self.fc_q(x)
        y = y.view(y.size(0), 1, -1)
        z = self.fc_v(x)
        z = z.view(z.size(0), 1, -1)

        am = self._am(y, x)

        z = self._z(z, am)

        out = self.fc_out(z)

        return out

    def _am(self, y, x):
        am = (y.transpose(1, 2) @ x) / self.sqrt_d_k

        return F.softmax(am, dim=1)

    def _z(self, z, am):
        return z @ am
