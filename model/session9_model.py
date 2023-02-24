import torch
import torch.nn as nn
import torch.nn.functional as F

dropout = 0.05


class TransformerModel(nn.Module):
    def __init__(self, num_classes=10, n_features=48):
        super(TransformerModel, self).__init__()

        """
        CONVOLUTION BLOCK

           InputDimensions        NIn   RFIn   K  P  S  Jin  Jout  RFOut  NOut       OutputDimensions          Notes
        [batch_size,  3, 32, 32]   32     1    3  1  1   1    1      3     32    [batch_size, 16, 32, 32]    Normal Conv
        [batch_size, 16, 32, 32]   32     3    3  1  1   1    1      5     32    [batch_size, 32, 32, 32]    Normal Conv
        [batch_size, 32, 32, 32]   32     5    3  1  1   1    1      7     32    [batch_size, 48, 32, 32]    Normal Conv
        """
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

        """
        GAP 1

        Input     = [batch_size, 48, 32, 32]
        Ouput (X) = [batch_size, 48,  1,  1]
        """
        self.gap = nn.AdaptiveAvgPool2d(1)

        """
        Ultimus Blocks

        Input     = [batch_size, 48]
        Ouput     = [batch_size, 48]
        """
        self.ultimus_blocks = nn.Sequential(
            UltimusBlock(),
            UltimusBlock(),
            UltimusBlock(),
            UltimusBlock(),
        )

        """
        Final FC Layer

        Input     = [batch_size, 1, 48]
        Ouput     = [batch_size, 1, 10]
        """
        self.final_fc = nn.Linear(
            in_features=n_features, out_features=num_classes, bias=False
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.gap(x)

        # Reshape [batch_size, 48, 1, 1] to [batch_size, 48]
        x = x.view(-1, 48)

        x = self.ultimus_blocks(x)

        out = self.final_fc(x)

        # Reshape to [batch_size, 10]
        out = out.view(out.size(0), -1)
        return out


class UltimusBlock(nn.Module):
    def __init__(self, in_features=48, out_features=8):
        super(UltimusBlock, self).__init__()

        self.d_k = out_features
        self.sqrt_d_k = self.d_k**0.5

        self.fc_k = nn.Linear(in_features=in_features, out_features=out_features)
        self.fc_q = nn.Linear(in_features=in_features, out_features=out_features)
        self.fc_v = nn.Linear(in_features=in_features, out_features=out_features)

        self.fc_out = nn.Linear(in_features=out_features, out_features=in_features)

    def forward(self, x):
        """
        |====================================|
        |  Variable  |        Shape          |
        |====================================|
        |     x      |  [batch_size,    48]) |
        |            |                       |
        |     k      |  [batch_size, 1,  8]) |
        |     q      |  [batch_size, 1,  8]) |
        |     v      |  [batch_size, 1,  8]) |
        |            |                       |
        |     am     |  [batch_size, 8,  8]) |
        |     z      |  [batch_size, 1,  8]) |
        |            |                       |
        |     out    |  [batch_size, 1, 48]) |
        =====================================|
        """
        k = self.fc_k(x)
        k = k.view(k.size(0), 1, -1)
        q = self.fc_q(x)
        q = q.view(q.size(0), 1, -1)
        v = self.fc_v(x)
        v = v.view(v.size(0), 1, -1)

        am = self._am(q, k)

        z = self._z(v, am)

        out = self.fc_out(z)

        return out

    def _am(self, q, k):
        am = (q.transpose(1, 2) @ k) / self.sqrt_d_k

        return F.softmax(am, dim=1)

    def _z(self, v, am):
        return v @ am
