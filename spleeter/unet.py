import torch
from torch import nn


def down_block(in_filters, out_filters):
    return nn.Sequential(
        nn.Conv2d(in_filters, out_filters, kernel_size=5,
                  stride=2, padding=2,
                  ),
        nn.BatchNorm2d(out_filters, track_running_stats=True),
        nn.LeakyReLU()
    )


def up_block(in_filters, out_filters, dropout=False):
    layers = [
        nn.ConvTranspose2d(in_filters, out_filters, kernel_size=5,
                           stride=2, padding=2, output_padding=1
                           ),
        nn.ReLU(0.2),
        nn.BatchNorm2d(out_filters, track_running_stats=True)
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))

    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, in_channels=2):
        super(UNet, self).__init__()
        self.down1 = down_block(in_channels, 16)
        self.down2 = down_block(16, 32)
        self.down3 = down_block(32, 64)
        self.down4 = down_block(64, 128)
        self.down5 = down_block(128, 256)
        self.down6 = down_block(256, 512)

        self.up1 = up_block(512, 256, dropout=True)
        self.up2 = up_block(512, 128, dropout=True)
        self.up3 = up_block(256, 64, dropout=True)
        self.up4 = up_block(128, 32)
        self.up5 = up_block(64, 16)
        self.up6 = up_block(32, 1)
        self.up7 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=4, dilation=2, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # import pdb; pdb.set_trace()
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6)
        u2 = self.up2(torch.cat([d5, u1], axis=1))
        u3 = self.up3(torch.cat([d4, u2], axis=1))
        u4 = self.up4(torch.cat([d3, u3], axis=1))
        u5 = self.up5(torch.cat([d2, u4], axis=1))
        u6 = self.up6(torch.cat([d1, u5], axis=1))
        u7 = self.up7(u6)

        return u7


if __name__ == '__main__':
    net = UNet(14)
    print(net(torch.rand(1, 14, 20, 48)).shape)
