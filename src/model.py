import torch
import torch.nn as nn


def conv_bn_relu(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SimplePoseNet(nn.Module):
    """شبکه ساده برای تشخیص ۱۷ مفصل انسان (COCO format)"""
    def __init__(self, num_keypoints=17):
        super().__init__()
        self.num_keypoints = num_keypoints

        # Encoder
        self.conv1 = conv_bn_relu(3, 32)
        self.conv2 = conv_bn_relu(32, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = conv_bn_relu(64, 64)
        self.conv4 = conv_bn_relu(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = conv_bn_relu(128, 128)
        self.conv6 = conv_bn_relu(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Decoder (upsampling)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7 = conv_bn_relu(256, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8 = conv_bn_relu(128, 64)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv9 = conv_bn_relu(64, 32)

        # Heatmap output: (B, num_keypoints, H, W)
        self.heatmap_out = nn.Conv2d(32, num_keypoints, 1)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)  # H/2

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)  # H/4

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)  # H/8

        # Decoder
        x = self.up1(x)
        x = self.conv7(x)

        x = self.up2(x)
        x = self.conv8(x)

        x = self.up3(x)
        x = self.conv9(x)

        heatmaps = self.heatmap_out(x)
        return heatmaps
