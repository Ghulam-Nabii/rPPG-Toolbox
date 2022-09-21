import torch
import torch.nn as nn
# from learning.utils import PeriodicFeatsToSpectrum, SpectrumToFreq


__all__ = ['resnet3d_18']


class Conv3DSimple(nn.Conv3d):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):
        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride


class BasicBlock(nn.Module):

    __constants__ = ['downsample']
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem"""
    def __init__(self):
        super(BasicStem, self).__init__(
            nn.Conv3d(3, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))


class VideoResNet(nn.Module):

    def __init__(self, block, conv_makers, layers, stem, num_frames=200, zero_init_residual=False):
        """Generic resnet video generator.
        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_frames (int, optional): Length of input video
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = 32

        self.stem = stem()
        self.layer1 = self._make_layer(block, conv_makers[0], 32, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 64, layers[1], stride=1)
        self.layer3 = self._make_layer(block, conv_makers[2], 128, layers[2], stride=1)
        self.layer4 = self._make_layer(block, conv_makers[3], 128, layers[3], stride=1)

        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)

        self.conv5 = nn.Sequential(
            nn.Conv3d(128, 1, (1, 1, 1)),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(num_frames, 1)
        # TODO: now hardcoded with Countix-specific parameters
        # self.spec_layer = PeriodicFeatsToSpectrum(num_frames, 30, 0, 21)
        # self.count_layer = SpectrumToFreq(0, 21, 10)

        # init weights
        self._initialize_weights()
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)    # [bs, 32, 200, 48, 48]

        x = self.layer1(x)
        x = self.pool1(x)   # [bs, 32, 200, 24, 24]

        x = self.layer2(x)
        x = self.pool2(x)   # [bs, 64, 200, 12, 12]

        x = self.layer3(x)
        x = self.pool3(x)   # [bs, 128, 200, 6, 6]

        x = self.layer4(x)
        x = self.pool4(x)   # [bs, 128, 200, 3, 3]

        x = self.conv5(x)   # [bs, 1, 200, 3, 3]
        x = x.squeeze(1)
        x = self.avgpool(x)
        encoding = x.view(x.size(0), -1)
        # x = self.spec_layer(encoding)
        # x = self.count_layer(x)
        return encoding

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _video_resnet(**kwargs):
    model = VideoResNet(**kwargs)
    return model


def SimPerConv3d(**kwargs):
    return _video_resnet(block=BasicBlock,
                         conv_makers=[Conv3DSimple] * 4,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)


if __name__ == '__main__':
    model = SimPerConv3d()

    dummy_inputs = torch.randn(2, 3, 20, 36, 36)
    print(f"ResNet3D model size: {sum(p.numel() for p in model.parameters()) / 1e6:.4f}M")
    outputs = model.forward(dummy_inputs)
    print(f"Outputs:\t{outputs.size()}")
