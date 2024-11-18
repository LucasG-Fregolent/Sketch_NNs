import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, batch_norm=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNetModel(nn.Module):
    def __init__(self, 
                 num_blocks, 
                 conv_out_channels, 
                 final_dense_layers, 
                 num_classes, 
                 batch_norm=True, 
                 dropout_rate=0.0, 
                 pooling_type="AvgPool"):
        super(ResNetModel, self).__init__()

        self.in_channels = conv_out_channels[0]
        self.conv1 = nn.Conv2d(3, conv_out_channels[0], kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(conv_out_channels[0]) if batch_norm else nn.Identity()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, stride=2, padding=1) if pooling_type == "MaxPool" else nn.AvgPool2d(3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(conv_out_channels[0], conv_out_channels[1], num_blocks[0], stride=1, batch_norm=batch_norm)
        self.layer2 = self._make_layer(conv_out_channels[1], conv_out_channels[2], num_blocks[1], stride=2, batch_norm=batch_norm)
        self.layer3 = self._make_layer(conv_out_channels[2], conv_out_channels[3], num_blocks[2], stride=2, batch_norm=batch_norm)
        self.layer4 = self._make_layer(conv_out_channels[3], conv_out_channels[4], num_blocks[3], stride=2, batch_norm=batch_norm)

        # Set conv_output_size to the number of output channels from the last convolutional layer
        conv_output_size = conv_out_channels[-1]

        # Dense layers
        dense_layers = []
        in_features = conv_output_size
        for out_features in final_dense_layers:
            dense_layers.append(nn.Linear(in_features, out_features))
            dense_layers.append(nn.ReLU())
            if dropout_rate > 0:
                dense_layers.append(nn.Dropout(dropout_rate))
            in_features = out_features

        # Final output layer
        dense_layers.append(nn.Linear(in_features, num_classes))
        self.fc = nn.Sequential(*dense_layers)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1, batch_norm=True):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
            )

        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample, batch_norm))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, batch_norm=batch_norm))
        
        return nn.Sequential(*layers)

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
