import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self,
                 conv_layers,
                 conv_out_channels,
                 kernel_size,
                 stride,
                 final_dense_layers,
                 num_classes,
                 batch_norm,
                 dropout_rate,
                 pooling_type,
                 pool_kernel_size):
        super(CNNModel, self).__init__()
        self._initialize_weights()

        layers = []
        in_channels = 3

        # Convolutional layers
        for i in range(conv_layers):
            out_channels = conv_out_channels[i] if i < len(conv_out_channels) else conv_out_channels[-1]
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1))
            
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
                
            layers.append(nn.ReLU())
            
            if pooling_type == "MaxPool":
                layers.append(nn.MaxPool2d(pool_kernel_size))
            elif pooling_type == "AvgPool":
                layers.append(nn.AvgPool2d(pool_kernel_size))
            
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        # Calculate the output size after convolutional layers to initialize dense layers
        sample_input = torch.zeros(1, 3, 32, 32)
        sample_output = self.conv(sample_input)
        conv_output_size = sample_output.view(-1).shape[0]

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

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for x in self.modules():
            if isinstance(x, nn.Linear) or isinstance(x, nn.Conv2d):
                torch.nn.init.xavier_uniform_(x.weight.data)
                if x.bias is not None:
                    x.bias.data.zero_()
