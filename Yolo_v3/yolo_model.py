##################################################             YOLO         ##########################################################################

# ------------------------------------------------       Defining the Model           ----------------------------------------------------------------

import torch
import torch.nn as nn

# Tuples are defined as (kernel_size, filters, stride, padding), all the layers are designed exactly as described in the original paper.

Yolo_configuration = [
    (7, 64, 2, 3),        # (7, 64, 2, 3) indicates 7*7 kernel, 64 output filter, stride of 2 and 3 padding
    "M",                  # maxpooling layer
    (3, 192, 1, 1),       # 3*3 kernel, 192 output filters, stride of 1 and 1 padding
    "M",                  # maxpooling layer with stride 2x2 and kernel 2x2
    (1, 128, 1, 0),       # 1*1 kernel, 128 output filters, stride of 1 and 0 padding
    (3, 256, 1, 1),       # 3*3 kernel, 192 output filters, stride of 1 and 1 padding
    (1, 256, 1, 0),       # 3*3 kernel, 192 output filters, stride of 1 and 1 padding
    (3, 512, 1, 1),       # 3*3 kernel, 192 output filters, stride of 1 and 1 padding
    "M",                  # maxpooling layer
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],    # 4 layers of 1*1 kernels, 256 output filters, 1 stride and 3*3 kernels, 512 output filters, 1 stride and 1 padding
    (1, 512, 1, 0),                         # 1*1 kernels, 512 output filters, 1 stride
    (3, 1024, 1, 1),                        # 3*3 kernels, 1024 output filters, 1 stride and 1 padding
    "M",                                    # maxpooling layer
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],   # 2 layers of 1*1 kernels, 512 output filters, 1 stride and 3*3 kernels, 1024 output filters, 1 stride and 1 padding
    (3, 1024, 1, 1),                        # 3*3 kernels, 1024 output filters, 1 stride and 1 padding
    (3, 1024, 2, 1),                        # 3*3 kernels, 1024 output filters, 2 stride and 1 padding
    (3, 1024, 1, 1),                        # 3*3 kernels, 1024 output filters, 1 stride and 1 padding
    (3, 1024, 1, 1),                        # 3*3 kernels, 1024 output filters, 1 stride and 1 padding
]

# ---------------------------------------  Defining the CNN Blocks   -------------------------------------------------------------
class CNNLayer(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        """
        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels (filters).
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride for the convolution.
            padding (int): Padding for the convolution.
        """
        super(CNNLayer, self).__init__()
        # Convolutional layer with no bias (bias=False for BatchNorm compatibility)
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        # Batch normalization layer to stabilize training
        self.batchnorm = nn.BatchNorm2d(output_channels)
        # Leaky ReLU activation function
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output after applying Conv2D, BatchNorm, and LeakyReLU.
        """
        x = self.conv(x)         # convolution
        x = self.batchnorm(x)    # batch normalization
        x = self.leakyrelu(x)    # LeakyReLU activation
        return x

# -----------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------  Defining the YOLO architecture   -------------------------------------------------------------

class YoloVersion1(nn.Module):
    def __init__(self, input_channels=3, **kwargs):
        """
        Initializing the YOLO1 model.

        Args:
            input_channels: Number of input channels (3 - RGB images).
            **kwargs: Additional parameters for fully connected layer creation.
        """
        super(YoloVersion1, self).__init__()
        self.architecture = Yolo_configuration    # Yolo configuration
        self.input_channels = input_channels       # Initial input channels
        self.darknet = self.create_cnn_layers(self.architecture)   # Convolutional layers
        self.fcs = self.create_fully_connected_layers(**kwargs)      # Fully connected layers

    def forward(self, x):
        """
        Forward pass through the YOLO.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output after passing through the model.
        """
        x = self.darknet(x)  # Passing through convolutional layers
        x = torch.flatten(x, start_dim=1)  # Flattening for fully connected layers
        return self.fcs(x)  # Passing through fully connected layers

    def create_cnn_layers(self, architecture):
        """
        Creating the convolutional layers of the YOLO model.

        Args:
            architecture: Configuration of the convolutional layers.

        Returns:
            nn.Sequential: A sequential container of the convolutional layers.
        """
        layers = []
        input_channels = self.input_channels

        for layer in architecture:
            if isinstance(layer, tuple):  # Standard convolutional layer
                kernel_size, filters, stride, padding = layer
                layers.append(
                    CNNLayer(
                        input_channels, filters,
                        kernel_size=kernel_size, stride=stride, padding=padding
                    )
                )
                input_channels = filters  # Updating the input channels for the next layer

            elif layer == "M":  # Max pooling layer
                layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

            elif isinstance(layer, list):  # Repeated blocks of convolutional layers
                conv1, conv2, num_repeats = layer
                for _ in range(num_repeats):
                    # First convolution in the repeated block
                    layers.append(
                        CNNLayer(
                            input_channels, conv1[1],
                            kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]
                        )
                    )
                    # Second convolution in the repeated block
                    layers.append(
                        CNNLayer(
                            conv1[1], conv2[1],
                            kernel_size=conv2[0], stride=conv2[2], padding=conv2[3]
                        )
                    )
                    input_channels = conv2[1]  # Updating input channels after each block

        return nn.Sequential(*layers)  # sequential container

    def create_fully_connected_layers(self, split_size, num_boxes, num_classes):
        """
        The fully connected layers of the YOLO model.

        Args:
            **kwargs: Parameters for the fully connected layers (e.g., output size).

        Returns:
            nn.Sequential: A sequential container of the fully connected layers.
        """
        S, B, C = split_size, num_boxes, num_classes

        # Below code is as defined in the original paper
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )