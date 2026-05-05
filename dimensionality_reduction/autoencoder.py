import math
import torch


class Unsqueeze(torch.nn.Module):
    """
    Helper module to unsqueeze in a torch.nn.Sequential pipeline
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class AutoEncoder2DConv(torch.nn.Module):
    """
    AutoEncoder used to reduce dimension of J (neuron activations)
    Usage of 2D convolutions on data (num_neurons, num_bins)

    Encoder and decoder have to adapt to the shape of J (number of neurons and bins can change)
    """
    def __init__(self, input_shape, latent_dim):
        """
        input_shape: (1, H, W)
        latent_dim: int
        """
        super(AutoEncoder2DConv, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        assert len(input_shape) == 3, f"Image data of shape (1, height, width), only 1 channel"
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        print("Building encoder...")
        # First conv block
        in_channels, out_channels, kernel_size, stride, padding, dilation = 1, 64, 6, 2, 1, 1
        shape = self.input_shape
        conv_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        new_c = out_channels
        new_h = math.floor((shape[1] + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
        new_w = math.floor((shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
        shape = (new_c, new_h, new_w)
        relu_1 = torch.nn.ReLU(inplace=True)
        kernel_size, stride, padding, dilation = 2, 2, 1, 1
        pool_1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=padding)
        new_h = math.floor((shape[1] + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
        new_w = math.floor((shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
        shape = (new_c, new_h, new_w)

        # Second conv block
        in_channels, out_channels, kernel_size, stride, padding, dilation = 64, 32, 6, 2, 1, 1
        conv_2 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding)
        new_c = out_channels
        new_h = math.floor((shape[1] + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
        new_w = math.floor((shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
        shape = (new_c, new_h, new_w)
        relu_2 = torch.nn.ReLU(inplace=True)
        kernel_size, stride, padding, dilation = 2, 2, 0, 1
        pool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        new_h = math.floor((shape[1] + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
        new_w = math.floor((shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
        shape = (new_c, new_h, new_w)

        # Third conv block
        required_kernel_size = (shape[1], shape[2])
        in_channels, out_channels, kernel_size, stride, padding, dilation = 32, 16, required_kernel_size, 1, 0, 1
        conv_3 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding)
        new_c = out_channels
        new_h = math.floor((shape[1] + 2 * padding - dilation * (kernel_size[0] - 1) - 1) / float(stride) + 1)
        new_w = math.floor((shape[2] + 2 * padding - dilation * (kernel_size[1] - 1) - 1) / float(stride) + 1)
        shape = (new_c, new_h, new_w)
        relu_3 = torch.nn.ReLU(inplace=True)
        assert shape == (16, 1, 1)

        # Linear layer
        flatten = torch.nn.Flatten()
        linear = torch.nn.Linear(16, self.latent_dim)

        encoder = torch.nn.Sequential(
            # Conv block 1
            conv_1,
            relu_1,
            pool_1,
            # Conv block 2
            conv_2,
            relu_2,
            pool_2,
            # Conv block 3
            conv_3,
            relu_3,
            # Flatten into linear
            flatten,
            linear,
        )
        return encoder

    def build_decoder(self):
        print("Building decoder...")
        shape = (self.latent_dim)
        unsqueeze_1 = Unsqueeze(-1)
        unsqueeze_2 = Unsqueeze(-1)
        shape = (self.latent_dim, 1, 1)

        # Conv block 1
        scale_factor = 3
        upsample_1 = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
        shape = (self.latent_dim, shape[1]*scale_factor, shape[2]*scale_factor)
        in_channels, out_channels, kernel_size, stride, padding, dilation = self.latent_dim, 16, 3, 1, 2, 1
        conv_1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        new_c = out_channels
        new_h = math.floor((shape[1] + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
        new_w = math.floor((shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
        shape = (new_c, new_h, new_w)
        relu_1 = torch.nn.ReLU(inplace=True)

        # Conv block 2
        scale_factor = 3
        upsample_2 = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
        shape = (shape[0], shape[1] * scale_factor, shape[2] * scale_factor)
        in_channels, out_channels, kernel_size, stride, padding, dilation = 16, 8, 3, 1, 2, 1
        conv_2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        new_c = out_channels
        new_h = math.floor((shape[1] + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
        new_w = math.floor((shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
        shape = (new_c, new_h, new_w)
        relu_2 = torch.nn.ReLU(inplace=True)

        # Conv block 3
        size = (self.input_shape[1] // 2, self.input_shape[2] // 2)
        upsample_3 = torch.nn.Upsample(size=size, mode="nearest")
        shape = (shape[0], size[0], size[1])
        in_channels, out_channels, kernel_size, stride, padding, dilation = 8, 4, 3, 1, 2, 1
        conv_3 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        new_c = out_channels
        new_h = math.floor((shape[1] + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
        new_w = math.floor((shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
        shape = (new_c, new_h, new_w)
        relu_3 = torch.nn.ReLU(inplace=True)

        # Conv block 4
        size = (self.input_shape[1], self.input_shape[2])
        upsample_4 = torch.nn.Upsample(size=size, mode="nearest")
        shape = (shape[0], size[0], size[1])
        in_channels, out_channels, kernel_size, stride, padding, dilation = 4, 1, 1, 1, 0, 1
        conv_4 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        new_c = out_channels
        new_h = math.floor((shape[1] + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
        new_w = math.floor((shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
        shape = (new_c, new_h, new_w)
        assert shape == tuple(self.input_shape)

        sigmoid = torch.nn.Sigmoid()

        decoder = torch.nn.Sequential(
            # Unsqueeze
            unsqueeze_1,
            unsqueeze_2,
            # Conv block 1
            upsample_1,
            conv_1,
            relu_1,
            # Conv block 2
            upsample_2,
            conv_2,
            relu_2,
            # Conv block 3
            upsample_3,
            conv_3,
            relu_3,
            # Conv block 4
            upsample_4,
            conv_4,
            sigmoid,
        )
        return decoder


    def forward(self, x):
        """
        Assumption that image pixels are pre-scaled between 0 and 1 (because of sigmoid usage)
        """
        assert len(x.shape) == 4, "Expected shape for x: (n_batch, 1, H, W)"
        assert x.shape[1] == 1, "Expected one channel"

        x_latent = self.encoder(x)
        x_rebuilt = self.decoder(x_latent)

        return x_rebuilt

    def forward_debug(self, x):
        assert len(x.shape) == 4, "Expected shape for x: (n_batch, 1, H, W)"
        assert x.shape[1] == 1, "Expected one channel"

        print(f"Pre-forward shape: {x.shape}")

        for name, module in self.encoder.named_children():
            x = module(x)
            print(f"ENCODER post-module {name} shape: {x.shape}")

        for name, module in self.decoder.named_children():
            x = module(x)
            print(f"DECODER post-module {name} shape: {x.shape}")

        return x

    def get_latent_repr(self, x):
        assert len(x.shape) == 4, "Expected shape for x: (n_batch, 1, H, W)"
        assert x.shape[1] == 1, "Expected one channel"

        x_latent = self.encoder(x)
        return x_latent


