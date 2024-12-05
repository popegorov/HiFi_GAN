from src.model.utils import get_padding, init_weights, ReLU_SLOPE
from torch import nn
from torch.nn.utils import weight_norm

import torch
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(
        self, 
        channels, 
        kernel_size=3, 
        dilation=(1, 3)
        ):
        super().__init__()

        self.convs = nn.Sequential()
        for i, d in enumerate(dilation):
            self.convs.add_module(
                f"Conv {i}",
                weight_norm(
                    nn.Conv1d(
                        in_channels=channels, 
                        out_channels=channels, 
                        kernel_size=kernel_size, 
                        stride=1, 
                        padding=get_padding(kernel_size, d), 
                        dilation=d
                    )
                )
            )
       
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, ReLU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x


class Generator(nn.Module):
    def __init__(
        self,
        in_channel=512,
        r_kernel_size=[3, 7, 11],
        r_dilation=[[1, 3], [1, 3], [1, 3]],
        u_rate=[8, 8, 2, 2],
        u_kernel_size=[16, 16, 4, 4],
    ):
        super().__init__()
        self.num_kernels = len(r_kernel_size)

        self.pre_conv = weight_norm(
            nn.Conv1d(
                in_channels=80, 
                out_channels=in_channel, 
                kernel_size=7, 
                stride=1, 
                padding=3
            )
        )
        self.lrelu_slope = ReLU_SLOPE

        self.up_layers = nn.Sequential()
        for i, (rate, kernel) in enumerate(zip(u_rate, 
                                            u_kernel_size)):
            self.up_layers.add_module(
                f'Upsample layer {i}', 
                weight_norm(
                    nn.ConvTranspose1d(
                        in_channels=in_channel//(2**i),
                        out_channels=in_channel//(2**(i+1)),
                        kernel_size=kernel, 
                        stride=rate, 
                        padding=(kernel-rate)//2
                    )
                )
            )

        self.res_layers = nn.Sequential()
        for i in range(len(self.up_layers)):
            cur_layer = nn.Sequential()
            cur_channel = in_channel // (2 ** (i + 1))

            for j, (kernel, dilation) in enumerate(zip(
                                           r_kernel_size,
                                           r_dilation)):
                cur_layer.add_module(
                    f"Resblock {j}",
                    ResBlock(cur_channel, kernel, dilation),
                )
            
            self.res_layers.add_module(
                f"Layer {i}",
                cur_layer,
            )

        self.post_conv = weight_norm(
            nn.Conv1d(
                in_channels=cur_channel, 
                out_channels=1, 
                kernel_size=7, 
                stride=1, 
                padding=3
            )
        )
        self.pre_conv.apply(init_weights)
        self.up_layers.apply(init_weights)
        self.post_conv.apply(init_weights)

    def forward(self, mel_spec):
        x = self.pre_conv(mel_spec)

        for u_layer, r_layer in zip(
                            self.up_layers, 
                            self.res_layers):
                        
            x = u_layer(F.leaky_relu(x, ReLU_SLOPE))
            xs = 0
            for resblock in r_layer:
                xs += resblock(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)

        x = self.post_conv(x)
        output_audio = torch.tanh(x)
        return output_audio

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info

