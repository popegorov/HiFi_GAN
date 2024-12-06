from src.model.utils import get_padding, ReLU_SLOPE
from torch.nn.utils import spectral_norm, weight_norm
from torch import nn

import torch
import torch.nn.functional as F


class PeriodSubDiscriminator(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period

        self.convs = nn.Sequential()
        channels = [
            [1, 32], 
            [32, 128], 
            [128, 512], 
            [512, 1024]
        ]

        for i, (i_channel, o_channel) in enumerate(channels):
            self.convs.add_module(
                f"Conv {i}",
                weight_norm(
                    nn.Conv2d(
                        in_channels=i_channel, 
                        out_channels=o_channel,
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )               
                )
            )
        self.convs.add_module(
            f"Conv {len(channels)}",
            weight_norm(
                nn.Conv2d(
                    in_channels=1024, 
                    out_channels=1024,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(2, 0),
                )               
            )
        )
        
        self.post_conv = weight_norm(
            nn.Conv2d(
                in_channels=1024, 
                out_channels=1, 
                kernel_size=(3, 1), 
                stride=1, 
                padding=(1, 0)
            )
        )

    def forward(self, audio):
        features = []

        batch_size, num_channels, seq_length = audio.shape
        if seq_length % self.period != 0:
            to_pad = self.period - (seq_length % self.period)
            audio = F.pad(
                input=audio, 
                pad=(0, to_pad), 
                mode="reflect",
            )
            seq_length = seq_length + to_pad
        x = audio.view(batch_size, num_channels, seq_length // self.period, self.period)

        for conv in self.convs:
            x = F.leaky_relu(conv(x), ReLU_SLOPE)
            features.append(x)
        x = self.post_conv(x)
        features.append(x)
        output = torch.flatten(x, 1, -1)

        return output, features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        layers = [PeriodSubDiscriminator(p) for p in periods]
        self.discriminators = nn.Sequential(*layers)

    def forward(self, y_real, y_generated):
        d_y_reals = []
        d_y_gens = []
        feature_reals = []
        feature_gens = []
        for discriminator in self.discriminators:
            d_y_real, feature_real = discriminator(y_real)
            d_y_gen, feature_gen = discriminator(y_generated)
            d_y_reals.append(d_y_real)
            feature_reals.append(feature_real)
            d_y_gens.append(d_y_gen)
            feature_gens.append(feature_gen)

        return d_y_reals, d_y_gens, feature_reals, feature_gens

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