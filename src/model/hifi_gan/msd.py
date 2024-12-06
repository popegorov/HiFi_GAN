from src.model.utils import ReLU_SLOPE, init_weights
from torch import nn
from torch.nn.utils import spectral_norm, weight_norm

import torch
import torch.nn.functional as F

class ScaleSubDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        normalization = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.Sequential()
        params = [
            [1, 128, 15, 1, 1, 7], 
            [128, 128, 41, 2, 4, 20],
            [128, 256, 41, 2, 16, 20],
            [256, 512, 41, 4, 16, 20],
            [512, 1024, 41, 4, 16, 20],
            [1024, 1024, 41, 4, 16, 20],
            [1024, 1024, 5, 1, 1, 2],
        ]

        for j, (i, o, k, s, g, p) in enumerate(params):
            self.convs.add_module(
                f"Conv {j}",
                normalization(
                    nn.Conv1d(
                        in_channels=i, 
                        out_channels=o,
                        kernel_size=k,
                        stride=s,
                        groups=g,
                        padding=p,
                    )               
                )
            )
        
        
        self.post_conv = normalization(
            nn.Conv1d(
                in_channels=1024, 
                out_channels=1, 
                kernel_size=3, 
                stride=1, 
                padding=1
            )
        )


    def forward(self, audio):
        features = []
        x = audio
        for conv in self.convs:
            x = F.leaky_relu(conv(x), ReLU_SLOPE)
            features.append(x)
        x = self.post_conv(x)
        features.append(x)
        output = torch.flatten(x, 1, -1)
        return output, features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.Sequential(*[
            ScaleSubDiscriminator(use_spectral_norm=True),
            ScaleSubDiscriminator(),
            ScaleSubDiscriminator(),
        ])
        self.meanpools = nn.Sequential(*[
            nn.AvgPool1d(4, 2, padding=1),
            nn.AvgPool1d(4, 2, padding=1)
        ])

    def forward(self, y_real, y_generated):
        d_y_reals = []
        d_y_gens = []
        feature_reals = []
        feature_gens = []
        for i, discriminator in enumerate(self.discriminators):
            if i != 0:
                y_real = self.meanpools[i-1](y_real)
                y_generated = self.meanpools[i-1](y_generated)
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
