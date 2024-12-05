import torch
from torch import nn

def feature_loss(real_feature_map, gen_feature_map, lambda_fm=2.0):
    loss = 0
    for real, gen in zip(real_feature_map, gen_feature_map):
        for real_layer, gen_layer in zip(real, gen):
            loss += torch.mean(torch.abs(real_layer - gen_layer))
    return lambda_fm * loss

def generator_loss(discriminator_output):
    loss = 0
    for output in discriminator_output:
        loss += torch.mean((1 - output)**2)
    return loss

def discriminator_loss(real_outputs, gen_outputs):
    loss = 0
    for real, gen in zip(real_outputs, gen_outputs):
        loss += torch.mean((1 - real)**2) + torch.mean(gen**2)
    return loss

def mel_spect_loss(real_mel_spec, gen_mel_spec, lambda_mel=45.0):
    return lambda_mel * nn.functional.l1_loss(real_mel_spec, gen_mel_spec)
