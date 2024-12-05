ReLU_SLOPE = 0.1

def init_weights(layer, mean=0.0, std=0.01):
    if layer.__class__.__name__.find("Conv") != -1:
        layer.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation)/2)
