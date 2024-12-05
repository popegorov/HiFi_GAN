from src.metrics.utils import get_wvmos
from src.metrics.base_metric import BaseMetric
import torch

class MOSMetric(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, args, kwargs)
        self.mos_calcer = get_wvmos(cuda=torch.cuda.is_available())

    def __call__(self, generated_audio, **batch):
        return self.mos_calcer.calculate_batch(generated_audio=generated_audio)
