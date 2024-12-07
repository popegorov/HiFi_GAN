import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_wav_dataloaders, get_text_dataloader
from src.trainer import Synthesizer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH


        
        
# login(token="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") # read
# login("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") # write
warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="synthesize")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)
    # login(token="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")

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_text_dataloader(config)

    # build model architecture, then print to console
    generator = instantiate(config.generator).to(device)
    print(generator)

    # get metrics
    metrics = {"inference": []}
    for metric_config in config.metrics.get("inference", []):
        metrics["inference"].append(
            instantiate(metric_config)
        )

    # save_path for model predictions
    save_path = ROOT_PATH / "data" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    synthesizer = Synthesizer(
        generator=generator,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
    )

    logs = synthesizer.synthesize()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
