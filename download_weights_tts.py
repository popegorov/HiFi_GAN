import gdown
import hydra
from pathlib import Path
import nltk

ROOT_PATH = Path(__file__).absolute().resolve().parent

@hydra.main(version_base=None, config_path="src/configs", config_name="download_text")
def main(config):
    tokenizer_link = config.vars.tokenizer_link
    model_link = config.vars.model_link
    save_path = ROOT_PATH / "saved" / config.vars.to_save
    save_path.mkdir(exist_ok=True, parents=True)
    
    gdown.download(tokenizer_link, str(save_path) + "/tokenizer.pth", fuzzy=True)
    gdown.download(model_link, str(save_path) + "/text_to_mel.pth", fuzzy=True)
    nltk.download('averaged_perceptron_tagger_eng')


if __name__ == "__main__":
    main()
