from pathlib import Path
from src.datasets.base_dataset import BaseDataset
import torch

from download_weights_tts import ROOT_PATH


class CustomDirDataset(BaseDataset):
    def __init__(self, text_dir, tts_dir, text, *args, **kwargs):
        data = []
        if not text_dir or not Path(text_dir).exists():
            print("Invalid path")
            return

        for path in Path(text_dir).iterdir():
            entry = {}
            if path.suffix == '.txt':
                entry["path"] = path
                with path.open() as f:
                    entry['text'] = f.read().strip()
            if len(entry) > 0:
                data.append(entry)

        if text is not None and type(text)==str:
            entry = {
                'path': Path("custom.txt"),
                'text': text
            }
            data.append(entry)

        self._index = data
        path = ROOT_PATH / "saved" / tts_dir
        self.tokenizer = torch.load(str(path) + "/tokenizer.pth")
        self.text_to_mel_model = torch.load(str(path) + "/text_to_mel.pth")

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        text = data_dict['text']
        path = data_dict['path']
        inputs = self.tokenizer(text, return_tensors="pt")
        spectrogram = self.text_to_mel_model(inputs["input_ids"].detach(), return_dict=True)['spectrogram'].detach()
        
        instance_data = {
            "text": text,
            "spectrogram": spectrogram,
            "path": path,
        }
        return instance_data
