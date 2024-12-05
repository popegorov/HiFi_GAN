from pathlib import Path
from src.datasets.base_dataset import BaseDataset
from tqdm import tqdm
import torchaudio

class WavDataset(BaseDataset):
    def __init__(self, wav_dir, *args, **kwargs):
        data = []
        if not wav_dir or not Path(wav_dir).exists():
            print("Invalid path")
            return

        for path in tqdm(Path(wav_dir).iterdir(), desc="Loading Wavs"):
            entry = {}
            if path.suffix == '.wav':
                entry["path"] = str(path)
                entry["audio_len"] = self._calculate_length(path)
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)

    def _calculate_length(self, audio_path: Path) -> float:
        audio_info = torchaudio.info(str(audio_path))
        return audio_info.num_frames / audio_info.sample_rate