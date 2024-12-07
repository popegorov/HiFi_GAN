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
                entry["path"] = path
                entry["audio_len"] = self._calculate_length(path)
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)

    def _calculate_length(self, audio_path: Path) -> float:
        audio_info = torchaudio.info(str(audio_path))
        return audio_info.num_frames

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio_len = data_dict["audio_len"]
        audio = self.load_audio(audio_path)
        if self.crop and audio_len > self.len_to_crop:
            audio = audio[:, :self.len_to_crop]
            audio_len = self.len_to_crop

        spectrogram = self.mel_spec(audio)

        instance_data = {
            "audio": audio,
            "spectrogram": spectrogram,
            "audio_path": audio_path,
            "audio_len": audio_len,
        }

        # TODO think of how to apply wave augs before calculating spectrogram
        # Note: you may want to preserve both audio in time domain and
        # in time-frequency domain for logging
        # instance_data = self.preprocess_data(instance_data)

        return instance_data