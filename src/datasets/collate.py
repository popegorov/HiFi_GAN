import torch


def wav_collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    batch = {}
    audios = []
    spectrograms = []
    paths = []
    lengths = []

    for item in dataset_items:
        audios.append(torch.transpose(torch.tensor(item['audio']), 0, 1))
        spectrograms.append(torch.transpose(torch.tensor(item['spectrogram']), 0, 2))
        paths.append(item['audio_path'])
        lengths.append(item['audio_len'])

    batch['audio'] = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
    batch['audio'] = torch.transpose(batch['audio'], 1, 2).squeeze(1)
    batch['spectrogram'] = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True, padding_value=-11.5129251)
    batch['spectrogram'] = torch.transpose(batch['spectrogram'], 1, 3).squeeze(1)
    batch['audio'] = batch['audio'][:, :batch['spectrogram'].shape[-1] * 256]
    batch['paths'] = paths
    batch['audio_len'] = torch.tensor(lengths)

    return batch

def text_collate_fn(dataset_items: list[dict]):
    batch = {}
    texts = []
    paths = []
    spectrograms = []
    audio_lens = []
    for item in dataset_items:
        audio_lens.append(item['spectrogram'].shape[1] * 256)
        spectrograms.append(item['spectrogram'].permute(1, 2, 0))
        paths.append(item['path'])
        texts.append(item['text'])
    
    batch['texts'] = texts
    batch['paths'] = paths
    batch['audio_len'] = torch.tensor(audio_lens)
    batch['spectrogram'] = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True, padding_value=-11.5129251)
    batch['spectrogram'] = torch.transpose(batch['spectrogram'], 1, 3).squeeze(1)
    return batch
    


    
