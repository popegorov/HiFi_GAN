import os
import urllib.request

path = os.path.join(os.path.expanduser('.'), ".cache/wv_mos.ckpt")


if (not os.path.exists(path)):
    print("Downloading the checkpoint for WV-MOS")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(
        "https://zenodo.org/record/6201162/files/wav2vec2.ckpt?download=1",
        path
    )
    print('Weights downloaded in: {} Size: {}'.format(path, os.path.getsize(path)))
