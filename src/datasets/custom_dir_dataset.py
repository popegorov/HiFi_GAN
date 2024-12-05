from pathlib import Path
from src.datasets.base_dataset import BaseDataset

class CustomDirDataset(BaseDataset):
    def __init__(self, text_dir, *args, **kwargs):
        data = []
        if not text_dir or not Path(text_dir).exists():
            print("Invalid path")
            return

        for path in Path(text_dir).iterdir():
            entry = {}
            if path.suffix == '.txt':
                entry["path"] = str(path)
                with path.open() as f:
                    entry['text'] = f.read().strip()
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
