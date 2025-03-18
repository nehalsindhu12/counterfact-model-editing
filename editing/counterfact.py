import json
import os
from pathlib import Path
import requests

class CounterFactDataset:
    BASE_URL = "https://rome.baulab.info/data/dsets/"

    def __init__(self, data_dir, multi=False, size=None):
        self.data_dir = Path(data_dir)
        self.multi = multi
        self.size = size

        self._download_data()
        self._load_data()

    def _download_data(self):
        file_name = "multi_counterfact.json" if self.multi else "counterfact.json"
        file_path = self.data_dir / file_name

        if not file_path.exists():
            os.makedirs(self.data_dir, exist_ok=True)
            remote_url = f"{self.BASE_URL}{file_name}"
            print(f"{file_name} not found in {self.data_dir}. Downloading from {remote_url}...")

            try:
                response = requests.get(remote_url, stream=True)
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded {file_name} to {file_path}.")
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to download {file_name} from {remote_url}. Error: {e}")

    def _load_data(self):
        file_name = "multi_counterfact.json" if self.multi else "counterfact.json"
        file_path = self.data_dir / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"{file_name} not found in {self.data_dir}. Please download it manually.")

        with open(file_path, "r") as f:
            self.data = json.load(f)

        if self.size:
            self.data = self.data[:self.size]

        print(f"Loaded dataset with {len(self.data)} elements.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
