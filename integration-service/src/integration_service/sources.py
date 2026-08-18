import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterator

import pandas as pd


class DataSource(ABC):
    @abstractmethod
    def read(self) -> Iterator[pd.DataFrame]:
        pass

    @abstractmethod
    def descriptor(self) -> dict:
        pass

class CsvFileSource(DataSource):
    def __init__(self, path, chunksize=25_000):
        self.path = path
        self.chunksize = chunksize


    def read(self) -> Iterator[pd.DataFrame]:
        return pd.read_csv(
            self.path,
            chunksize=self.chunksize,
            dtype=str,
            keep_default_na=False
        )


    def descriptor(self) -> dict:
        sha256 = hashlib.sha256()
        with open(self.path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return {
            'source_file': self.path,
            'sha256': sha256.hexdigest()
        }