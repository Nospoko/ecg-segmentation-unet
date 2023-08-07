import os
from glob import glob

import wfdb
import pandas as pd


def ltafdb_paths(folder: str) -> list[str]:
    # Get all the signal headers
    query = os.path.join(folder, "*.hea")
    paths = glob(query)

    # Get rid of the extension
    paths = [path[:-4] for path in paths]

    return paths


def load_ltafdb_record(record_path: str):
    ann = wfdb.rdann(record_path, "atr")
    # Convert to a convenient dataframe
    adf = pd.DataFrame({"symbol": ann.symbol, "aux": ann.aux_note, "position": ann.sample})

    signals, fields = wfdb.rdsamp(record_path)

    return signals, adf


if __name__ == "__main__":
    # That's where I'm downloading the LTAFDB data
    folder = "physionet.org/files/ltafdb/1.0.0/"
    paths = ltafdb_paths(folder)

    signals, adf = load_ltafdb_record(paths[3])

    # It has 2 channels
    print("Signals shape:", signals.shape)

    print("Number of annotations:", adf.shape[0])
    print("Annotation distribution:")
    print(adf.symbol.value_counts())
