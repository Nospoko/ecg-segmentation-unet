import os
from glob import glob

import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import binary_dilation


def create_save_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


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

    return signals, adf, fields


def create_mask(data: pd.DataFrame, dilation: int):
    condition = data["symbol"].isin(["N", "V", "A"])

    mask = np.where(binary_dilation(condition, iterations=dilation), 1, 0)

    return mask


def process_file(path: str, save_dir: str, sequence_window: int = 1000, area_around_beat_ms: float = 100):
    signals, adf, fields = load_ltafdb_record(path)
    # get file name
    filename = path.split("/")[-1]

    # extract sampling rate
    fs = fields["fs"]
    # calculating dilation to get fields that are part of the beat
    # area_around_beat_ms is value in miliseconds
    dilation = int((area_around_beat_ms * fs / 1000) // 2)

    # create dataframe from signals
    signals_df = pd.DataFrame(signals, columns=["channel_1", "channel_2"])
    # set beat position as index, it will be used to merge signals and adf
    adf.set_index("position", inplace=True)

    data = pd.merge(signals_df, adf, how="left", right_index=True, left_index=True)

    # add mask column
    data["mask"] = create_mask(data, dilation=dilation)
    data = data[["channel_1", "channel_2", "mask"]]

    # create save dir if doesn't exist
    create_save_path(save_dir)

    for subset in tqdm(data.rolling(window=sequence_window, step=sequence_window)):
        if len(subset) != sequence_window:
            continue

        # create save path
        save_path = f"{save_dir}/file-{filename}-indices-{subset.index[0]}-{subset.index[-1]}.csv"
        # write subset to save dir
        subset.to_csv(save_path)


if __name__ == "__main__":
    # That's where I'm downloading the LTAFDB data
    folder = "physionet.org/files/ltafdb/1.0.0/"
    paths = ltafdb_paths(folder)

    process_file(path=paths[0], save_dir="dataset", sequence_window=1000, area_around_beat_ms=100)
