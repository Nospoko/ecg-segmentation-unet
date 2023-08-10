import torch
import einops
from datasets import load_dataset, Features, Array2D, Value
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, huggingface_path: str):
        super().__init__()

        features = Features(
            {
                "record_id": Value(dtype="string"),
                "signal": Array2D(dtype="float32", shape=(1000, 2)),
                "mask": Array2D(dtype="int8", shape=(1000, 1)),
            }
        )
        self.data = load_dataset(huggingface_path, features=features, split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.data[index]

        # wrap signal and mask to torch.Tensor
        signal = torch.tensor(record["signal"], dtype=torch.float32)
        mask = torch.tensor(record["mask"], dtype=torch.float32)

        # reshape data: [sequence_length, num_channels] -> [num_channels, sequence_length]
        signal = einops.rearrange(signal, "l c -> c l")
        # loss
        mask = einops.rearrange(mask, "l c -> c l")

        return signal, mask
