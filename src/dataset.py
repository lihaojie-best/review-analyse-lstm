import pandas as pd
import torch
from torch._C import dtype
from torch.utils.data import Dataset,DataLoader
from triton.language import tensor
import config


class ReviewDataset(Dataset):
    def __init__(self,data_path):
        self.data = pd.read_json(data_path,lines=True,orient="records").to_dict(orient="records")

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        input_tensor = tensor(self.data[index]['review'], dtype=tensor.long)
        target_tensor = tensor(self.data[index]['label'],dtype=tensor.float32)
        return input_tensor,target_tensor


# 获取dataloader的方法
def get_dataloader(train = True):
    data_path = config.PROCESSED_DATA_DIR / ("indexed_train.jsonl" if train else "indexed_test.jsonl")
    dataset = ReviewDataset(data_path)
    return DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=True)

if __name__ == "__main__":
    train_dataloader = get_dataloader()
    print(f"train batch 个数：{len(train_dataloader)}")
    test_dataloader = get_dataloader(train=False)
    print(f"test batch 个数:{len(test_dataloader)}")


