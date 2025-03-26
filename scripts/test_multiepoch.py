import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

# 创建一个简单的数据集
class SimpleDataset(Dataset):
    def __init__(self, size=10):
        self.data = list(range(size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 测试函数：验证shuffle功能
def test_dataloader_shuffle():
    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 创建数据集
    dataset = SimpleDataset(size=10)

    # 创建两个DataLoader，一个有shuffle，一个没有
    dataloader_with_shuffle = MultiEpochsDataLoader(dataset, batch_size=1, shuffle=True)

    dataloader_without_shuffle = MultiEpochsDataLoader(dataset, batch_size=1, shuffle=False)

    print("测试多个epoch的数据顺序：")

    # 测试有shuffle的DataLoader在多个epoch中的顺序
    print("\n有shuffle的DataLoader:")
    epoch_orders = []
    for epoch in range(3):
        epoch_data = []
        for batch in dataloader_with_shuffle:
            epoch_data.append(batch.item())
        epoch_orders.append(epoch_data)
        print(f"Epoch {epoch + 1}: {epoch_data}")

    # 检查不同epoch的顺序是否不同
    all_same = all(epoch_orders[0] == epoch_orders[i] for i in range(1, len(epoch_orders)))
    print(f"所有epoch的顺序相同? {'是' if all_same else '否'}")

    # 测试无shuffle的DataLoader在多个epoch中的顺序
    print("\n无shuffle的DataLoader:")
    epoch_orders = []
    for epoch in range(3):
        epoch_data = []
        for batch in dataloader_without_shuffle:
            epoch_data.append(batch.item())
        epoch_orders.append(epoch_data)
        print(f"Epoch {epoch + 1}: {epoch_data}")

    # 检查不同epoch的顺序是否相同
    all_same = all(epoch_orders[0] == epoch_orders[i] for i in range(1, len(epoch_orders)))
    print(f"所有epoch的顺序相同? {'是' if all_same else '否'}")


# 运行测试
if __name__ == "__main__":
    test_dataloader_shuffle()
