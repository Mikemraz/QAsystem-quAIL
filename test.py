from utils.util import get_dataset_race
from torch.utils.data import DataLoader

dataset_train, dataset_dev = get_dataset_race()

dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=64)
dataloader_dev = DataLoader(dataset_dev, shuffle=True, batch_size=64)


print(len(dataloader_train))
print(len(dataloader_dev))
for cw_idxs, qw_idxs, y in dataloader_train:
    print(cw_idxs[0])
    print(qw_idxs[0])
    break