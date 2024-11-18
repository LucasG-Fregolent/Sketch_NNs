from torch.utils.data import Dataset

# Create a combined dataset object
class CombinedCIFAR10(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]

        return image, target