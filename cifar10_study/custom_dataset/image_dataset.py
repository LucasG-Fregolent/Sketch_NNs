from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        # Define the mapping from original labels to [0, 1, 2]
        # This is necessary becuase the last layer should have only 3 nodes, from 0 to 2
        self.class_mapping = {0: 0, 1: 1, 9: 2}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image, target = self.dataset[actual_idx]
        
        # Map the original target label to the new class index
        target = self.class_mapping[target]

        if self.transform:
            image = self.transform(image)

        return image, target