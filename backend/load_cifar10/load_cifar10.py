import torch
import torchvision.transforms.v2 as transforms
import random
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter
from custom_dataset.image_dataset import ImageDataset
from custom_dataset.combined_cifar10 import CombinedCIFAR10
import torchvision.datasets as datasets

class LoadCIFAR10:
    def __init__(self, selected_classes, partition):
        self.selected_classes = selected_classes
        self.partition = partition
        self.input_width = 32
        self.input_height = 32
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize((self.input_width, self.input_height), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.test_val_transform = transforms.Compose([
            transforms.Resize((self.input_width, self.input_height), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

    def load_data(self):
        cifar_train = datasets.CIFAR10(root='./data', train=True, download=True)
        cifar_test = datasets.CIFAR10(root='./data', train=False, download=True)

        class_names = cifar_train.classes

        combined_data = np.concatenate((cifar_train.data, cifar_test.data), axis=0)
        combined_targets = cifar_train.targets + cifar_test.targets

        combined_dataset = CombinedCIFAR10(combined_data, combined_targets)
        class_to_indices = {cls: [] for cls in self.selected_classes}

        for idx, target in enumerate(combined_targets):
            if target in self.selected_classes:
                class_to_indices[target].append(idx)

        num_train, num_val, num_test = self.partition
        train_indices, val_indices, test_indices = [], [], []

        for cls in self.selected_classes:
            indices = class_to_indices[cls]
            random.shuffle(indices)
            train_indices.extend(indices[:num_train])
            val_indices.extend(indices[num_train:num_train + num_val])
            test_indices.extend(indices[num_train + num_val:num_train + num_val + num_test])

        train_dataset = ImageDataset(combined_dataset, train_indices, transform=self.train_transform)
        val_dataset = ImageDataset(combined_dataset, val_indices, transform=self.test_val_transform)
        test_dataset = ImageDataset(combined_dataset, test_indices, transform=self.test_val_transform)

        batch_size = 64
        num_workers = 4
        pin_memory = torch.cuda.is_available()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        return train_loader, val_loader, test_loader, class_names

    @staticmethod
    def verify_split(loader, split_name, selected_classes, class_names, class_mapping):
        targets = []
        for _, labels in loader:
            targets.extend(labels.tolist())
        counter = Counter(targets)
        print(f"{split_name} set class distribution:")
        for cls in selected_classes:
            print(f"  Class {cls} ({class_names[cls]}): {counter.get(class_mapping[cls], 0)} images")
