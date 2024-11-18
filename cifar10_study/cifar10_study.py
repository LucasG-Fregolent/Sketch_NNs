import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils
from load_cifar10.load_cifar10 import LoadCIFAR10
from models.random_forest_cifar10 import RandomForestCIFAR10
from trainers.cnn_trainer import CNNTrainer
from models.scatch_cnn import ScatchCNN
import torchvision.datasets as datasets
import numpy as np
from utils.utils import Utils
from models.cnn_model import CNNModel
from models.resnet_model import ResNetModel

utils = Utils()

selected_classes = [0, 1, 9]
partition = [400, 200, 300]

# Instantiate and load CIFAR-10 data
data_loader = LoadCIFAR10(selected_classes, partition)
train_loader, val_loader, test_loader, class_names = data_loader.load_data()

print("Selected Classes:", [class_names[i] for i in selected_classes])

# Verifying data split
class_mapping = {0: 0, 1: 1, 9: 2}
data_loader.verify_split(train_loader, "Training", selected_classes, class_names, class_mapping)
data_loader.verify_split(val_loader, "Validation", selected_classes, class_names, class_mapping)
data_loader.verify_split(test_loader, "Test", selected_classes, class_names, class_mapping)

inverse_class_mapping = {0: 0, 1: 1, 2: 9}

# def imshow(img, title=None):
#     img = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
#     img = img.numpy()
#     npimg = np.transpose(img, (1, 2, 0))
#     npimg = np.clip(npimg, 0, 1)
#     plt.imshow(npimg)
#     if title:
#         plt.title(title)
#     plt.axis('off')
#     plt.show()

# # Get a batch of training data for display
# dataiter = iter(train_loader)
# images, labels = next(dataiter)
# imshow(torchvision.utils.make_grid(images[:8], nrow=4),
#        title=[class_names[inverse_class_mapping[labels[j].item()]] for j in range(8)])

# """
# Training and evaluation of a baseline model in the CIFAR10 dataset.

# The chosen baseline model were the sklearn implementation of the
# RandomForestClassifier model.
# """
# # Run RandomForest on CIFAR-10
# rf_cifar10 = RandomForestCIFAR10(train_loader, val_loader, test_loader).run()

"""
Training and evaluation of CNN models on CIFAR10.

Parts of the CNN tested
    * Number of convolutional layers and different output sizes
    * Kernel size of filters
    * Number of training epochs
    * Learning rate of the network
    * Batchsize used in dataloader
    * Stride on convolutional or pooling layers
    * Different architecture for the final dense layer (with or without hidden layers and their size)
"""

def criterion(model, preds, targets, device):
        ce = nn.CrossEntropyLoss().to(device)
        loss = ce(preds, targets.long())
        pred_labels = torch.max(preds.data, 1)[1]
        acc = torch.sum(pred_labels == targets.data)
        n = pred_labels.size(0)
        acc = acc / n
        return loss, acc

"""
PART 1:
Training of an example CNN model (called ScratchCNN) with different optimizers.

Each ScratchCNN model uses a different optimizer, but the architecture is the same.
"""
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model1, model2, model3 = ScatchCNN(), ScatchCNN(), ScatchCNN()
# optimizer1 = optim.Adam(model1.parameters(), lr=0.001, weight_decay=5e-4)
# optimizer2 = optim.SGD(model2.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
# optimizer3 = optim.AdamW(model3.parameters(), lr=0.001, weight_decay=5e-4)

# num_epochs = 10


# for i, (model, optimizer) in enumerate([(model1, optimizer1), (model2, optimizer2), (model3, optimizer3)], 1):
#     print(f"Training with optimizer {i}")
#     trainer = CNNTrainer(model, train_loader, val_loader, test_loader, criterion, optimizer, device)
#     trainer.train(num_epochs)


"""
PART 2:
Training of CNN models to improve classification accuracy (with CNNModel models)
"""

def create_model_from_config(config, model_name):
    model_config = config["models"][model_name]

    model = CNNModel(
        conv_layers=model_config["conv_layers"],
        conv_out_channels=model_config["conv_out_channels"],
        kernel_size=model_config["kernel_size"],
        stride=model_config["stride"],
        final_dense_layers=model_config["final_dense_layers"],
        num_classes=model_config["num_classes"],
        batch_norm=model_config["batch_norm"],
        dropout_rate=model_config["dropout_rate"],
        pooling_type=model_config["pooling_type"],
        pool_kernel_size=model_config["pool_kernel_size"]
    )

    return model

def create_model_from_config_resnet(config, model_name):
    model_config = config["models"][model_name]

    model = ResNetModel(
        num_blocks=model_config["num_blocks"],
        conv_out_channels=model_config["conv_out_channels"],
        final_dense_layers=model_config["final_dense_layers"],
        num_classes=model_config["num_classes"],
        batch_norm=model_config["batch_norm"],
        dropout_rate=model_config["dropout_rate"],
        pooling_type=model_config["pooling_type"]
    )

    return model



config = utils.load_config("models/cnn_model_config.yaml")

model = create_model_from_config(config, "model_4")

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 20
trainer = CNNTrainer(model, train_loader, val_loader, test_loader, criterion, optimizer, device)
trainer.train(num_epochs)

print(model)

# model = create_model_from_config_resnet(config, "resnet_model_1")
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_epochs = 51
# trainer = CNNTrainer(model, train_loader, val_loader, test_loader, criterion, optimizer, device)    
# trainer.train(num_epochs)