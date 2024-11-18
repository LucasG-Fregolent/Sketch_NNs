import torch
import torch.nn as nn
import torch.optim as optim

class ResNetTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def train_batch(self, data):
        self.model.train()  # Set the model to training mode
        ims, targets = data
        ims, targets = ims.to(self.device), targets.to(self.device)

        # Forward pass
        preds = self.model(ims)
        loss = self.criterion(preds, targets)

        # Calculate accuracy
        _, predicted = torch.max(preds, 1)
        correct = (predicted == targets).sum().item()
        accuracy = correct / targets.size(0)

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item(), accuracy

    @torch.no_grad()
    def validate_batch(self, data):
        self.model.eval()  # Set the model to evaluation mode
        ims, targets = data
        ims, targets = ims.to(self.device), targets.to(self.device)

        # Forward pass
        preds = self.model(ims)
        loss = self.criterion(preds, targets)

        # Calculate accuracy
        _, predicted = torch.max(preds, 1)
        correct = (predicted == targets).sum().item()
        accuracy = correct / targets.size(0)

        return loss.item(), accuracy

    def train_epoch(self):
        total_loss, total_acc = 0.0, 0.0
        for data in self.train_loader:
            loss, acc = self.train_batch(data)
            total_loss += loss
            total_acc += acc
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_acc / len(self.train_loader)
        return avg_loss, avg_acc

    def validate_epoch(self):
        total_loss, total_acc = 0.0, 0.0
        for data in self.val_loader:
            loss, acc = self.validate_batch(data)
            total_loss += loss
            total_acc += acc
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_acc / len(self.val_loader)
        return avg_loss, avg_acc

    def test_epoch(self):
        total_loss, total_acc = 0.0, 0.0
        for data in self.test_loader:
            loss, acc = self.validate_batch(data)
            total_loss += loss
            total_acc += acc
        avg_loss = total_loss / len(self.test_loader)
        avg_acc = total_acc / len(self.test_loader)
        return avg_loss, avg_acc

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        test_loss, test_acc = self.test_epoch()
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
