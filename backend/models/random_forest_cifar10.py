import torch
from torchvision import datasets, transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

class RandomForestCIFAR10:
    def __init__(self, train_loader, val_loader, test_loader, random_state=42):

        # param_grid = {
        #     'n_estimators': [100, 150, 200],
        #     'max_depth': [1, 10, 15],
        #     'max_features': ['log2', 'sqrt'],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4]
        # }

        # grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=2)

        # train_features, train_labels = self.flatten_data(train_loader)

        # grid_search.fit(train_features, train_labels)
        # best_params = grid_search.best_params_
        # print(f"Best RandomForestClassifier parameters: {best_params}")
        
        best_params = {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
        self.model = RandomForestClassifier(**best_params, random_state=42)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def flatten_data(self, data_loader):
        X, y = [], []
        for images, labels in data_loader:
            # Flatten images and convert to numpy
            images = images.view(images.size(0), -1).numpy()
            X.append(images)
            y.append(labels.numpy())
        
        # Concatenate all batches into single arrays
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        return X, y
    
    def train(self):
        # Flatten and prepare data for training
        X_train, y_train = self.flatten_data(self.train_loader)
        self.model.fit(X_train, y_train)
        print("Training complete.")
    
    def evaluate(self, data_loader):
        # Flatten and prepare data for evaluation
        X, y = self.flatten_data(data_loader)
        predictions = self.model.predict(X)
        accuracy = accuracy_score(y, predictions)
        return accuracy, y, predictions
    
    def run(self):
        # Train the model
        self.train()
        
        # Evaluate on validation set
        val_accuracy, _, _ = self.evaluate(self.val_loader)
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
        
        # Evaluate on test set and get predictions
        test_accuracy, y_true, y_pred = self.evaluate(self.test_loader)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        
        # Plot the confusion matrix
        self.plot_confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix for CIFAR-10")
        plt.show()
