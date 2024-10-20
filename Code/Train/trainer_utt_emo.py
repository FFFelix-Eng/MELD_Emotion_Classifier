import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import os


class Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, test_dataset=None, batch_size=32, learning_rate=0.001,
                 save_path='./models', device=None, model_name=''):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model and Data
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Settings
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_path = save_path
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

        # initialization of the metrics file
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.model_name = model_name

        self.metrics_file = os.path.join(self.save_path, 'metrics.csv')
        self.init_metrics_file()

    def init_metrics_file(self):
        # Initialize the metrics CSV file
        df = pd.DataFrame(columns=['epoch', 'phase', 'loss', 'accuracy', 'macro_f1'])
        df.to_csv(self.metrics_file, index=False)

    def save_metrics(self, epoch, phase, loss, accuracy, macro_f1):
        df = pd.read_csv(self.metrics_file)
        new_row = pd.DataFrame({'epoch': [epoch], 'phase': [phase], 'loss': [loss], 'accuracy': [accuracy], 'macro_f1': [macro_f1]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.metrics_file, index=False)

    def compute_metrics(self, dataloader):
        self.model.eval()
        all_labels = []
        all_predictions = []
        total_loss = 0

        with torch.no_grad():
            for input_ids, attention_mask, labels in dataloader:
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(
                    self.device), labels.to(self.device)
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # get the output class (by index)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        macro_f1 = f1_score(all_labels, all_predictions, average='macro')

        return avg_loss, accuracy, macro_f1

    def train(self, epochs):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                shuffle=False) if self.val_dataset else None
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                 shuffle=False) if self.test_dataset else None

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for input_ids, attention_mask, labels in train_loader:
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(
                    self.device), labels.to(self.device)
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            train_loss, train_accuracy, train_macro_f1 = self.compute_metrics(train_loader)
            self.save_metrics(epoch, 'train', train_loss, train_accuracy, train_macro_f1)

            if val_loader:
                val_loss, val_accuracy, val_macro_f1 = self.compute_metrics(val_loader)
                self.save_metrics(epoch, 'val', val_loss, val_accuracy, val_macro_f1)

            if test_loader:
                test_loss, test_accuracy, test_macro_f1 = self.compute_metrics(test_loader)
                self.save_metrics(epoch, 'test', test_loss, test_accuracy, test_macro_f1)

            model_save_path = os.path.join(self.save_path, f'{self.model_name}_{epoch}.pth')
            torch.save(self.model.state_dict(), model_save_path)

            print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Macro F1: {train_macro_f1:.4f}")
            if val_loader:
                print(
                    f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Macro F1: {val_macro_f1:.4f}")
            if test_loader:
                print(
                    f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Macro F1: {test_macro_f1:.4f}")