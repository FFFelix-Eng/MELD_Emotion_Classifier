import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random



class Trainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size=32, learning_rate=0.001, device=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.random_seed = 2024
        self._randomness_contrtol()


    def _randomness_contrtol(self):
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
    
    def train(self, epochs=5):
        self.model.train()
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for input_ids, attention_mask, labels in train_loader:
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            

            val_accuracy = self.validate(self.val_dataset)
            print(f"Validation Accuracy: {val_accuracy:.2f}%")
    
    def validate(self, dataset):
        self.model.eval()
        val_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                outputs = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def test(self, dataset):
        self.model.eval()
        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for input_ids, attention_mask, labels in test_loader:
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                outputs = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

# Example usage

