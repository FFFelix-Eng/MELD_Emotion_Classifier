import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import os

from Code.util.metrics import compute_metrics, save_metrics, init_metrics_file


class Trainer_dia_emo:
    def __init__(self, model, train_dataset, val_dataset=None, test_dataset=None, collate_custom=None, batch_size=32, learning_rate=0.001, save_path='./models', device=None, model_name='', csv_name=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model and Data
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.collate_custom = collate_custom

        # Settings
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_path = save_path
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')  # Use reduction='none' to get individual losses

        # initialization of the metrics file
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.model_name = model_name

        self.metrics_file = os.path.join(self.save_path, f'{csv_name}.csv')
        init_metrics_file(self.metrics_file)



    def compute_metrics_using_util(self, dataloader):
        self.model.eval()
        all_labels = []
        all_predictions = []
        all_masks = []
        # total_loss = 0
        # total_count = 0

        with torch.no_grad():
            for batch_idx, batch_content in enumerate(dataloader):
                dialogues, labels, mask, dialogue_id = batch_content['utts'], batch_content['label'], batch_content[
                    'mask'], batch_content['d_id']
                # input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                # sentence_mask = self.compute_sentence_mask(mask)
                predictions = self.model(dialogues, self.device)
                predictions_class_output = torch.argmax(predictions, dim=2)

                # labels_unbind = list(torch.unbind(labels))
                # sentence_mask_unbind = list(torch.unbind(sentence_mask))
                # predictions_unbind = predictions.tolist()

                all_predictions.append(predictions_class_output.cpu())
                all_labels.append(labels)
                all_masks.append(mask)

        metrics = compute_metrics(all_predictions.copy(), all_labels.copy(), all_masks.copy())
        return metrics


                # Flatten the predictions and labels
                # predictions = predictions.view(-1, self.model.fc.out_features)
                # labels = labels.view(-1)
                # sentence_mask = sentence_mask.view(-1)

                # Compute loss for each element
                # loss = self.criterion(predictions, labels)

                # Apply sentence mask to the loss
                # loss = loss * sentence_mask.float()
                # total_loss += loss.sum().item()
                # total_count += sentence_mask.sum().item()

                # Apply sentence mask to the predictions and labels
                # active_outputs = predictions[sentence_mask == 1]
                # active_labels = labels[sentence_mask == 1]

                # Get the predicted class (by index)
                # _, predicted = torch.max(active_outputs, 1)
                # all_labels.extend(active_labels.cpu().numpy())
                # all_predictions.extend(predicted.cpu().numpy())

        # avg_loss = total_loss / total_count
        # accuracy = accuracy_score(all_labels, all_predictions)
        # macro_f1 = f1_score(all_labels, all_predictions, average='weighted')
        #
        # return avg_loss, accuracy, macro_f1

    def compute_sentence_mask(self, attention_mask):
        # Compute sentence mask by checking if all tokens in a sentence are padding (0)
        # sentence_mask = (attention_mask.sum(dim=2) > 0)
        sentence_mask = []
        return sentence_mask

    def convert_double_list_to_tensor(self, list_target):
        """
        Convert labels to a tensor. If labels is a nested list (i.e., list of lists),
        flatten it before converting.
        """
        # print(list_target)
        if isinstance(list_target, list) and any(isinstance(lbl, list) for lbl in list_target):
            # Flatten the nested list
            flattened_labels = [item for sublist in list_target for item in sublist]
        else:
            flattened_labels = list_target

        # Convert to tensor
        # print(flattened_labels)
        return torch.tensor(flattened_labels).view(-1)

    def train(self, epochs):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_custom)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_custom) if self.val_dataset else None
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_custom) if self.test_dataset else None

        for epoch in range(epochs):
            print(f"Epoch {epoch}/{epochs-1}")
            self.model.train()
            total_loss = 0
            total_count = 0

            for batch_idx, batch_content in enumerate(train_loader):
                dialogues, labels, mask, dialogue_id = batch_content['utts'], batch_content['label'], batch_content['mask'], batch_content['d_id']
                # dialogues, labels, mask = dialogues.to(self.device), labels.to(
                #     self.device), mask.to(self.device)


                outputs = self.model(dialogues, self.device)

                # Flatten the outputs and labels
                outputs = outputs.view(-1, self.model.fc.out_features)
                labels = self.convert_double_list_to_tensor(labels)
                mask = self.convert_double_list_to_tensor(mask)



                # Compute loss for each element
                valid_indices = (mask == 1).nonzero(as_tuple=False).view(-1).tolist()
                filtered_outputs = outputs[valid_indices]
                filtered_labels = labels[valid_indices].to(self.device)
                loss = self.criterion(filtered_outputs, filtered_labels.long())
                loss = loss.mean()


                if torch.isnan(loss):
                    print("Loss is NaN. Stopping training.")
                    return

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # total_loss += loss.item() * mask.sum().item()
                # total_count += sentence_mask.sum().item()


            train_metrics = self.compute_metrics_using_util(train_loader)

            save_metrics(self.metrics_file, epoch, 'train', train_metrics)

            if val_loader:
                val_metrics = self.compute_metrics_using_util(val_loader)
                save_metrics(self.metrics_file, epoch, 'val', val_metrics)

            if test_loader:
                test_metrics = self.compute_metrics_using_util(test_loader)
                save_metrics(self.metrics_file, epoch, 'test', test_metrics)

            model_save_path = os.path.join(self.save_path, f'{self.model_name}_{epoch}.pth')
            torch.save(self.model.state_dict(), model_save_path)

            print(f"Epoch {epoch + 1}/{epochs} \nTrain Metrics: {train_metrics}")
            if val_loader:
                print(f"Epoch {epoch + 1}/{epochs} \nValidation Metrics: {val_metrics}")
            if test_loader:
                print(f"Epoch {epoch + 1}/{epochs} \nTest Metrics: {test_metrics}")
