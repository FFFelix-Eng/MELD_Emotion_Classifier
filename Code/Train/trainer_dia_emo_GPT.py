import os

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd

from Code.util.metrics_old import init_metrics_file


class Trainer:
    def __init__(self, model, model_name, train_dataset, val_dataset=None, test_dataset=None, batch_size=32,
                 save_path='./models/'):
        self.model = model
        self.model_name = model_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.save_path = save_path
        self.metrics_file = os.path.join(self.save_path, f"{self.model_name}_metrics.csv")

        # Create DataLoader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                     shuffle=False) if self.val_dataset else None
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                      shuffle=False) if self.test_dataset else None

        # Ensure save directory exists
        os.makedirs(self.save_path, exist_ok=True)

        # Initialize metrics CSV
        init_metrics_file(self.metrics_file)

        self.id2label = {
            0: 'anger',
            1: 'disgust',
            2: 'fear',
            3: 'joy',
            4: 'neutral',
            5: 'sadness',
            6: 'surprise'
        }


    def save_metrics(self, epoch, phase, metrics):
        """Save metrics for a given phase (train, val, or test) to the CSV file."""
        df = pd.read_csv(self.metrics_file)
        new_row = pd.DataFrame({
            'epoch': [epoch],
            'phase': [phase],
            'loss': [metrics.get('loss', 'N/A')],
            'accuracy': [metrics['overall_accuracy']],
            'weighted_f1': [metrics['weighted_f1']],
            'accuracy_anger': [metrics['per_class_accuracy'][0]],
            'accuracy_disgust': [metrics['per_class_accuracy'][1]],
            'accuracy_fear': [metrics['per_class_accuracy'][2]],
            'accuracy_joy': [metrics['per_class_accuracy'][3]],
            'accuracy_neutral': [metrics['per_class_accuracy'][4]],
            'accuracy_sadness': [metrics['per_class_accuracy'][5]],
            'accuracy_surprise': [metrics['per_class_accuracy'][6]],
            'f1_anger': [metrics['per_class_f1'][0]],
            'f1_disgust': [metrics['per_class_f1'][1]],
            'f1_fear': [metrics['per_class_f1'][2]],
            'f1_joy': [metrics['per_class_f1'][3]],
            'f1_neutral': [metrics['per_class_f1'][4]],
            'f1_sadness': [metrics['per_class_f1'][5]],
            'f1_surprise': [metrics['per_class_f1'][6]]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.metrics_file, index=False)

    def compute_metrics_old(self, predictions, labels, mask):
        """Compute accuracy and macro F1 score, ignoring padded labels."""
        all_preds = []
        all_labels = []

        for pred_batch, label_batch, mask_batch in zip(predictions, labels, mask):
            for pred, label, m in zip(pred_batch, label_batch, mask_batch):
                # if either mask code is 0 or label is -1, skip the sample
                if m == 1 and label != -1:
                    all_preds.append(pred)
                    all_labels.append(label)

        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        return {'accuracy': accuracy, 'weighted_f1': macro_f1}

    def compute_metrics(self, predictions, labels, mask, num_classes=7):
        """
        Compute overall accuracy, macro F1 score, and per-class accuracy and F1 scores.

        Args:
            predictions (list of list of int): Predicted labels for each utterance in each dialogue.
            labels (list of list of int): True labels for each utterance in each dialogue.
            mask (list of list of int): Mask indicating valid utterances (1 = valid, 0 = padded).
            num_classes (int): Number of emotion classes.

        Returns:
            dict: A dictionary containing overall accuracy, macro F1, and per-class accuracies and F1-scores.
        """
        all_preds = []
        all_labels = []

        # Batch
        for preds_batch, labels_batch, masks_batch in zip(predictions, labels, mask):
            if isinstance(masks_batch[0], Tensor):
                # print("wrap the result for single dia in batch (from metrics computation)")
                # If it's a list of utterances, wrap it inside another list to treat it as a single dialogue batch
                # Loader seems to change every string to a string wraped by ()
                # preds_batch = [preds_batch]
                labels_batch = [labels_batch]
                masks_batch = [masks_batch]
            else:
                print("DEBUG:")
                print(masks_batch)

            # Dialogue in batch
            for preds_dia, labels_dia, masks_dia in zip(preds_batch, labels_batch, masks_batch):
                # Utterance in dialouge
                # they are all tensor, IDK whyy
                for pred, label, m in zip(preds_dia, labels_dia, masks_dia):
                    # print(pred)
                    # print(label)
                    label = label.item()
                    m = m.item()

                    # print(f"pred {pred} \nlabel {label} \nm {m}")
                    # if either mask code is 0 or label is -1, skip the sample
                    if m == 1 and label != -1:
                        all_preds.append(pred)
                        all_labels.append(label)



        # Compute overall accuracy and macro F1
        accuracy = accuracy_score(all_labels, all_preds)
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

        # Compute per-class F1 scores, precision, recall
        classification_rep = classification_report(all_labels, all_preds, labels=np.arange(num_classes),
                                                   output_dict=True)

        # Extract per-class accuracy and F1
        per_class_f1 = [0]*num_classes
        per_class_accuracy = [0]*num_classes
        per_class_support = [0]*num_classes

        for i in range(num_classes):
            emotion_stats = classification_rep.get(str(i))

            per_class_f1[i] = emotion_stats.get('f1-score')

            # Compute accuracy for this class
            class_mask = int(emotion_stats.get('support'))
            if np.sum(class_mask) > 0:  # Avoid division by zero
                true_class_mask = np.array(all_labels) == i
                per_class_accuracy[i] = np.sum(np.array(all_preds)[true_class_mask] == i) / np.sum(true_class_mask)
            else:
                per_class_accuracy[i] = 0.0

            per_class_support[i] = class_mask

        metrics = {
            'overall_accuracy': accuracy,
            'weighted_f1': weighted_f1,
            'per_class_f1': per_class_f1,
            'per_class_accuracy': per_class_accuracy,
            'per_class_support': per_class_support
        }

        return metrics

    def save_predictions(self, predictions, labels, mask, dialogue_id, dialogues):
        df = pd.DataFrame(columns=['dialogue_id', 'utterance_id', 'utterance', 'label', 'prediction'])

        # Each Batch
        for pred_batch, label_batch, mask_batch, id_batch, dia_batch in zip(predictions, labels, mask, dialogue_id,
                                                                            dialogues):

            # print(f"DEBUGGG0: {pred_batch}")
            # print(f"DEBUGGG1: {label_batch}")
            # print(f"DEBUGGG2: {dia_batch}")
            # wrapping for single dialogues
            if isinstance(dia_batch[0][0], str):
                # If it's a list of utterances, wrap it inside another list to treat it as a single dialogue batch
                # Loader seems to change every string to a string wraped by ()
                # print("Wrap the single dialogue batch (from saving predictions)")
                # pred_batch = [pred_batch]
                label_batch = [label_batch]
                mask_batch = [mask_batch]
                id_batch = [id_batch]
                dia_batch = [dia_batch]

            # Each Dialogue in Batch
            for pred_dia, label_dia, m_dia, d_id, dia in zip(pred_batch, label_batch, mask_batch, id_batch, dia_batch):
                utt_id = 0
                # Each utterance in dialogue
                for pred_utt, label_utt, m_utt, utt in zip(pred_dia, label_dia, m_dia, dia):
                    # if either mask code is 0 or label is -1, skip the sample
                    # print(f"pred: {pred_utt}\nlabel: {label_utt}\nmask: {m_utt}\nutt:{utt}")
                    label_utt = label_utt.item()
                    m_utt = m_utt.item()

                    if m_utt == 1 and label_utt != -1:
                        utt_id += 1
                        # create a new line in dataframe
                        temp_df = pd.DataFrame({
                            'dialogue_id': d_id,
                            'utterance_id': utt_id,
                            'utterance': utt,
                            'label': self.id2label.get(label_utt),
                            'prediction': self.id2label.get(pred_utt)
                        })
                        # print(temp_df)
                        df = pd.concat([df, temp_df], ignore_index=True)
        df.to_csv(os.path.join(self.save_path, f"{self.model_name}_predictions.csv"), index=False)

    def evaluate(self, data_loader):
        """Evaluate the model on a given dataset(validation/test)."""
        all_predictions = []
        all_labels = []
        all_masks = []
        all_dia_id = []
        all_dia = []

        with torch.no_grad():
            for batch_idx, batch_content in enumerate(data_loader):
                print(f'batch {batch_idx}/{len(data_loader)-1}')
                dialogues, labels, mask, dialogue_id = batch_content
                predictions = self.model.predict(dialogues.copy())

                all_predictions.append(predictions)
                all_labels.append(labels)
                all_masks.append(mask)
                all_dia_id.append(dialogue_id)
                all_dia.append(dialogues)

        # calculate metrics
        metrics = self.compute_metrics(all_predictions.copy(), all_labels.copy(), all_masks.copy())
        # record predictions and labels in a csv
        self.save_predictions(predictions=all_predictions.copy(),
                              labels=all_labels.copy(),
                              mask=all_masks.copy(),
                              dialogue_id=all_dia_id.copy(),
                              dialogues=all_dia.copy())

        return metrics

    def train_epoch(self):
        """Train the model for one epoch."""

        all_predictions = []
        all_labels = []
        all_masks = []

        for batch_idx, batch_content in enumerate(self.train_loader):
            print(f'batch {batch_idx}/{len(self.train_loader)}')
            dialogues, labels, mask, dialogue_id = batch_content
            predictions = self.model.predict(dialogues)

            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_masks.extend(mask)

        metrics = self.compute_metrics(all_predictions, all_labels, all_masks)
        return metrics

    def train(self, epochs=1):
        """Run the training and evaluation process."""
        for epoch in range(epochs):
            # # Train for one epoch
            # train_metrics = self.train_epoch()
            # print(f"Epoch {epoch+1}/{epochs} Training Metrics: {train_metrics}")
            # self.save_metrics(epoch, 'train', train_metrics)
            #
            # # Validate the model
            # if self.val_loader:
            #     val_metrics = self.evaluate(self.val_loader)
            #     print(f"Epoch {epoch+1}/{epochs} Validation Metrics: {val_metrics}")
            #     self.save_metrics(epoch, 'val', val_metrics)

            # Test the model (optional)
            if self.test_loader:
                test_metrics = self.evaluate(self.test_loader)
                print(f"Epoch {epoch + 1}/{epochs} \nTest Metrics: {test_metrics}")
                self.save_metrics(epoch, 'test', test_metrics)

        print("Training Complete.")
