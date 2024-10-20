import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from torch import Tensor


def init_metrics_file(metrics_file):
    # Initialize the metrics CSV file
    df = pd.DataFrame(columns=['epoch', 'phase', 'loss',
                               'accuracy',
                               'weighted_f1',
                               'accuracy_anger',
                               'accuracy_disgust',
                               'accuracy_fear',
                               'accuracy_joy',
                               'accuracy_neutral',
                               'accuracy_sadness',
                               'accuracy_surprise',
                               'f1_anger',
                               'f1_disgust',
                               'f1_fear',
                               'f1_joy',
                               'f1_neutral',
                               'f1_sadness',
                               'f1_surprise'])
    df.to_csv(metrics_file, index=False)

def save_metrics(metrics_file, epoch, phase, metrics):
    """Save metrics for a given phase (train, val, or test) to the CSV file."""
    df = pd.read_csv(metrics_file)
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
    df.to_csv(metrics_file, index=False)

def compute_metrics(predictions, labels, mask, num_classes=7):
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

        # Dialogue in batch
        for preds_dia, labels_dia, masks_dia in zip(preds_batch, labels_batch, masks_batch):
            # Utterance in dialouge
            # they are all tensor, IDK why
            for pred, label, m in zip(preds_dia, labels_dia, masks_dia):

                # For the compatibility, check if there are wrapping for each element
                if isinstance(label, Tensor):
                    label = label.item()
                if isinstance(m, Tensor):
                    m = m.item()

                # print(f"pred {pred} \nlabel {label} \nm {m}")
                # if either mask code is 0 or label is -1, skip the sample
                if m == 1:
                    # print(pred)
                    # print(label)
                    all_preds.append(pred)
                    all_labels.append(label)

    # Compute overall accuracy and macro F1
    accuracy = accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    # Compute per-class F1 scores, precision, recall
    classification_rep = classification_report(all_labels, all_preds, labels=np.arange(num_classes),
                                               output_dict=True)

    # Extract per-class accuracy and F1
    per_class_f1 = [0] * num_classes
    per_class_accuracy = [0] * num_classes
    per_class_support = [0] * num_classes

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
