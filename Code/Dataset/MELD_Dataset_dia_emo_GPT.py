import pandas as pd
from torch.utils.data import Dataset

class MELDDatasetGPT(Dataset):
    def __init__(self, data_path, split='train', max_dialogue_len=40, max_utterance_len=330):
        self.label2id = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'joy': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6
        }

        self.data_path = data_path
        self.split = split
        self.max_dialogue_len = max_dialogue_len
        self.max_utterance_len = max_utterance_len

        # Load dataset into a DataFrame
        self.data = pd.read_csv(f'{data_path}/{split}.csv')

        # Group data by dialogues
        self.dialogues = self.data.groupby('Dialogue_ID')

        # Create a dialogue list for proper indexing
        self.dialogue_list = list(self.dialogues.groups.keys())

    def __len__(self):
        return len(self.dialogue_list)

    # def __len__(self):
    #     # Limit to the first 5 dialogues, for test
    #     return min(5, len(self.dialogue_list))

    def __getitem__(self, idx):
        # Get the dialogue by index from the dialogue list
        dialogue_id = self.dialogue_list[idx]
        dialogue_data = self.dialogues.get_group(dialogue_id)

        # Extract the utterances
        utterances = dialogue_data['Utterance'].tolist()

        # Trim each utterance to max_utterance_len
        extended_trimmed_utterances = self.extend_trimmed_utterances(utterances)

        # Extract emotion labels and pad to max_dialogue_len
        labels = [self.label2id[label] for label in dialogue_data['Emotion']]
        padded_labels = labels + [-1] * (self.max_dialogue_len - len(labels))  # Assuming -1 is the padding label

        # Generate attention mask for valid utterances (1 for valid utterances, 0 for padding)
        attention_mask = [1] * len(labels) + [0] * (self.max_dialogue_len - len(labels))

        return extended_trimmed_utterances, padded_labels, attention_mask, dialogue_id

        # return trimmed_utterances, padded_labels, attention_mask, dialogue_id

    def trim_utterance(self, utterance):
        """
        Trim utterances to the max_utterance_len. If the utterance is shorter than
        max_utterance_len, pad it with spaces.
        """
        # Truncate the utterance to the maximum allowed length
        trimmed_utterance = utterance[:self.max_utterance_len]

        # If the trimmed utterance is shorter than the max length, pad it with spaces
        if len(trimmed_utterance) < self.max_utterance_len:
            num_spaces_to_add = self.max_utterance_len - len(trimmed_utterance)
            trimmed_utterance += ' ' * num_spaces_to_add

        # Ensure the trimmed utterance has exactly max_utterance_len characters
        if len(trimmed_utterance) != self.max_utterance_len:
            raise ValueError(f"Error: The trimmed utterance does not have the required "
                             f"{self.max_utterance_len} characters. Actual length: "
                             f"{len(trimmed_utterance)} characters.")

        return trimmed_utterance

    def extend_trimmed_utterances(self, utterances):
        """
        Extend the list of trimmed utterances to reach max_dialogue_len with empty strings.
        """
        # Trim each utterance in the list
        trimmed_utterances = [self.trim_utterance(utt) for utt in utterances]

        # Calculate how many empty strings need to be added
        num_empty_to_add = self.max_dialogue_len - len(trimmed_utterances)

        if num_empty_to_add > 0:
            trimmed_utterances.extend([' ' * self.max_utterance_len] * num_empty_to_add)

        # Ensure the list has exactly max_dialogue_len elements
        if len(trimmed_utterances) != self.max_dialogue_len:
            raise ValueError(f"Error: The list of trimmed utterances does not have the required "
                             f"{self.max_dialogue_len} elements. Actual length: "
                             f"{len(trimmed_utterances)} elements.")

        return trimmed_utterances