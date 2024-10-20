from dotenv import load_dotenv

from Dataset.MELD_Dataset_dia_emo_NEW import MELDDataset_NEW
from Model.Classifier_GPT_bcLSTM import DialogueClassifier
from Train.trainer_dia_emo_GPT_LSTM import Trainer_dia_emo
import os

if __name__ == "__main__":
    model_name = 'GPT_LSTM1'
    cache_dir = '../Cache/'
    save_path = './result/'
    data_path = '../data/text2/'

    csv_name=model_name

    # load_dotenv('proj_key.env')

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)


    max_dialogue_len = 40  # Set the maximum dialogue length (can be adjusted based on your dataset analysis)
    max_utterance_len = 80  # Set the maximum utterance length (adjust based on your dataset)
    hidden_dim = 768  # same as BERT embedding_len
    output_dim = 7  # Number of emotion classes

    def list_collate_fn(batch):
        collated_batch = {}
        for key in batch[0]:
            # for each component in the dataset, put them in a list of batch
            collated_batch[key] = [item[key] for item in batch]
        return collated_batch

    # Load datasets
    train_dataset = MELDDataset_NEW(data_path, split='train', max_dialogue_len=max_dialogue_len, max_utterance_len=max_utterance_len)
    val_dataset = MELDDataset_NEW(data_path, split='dev', max_dialogue_len=max_dialogue_len, max_utterance_len=max_utterance_len)
    test_dataset = MELDDataset_NEW(data_path, split='test', max_dialogue_len=max_dialogue_len, max_utterance_len=max_utterance_len)

    model = DialogueClassifier(hidden_dim=hidden_dim, output_dim=output_dim)

    trainer = Trainer_dia_emo(model, train_dataset, val_dataset, test_dataset, collate_custom=list_collate_fn, batch_size=4, learning_rate=0.0001,
                              model_name=model_name, save_path=save_path, csv_name=csv_name)
    trainer.train(epochs=5)
