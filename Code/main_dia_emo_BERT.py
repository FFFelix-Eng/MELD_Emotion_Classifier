from Dataset.MELD_Dataset_dia_emo_BERT import MELDDataset_dia_emo
from Model.Classifier3_dia_emo_BERT import DialogueClassifier
from Train.trainer_dia_emo_BERT import Trainer_dia_emo
import os

if __name__ == "__main__":
    model_name = 'BERT1_dia_emo_3'
    cache_dir = '../Cache/'
    save_path = './text/'
    data_path = '../data/text2/'

    csv_name=model_name

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    bert_model_name = 'bert-base-uncased'
    train_dataset = MELDDataset_dia_emo(data_path, bert_model_name, split='train', cache_dir=cache_dir)
    val_dataset = MELDDataset_dia_emo(data_path, bert_model_name, split='dev', cache_dir=cache_dir)
    test_dataset = MELDDataset_dia_emo(data_path, bert_model_name, split='test', cache_dir=cache_dir)

    hidden_dim = 768 # same as BERT embedding_len
    output_dim = 7  # Number of emotion classes

    model = DialogueClassifier(bert_model_name, hidden_dim, output_dim, cache_dir=cache_dir)

    trainer = Trainer_dia_emo(model, train_dataset, val_dataset, test_dataset, batch_size=4, learning_rate=0.0001,
                              model_name=model_name, save_path=save_path, csv_name=csv_name)
    trainer.train(epochs=20)
