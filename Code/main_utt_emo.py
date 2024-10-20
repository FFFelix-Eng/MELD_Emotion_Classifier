from Dataset.MELD_Dateset_utt_emo import MELDDataset
from Model.Classifier1_emo import BERTEmotionClassifier_utt_emo
from Dataset.MELD_Dateset_utt_emo import MELDDataset
from Train.trainer_utt_emo import Trainer
import os

if __name__ == "__main__":
    
    model_name = 'BERT1_emo'
    cache_dir = '../Cache/'
    save_path = './text/'
    data_path = '../data/text/'

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    bert_model_name = 'bert-base-uncased'
    train_dataset = MELDDataset(data_path, bert_model_name, split='train', cache_dir=cache_dir)
    val_dataset = MELDDataset(data_path, bert_model_name, split='dev', cache_dir=cache_dir)
    test_dataset = MELDDataset(data_path, bert_model_name, split='test', cache_dir=cache_dir)
    
    hidden_dim = 128
    output_dim = train_dataset.num_classes  # Number of emotion classes

    model = BERTEmotionClassifier_utt_emo(bert_model_name, hidden_dim, cache_dir=cache_dir)
    
    trainer = Trainer(model, train_dataset, val_dataset, test_dataset, batch_size=32, learning_rate=0.001, model_name=model_name)
    trainer.train(epochs=10)
