from Dataset.MELD_Dataset_dia_emo_glove import MELDDataset_dia_emo_glove
from Model.Classifier_dia_emo_glove import TextCNNLSTM
from Train.trainer_dia_emo_glove import Trainer_dia_emo_glove
import os

if __name__ == "__main__":
    model_name = 'TextCNNLSTM_dia_emo_glove'
    data_path = '../data/text2/'
    glove_file_path = '../glove.6B/glove.6B.100d.txt'
    save_path = './text/'
    cache_dir = '../Cache/'

    csv_name = model_name

    os.makedirs(save_path, exist_ok=True)

    train_dataset = MELDDataset_dia_emo_glove(data_path, glove_file_path, split='train',cache_dir=cache_dir)
    val_dataset = MELDDataset_dia_emo_glove(data_path, glove_file_path, split='dev',cache_dir=cache_dir)
    test_dataset = MELDDataset_dia_emo_glove(data_path, glove_file_path, split='test',cache_dir=cache_dir)

    vocab_size = len(train_dataset.vocab)
    embedding_dim = 100
    num_filters = 100  # 你可以根据需要调整这个参数
    filter_sizes = [3, 4, 5]
    lstm_hidden_dim = 300
    output_dim = train_dataset.num_classes

    model = TextCNNLSTM(vocab_size, embedding_dim, train_dataset.embedding_matrix, num_filters, filter_sizes, lstm_hidden_dim, output_dim)

    trainer = Trainer_dia_emo_glove(model, train_dataset, val_dataset, test_dataset, batch_size=4, learning_rate=0.0001, model_name=model_name, csv_name=csv_name, save_path=save_path)
    trainer.train(epochs=20)
