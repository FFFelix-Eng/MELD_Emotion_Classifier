import os

from dotenv import load_dotenv

from Dataset.MELD_Dataset_dia_emo_GPT import MELDDatasetGPT
from Model.GPT_connector import GPTAssistantPredictor
from Train.trainer_dia_emo_GPT import Trainer

if __name__ == "__main__":
    model_name = 'GPT_dia_emo'
    data_path = '../data/text2/'  # Path to your data directory
    save_path = './models/'  # Path to save the trained model and metrics

    load_dotenv('./proj_key.env')

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Dataset parameters
    max_dialogue_len = 40  # Set the maximum dialogue length (can be adjusted based on your dataset analysis)
    max_utterance_len = 80  # Set the maximum utterance length (adjust based on your dataset)

    # Load datasets
    train_dataset = MELDDatasetGPT(data_path, split='train', max_dialogue_len=max_dialogue_len, max_utterance_len=max_utterance_len)
    val_dataset = MELDDatasetGPT(data_path, split='dev', max_dialogue_len=max_dialogue_len, max_utterance_len=max_utterance_len)
    test_dataset = MELDDatasetGPT(data_path, split='test', max_dialogue_len=max_dialogue_len, max_utterance_len=max_utterance_len)

    # Initialize GPT predictor
    gpt_predictor = GPTAssistantPredictor(model="gpt-4o-mini", max_dialogue_len=max_dialogue_len)

    # Initialize the Trainer class
    trainer = Trainer(
        model=gpt_predictor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=1,
        save_path=save_path,
    )

    # Start the training loop (specify number of epochs)
    trainer.train(epochs=1)

