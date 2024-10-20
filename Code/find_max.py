# Re-upload the file to read it again correctly from the system
import pandas as pd

# Define the function to calculate max lengths
def find_max_lengths(file_path, epoch):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)

    # Group data by 'Dialogue_ID'
    grouped = data.groupby('Dialogue_ID')

    # Find the maximum number of utterances in a dialogue
    max_dialogue_len = grouped.size().max()

    # Find the maximum number of words in an utterance
    max_utterance_len = data['Utterance'].apply(lambda x: len(x)).max()

    print(f"{epoch}: The max dialogue length is {max_dialogue_len}, The max utterance length is {max_utterance_len}")

# Call the function using an example path
file_path = '../data/text2/train.csv'
find_max_lengths('../data/text2/train.csv', 'train')
find_max_lengths('../data/text2/dev.csv', 'dev')
find_max_lengths('../data/text2/test.csv', 'test')
