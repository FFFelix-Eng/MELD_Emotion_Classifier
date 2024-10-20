import pandas as pd

def compute_dialogue_length_distribution(csv_file_path):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Ensure the 'Dialogue_ID' column exists
    if 'Dialogue_ID' not in df.columns:
        raise ValueError("The CSV file must contain a 'Dialogue_ID' column.")

    # Group the data by 'Dialogue_ID' and compute the number of utterances per dialogue
    dialogue_lengths = df.groupby('Dialogue_ID').size()

    # Compute the distribution of dialogue lengths
    dialogue_length_distribution = dialogue_lengths.value_counts().sort_index()

    # Print the distribution of dialogue lengths
    print("Dialogue Length Distribution:")
    print(dialogue_length_distribution)

    # Print basic statistics
    print(f"\nTotal dialogues: {len(dialogue_lengths)}")
    print(f"Mean dialogue length: {dialogue_lengths.mean()}")
    print(f"Max dialogue length: {dialogue_lengths.max()}")
    print(f"Min dialogue length: {dialogue_lengths.min()}")

# Example usage
csv_file_path = '../data/text2/train.csv'
compute_dialogue_length_distribution(csv_file_path)

