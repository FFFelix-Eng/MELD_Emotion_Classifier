import os
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def preprocess_text(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Tokenize utterances and turn to lowercase
    splits = ['train', 'dev', 'test']
    for split in splits:
        input_file = os.path.join(data_dir,split, f'{split}_sent_emo.csv')
        df = pd.read_csv(input_file)
        
        # Clean text and tokenize using NLTK
        df['Utterance'] = df['Utterance'].apply(lambda x: word_tokenize(re.sub(r'[^a-zA-Z0-9\s]', '', x).lower()))
        
        output_file = os.path.join(output_dir, f'{split}.csv')
        df.to_csv(output_file, index=False)
        print(f"Processed {split} data saved to {output_file}")
    
if __name__ == "__main__":
    preprocess_text(data_dir=f'../../MELD_data/MELD_Raw', 
                    output_dir=f'../../data/text/')
    print('Finshed')






