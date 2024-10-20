import json
import os
import pickle
import time

import torch
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv


class GPTEmbeddingGenerator:
    """
    This is a predictor that connect to an OPENAI GPT assistant
    For each batch of input, the predictor creates a new thread and delete it after use
    The assistant thus has no context memory but only the description (and system prompt)


    """
    def __init__(self, model="text-embedding-3-small", max_dialogue_len=40, utterance_embedding_len=1536):
        """
        Args:
            model (str): The model version to use, e.g., 'gpt-4'.
            max_dialogue_len (int): Maximum dialogue length for padding.
        """

        # Initialize the OpenAI client (no need to include the key explicitly)
        self.client = OpenAI()
        self.model = model
        self.max_dialogue_len = max_dialogue_len
        self.utterance_embedding_len = utterance_embedding_len


    # def get_single_embedding(self, text):
    #     text = text.replace("\n", " ")
    #     return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding

    def get_multi_embedding(self, texts, max_retries=20):

        # remove the tuple on strings
        if isinstance(texts[0], tuple):
            texts = [text[0] for text in texts]
        processed_texts = [text.replace("\n", " ") for text in texts]

        retries = 0
        while retries < max_retries:
            try:
                response = self.client.embeddings.create(input=processed_texts, model=self.model)
                embeddings = [item.embedding for item in response.data]
                return embeddings
            except RateLimitError as e:
                retries += 1
                wait_time = 2 ** retries  # Exponential backoff: 2, 4, 8, 16, etc. seconds
                print(f"Rate limit reached. Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                time.sleep(wait_time)
            raise Exception(f"Failed after {max_retries} retries due to rate limit.")

    def get_batch_embeddings(self, batch_list, max_dialogue_len=40):
        batch_embeddings = []

        if isinstance(batch_list[0], tuple):
            batch_list = [batch_list]

        for dialogue in batch_list:
            dialogue_embeddings = []
            embeddings = self.get_multi_embedding(dialogue)
            dialogue_embeddings.extend(embeddings)

            padding_num_of_utt = max_dialogue_len - len(dialogue_embeddings)
            if padding_num_of_utt > 0:
                padding = [[0.0] * self.utterance_embedding_len]*padding_num_of_utt
                dialogue_embeddings.extend(padding)

            batch_embeddings.append(dialogue_embeddings)

        embeddings_tensor = torch.tensor(batch_embeddings, dtype=torch.float32)


        return embeddings_tensor, len(batch_list), max_dialogue_len




    def dialogues_to_prompt(self, dialogues):
        """
        Converts the dialogues into a clear, structured prompt that specifies the task and lists possible emotions.

        Args:
            dialogues (list of list of str): A batch of dialogues.

        Returns:
            str: The prompt for GPT.
        """

        content = "Classify the emotions for the following dialogues:\n"

        for i, dialogue in enumerate(dialogues):
            content += f"Dialogue {i + 1}:\n"
            for j, utterance in enumerate(dialogue):
                content += f"Utterance {j + 1}: \"{utterance[0]}\"\n"
            content += "\n"

        print(content)
        return content


