import os
import time

from openai import OpenAI, RateLimitError
from dotenv import load_dotenv


class GPTAssistantPredictor:
    """
    This is a predictor that connect to an OPENAI GPT assistant
    All the conversations with assistants are single message
    The method is old and please check GPT_connector_new
    """
    def __init__(self, model="gpt-4o", max_dialogue_len=40):
        """
        Args:
            model (str): The model version to use, e.g., 'gpt-4'.
            max_dialogue_len (int): Maximum dialogue length for padding.
        """
        # Load the environment variables from the .env file
        self.api_key = os.getenv("OPENAI_API_KEY")

        if self.api_key is None:
            raise ValueError("OpenAI API key not found. Make sure to set OPENAI_API_KEY in the .env file.")

        # Initialize the OpenAI client with the API key
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_dialogue_len = max_dialogue_len
        self.assistant = None

    def predict(self, dialogues):
        """
        Transforms the batch of dialogues to a prompt, sends it to GPT, and reverses the predictions into a usable format.

        Args:
            dialogues (list of list of str): A batch of dialogues, where each dialogue is a list of utterances (str).

        Returns:
            predictions (list of list of int): Predicted emotion labels for each utterance in each dialogue.
                                               Shape (B, dia_len), padded with -1 for empty utterances.
        """
        # Step 1: Transform dialogues into a prompt format
        if isinstance(dialogues[0][0], str):
            # If it's a list of utterances, wrap it inside another list to treat it as a single dialogue batch
            # Loader seems to change every string to a string wraped by ()
            dialogues = [dialogues]

        prompt = self.dialogues_to_prompt(dialogues)

        # Step 2: Send the prompt to GPT using the chat completion API
        gpt_response = self.send_to_gpt(prompt)
        print(gpt_response+"\n\n")

        # Step 3: Parse the response and convert it into predictions
        predictions = self.response_to_predictions(gpt_response, dialogues)

        return predictions

    def dialogues_to_prompt(self, dialogues):
        """
        Converts the dialogues into a clear, structured prompt that specifies the task and lists possible emotions.

        Args:
            dialogues (list of list of str): A batch of dialogues.

        Returns:
            str: The prompt for GPT.
        """

        prompt = (
            "You are an emotion classifier. For each utterance in the following dialogues, "
            "classify the emotion of the speaker. The possible emotions are: "
            "'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'.\n\n"
        )

        prompt += (
            "For each dialogue, respond with a list of the predicted emotion for each utterance in the dialogue.\n"
            "Use the exact format: ['emotion1', 'emotion2', ...]. For example:\n\n"
            "(Input would be:)\n"
            "Dialogue 1:\n"
            "Utterance 1: \"How are you?\"\n"
            "Utterance 2: \"I'm fine.\"\n"
            "Dialogue 2: \n"
            "Utterance 1: \"I'm not good.\"\n\n"
            "(Your output would be:)\n"
            "Dialogue 1: ['neutral', 'joy']\n"
            "Dialogue 2: ['sadness']\n\n"
        )

        prompt += "Now, classify the emotions for the following dialogues:\n"

        for i, dialogue in enumerate(dialogues):
            prompt += f"Dialogue {i + 1}:\n"
            for j, utterance in enumerate(dialogue):
                prompt += f"Utterance {j + 1}: \"{utterance}\"\n"
            prompt += "\n"

        print(prompt)
        return prompt

    def send_to_gpt(self, prompt, max_retries=20):
        """
        Sends the prompt to the GPT chat completion API with retry logic for rate limits.

        Args:
            prompt (str): The prompt string for GPT.
            max_retries (int): Maximum number of retries in case of rate limit errors.

        Returns:
            str: The response from GPT.
        """
        retries = 0
        while retries < max_retries:
            try:
                # This part is from their github quickstart
                response = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=self.model,
                )
                return response.choices[0].message.content.strip()

            except RateLimitError as e:
                retries += 1
                wait_time = 2 ** retries  # Exponential backoff: 2, 4, 8, 16, etc. seconds
                print(f"Rate limit reached. Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                time.sleep(wait_time)

        raise Exception(f"Failed after {max_retries} retries due to rate limit.")

    def response_to_predictions(self, gpt_response, dialogues):
        """
        Parses the GPT response and converts it into predictions of shape (B, dia_len).

        Args:
            gpt_response (str): The text response from GPT.
            dialogues (list of list of str): The original dialogue structure.

        Returns:
            list of list of int: Predicted emotion labels for each utterance in each dialogue (padded with -1 for empty utterances).
        """
        # Debug print to inspect the GPT response
        # print("GPT Response:\n", gpt_response)

        label2id = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'joy': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6
        }

        predictions = []
        # Split the GPT response into lines and filter for lines starting with "Dialogue"
        response_lines = [line for line in gpt_response.split('\n') if line.startswith("Dialogue")]

        # Check if the response_lines are correct
        if len(response_lines) == 0:
            print("Warning: No 'Dialogue' found in the response.")
            print("RESPONSE: "+gpt_response)
            return [[-1] * self.max_dialogue_len for _ in dialogues]

        for i, dialogue in enumerate(dialogues):
            dialogue_predictions = []

            try:
                # Extract the emotion list from the corresponding dialogue prediction
                predicted_emotions_str = response_lines[i].split(": ")[1].strip("[]").replace("'", "")
                predicted_emotions = predicted_emotions_str.split(', ')

                # Map each emotion to its corresponding integer
                for j, emotion in enumerate(predicted_emotions):
                    dialogue_predictions.append(label2id.get(emotion, -1))  # Map label to ID or use -1 for unknown
            except (IndexError, ValueError) as e:
                print(f"Warning: Issue with parsing response for dialogue {i}. Skipping this dialogue. Error: {e}")
                print(f'Dialogue: {dialogues}')
                dialogue_predictions = [-1] * self.max_dialogue_len

            # Pad the predictions for short dialogues
            if len(dialogue_predictions) < self.max_dialogue_len:
                dialogue_predictions += [-1] * (self.max_dialogue_len - len(dialogue_predictions))

            predictions.append(dialogue_predictions)

        return predictions
