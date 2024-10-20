import json
import os
import pickle
import time

from openai import OpenAI, RateLimitError
from dotenv import load_dotenv


class GPTAssistantPredictor_new:
    """
    This is a predictor that connect to an OPENAI GPT assistant
    For each batch of input, the predictor creates a new thread and delete it after use
    The assistant thus has no context memory but only the description (and system prompt)


    """
    def __init__(self, model="gpt-4o-mini", max_dialogue_len=40):
        """
        Args:
            model (str): The model version to use, e.g., 'gpt-4'.
            max_dialogue_len (int): Maximum dialogue length for padding.
        """

        # Initialize the OpenAI client (no need to include the key explicitly)
        self.client = OpenAI()
        self.model = model
        self.max_dialogue_len = max_dialogue_len
        self.assistant = None
        self.instruction = None

    def generate_instruction(self, addon=None):
        instruction = (
            "You are an emotion classifier. For each utterance in the following dialogues, "
            "classify the emotion of the speaker. The possible emotions are: "
            "'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'.\n\n"
        )

        instruction += (
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

        if addon:
            instruction += addon

        self.instruction = instruction

    def save_assistant_pickle(self, path):
        """
        Save the assistant object using pickle.

        Args:
            path (str): The file path where the assistant will be saved.
        """
        with open(path, 'wb') as file:
            pickle.dump(self.assistant, file)
        print(f"Assistant saved to {path}")

    def load_assistant_pickle(self, path):
        """
        Load the assistant object using pickle.

        Args:
            path (str): The file path from where the assistant will be loaded.
        """
        if os.path.exists(path):
            with open(path, 'rb') as file:
                self.assistant = pickle.load(file)
            print(f"Assistant loaded from {path}")
        else:
            print(f"No assistant found at {path}")

    def get_or_create_assistant(self, path):
        """
        Read an existing assistant from a pickle file or create a new one and assign it with the instruction, then save it.

        Args:
            path (str): The file path of the assistant object saved using pickle.
            instruction (str): The instruction used to create a new assistant if it doesn't exist.
        """

        instruction = self.instruction
        if os.path.exists(path):
            # Load the assistant using pickle
            self.load_assistant_pickle(path)
            print(f"The assistant exists and has been loaded ({self.assistant.id}).\n")
        else:
            # Create a new assistant
            self.assistant = self.client.beta.assistants.create(
                name="Custom Assistant",
                instructions=instruction,
                model=self.model
            )
            # Save the assistant using pickle
            self.save_assistant_pickle(path)
            print(f"Did not find exising assistant, created a new one: {self.assistant.id}\n")

    def update_assistant_instruction(self, path, new_instruction):
        """
        Update the instruction of an existing assistant or report an error if the file doesn't exist.

        Args:
            path (str): The file path of the assistant object saved using pickle.
            new_instruction (str): The new instruction for the assistant.

        Raises:
            FileNotFoundError: If the assistant file doesn't exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file at path '{path}' does not exist.")

        # Load the assistant from the file
        self.load_assistant_pickle(path)

        # Check if the current instruction matches the new one
        current_instruction = self.assistant.get("instructions", "")
        if current_instruction == new_instruction:
            print("The current instruction is the same as the new one. No update needed.")
            return

        # Update the instruction if different
        assistant_id = self.assistant['id']
        updated_assistant = self.client.beta.assistants.update(
            assistant_id=assistant_id,
            instructions=new_instruction
        )

        # Save the updated assistant
        self.assistant = updated_assistant
        self.save_assistant_pickle(path)
        print(f"Assistant instruction updated. New instructions: \n{self.assistant['instructions']}\n")



    def predict(self, dialogues):
        """
        Transforms the batch of dialogues to a content, sends it to GPT, and reverses the predictions into a usable format.

        Args:
            dialogues (list of list of str): A batch of dialogues, where each dialogue is a list of utterances (str).

        Returns:
            predictions (list of list of int): Predicted emotion labels for each utterance in each dialogue.
                                               Shape (B, dia_len), padded with -1 for empty utterances.
        """
        # Step 1: Transform dialogues into a content format
        if isinstance(dialogues[0][0], str):
            # If it's a list of utterances, wrap it inside another list to treat it as a single dialogue batch
            # Loader seems to change every string to a string wraped by ()
            dialogues = [dialogues]

        content = self.dialogues_to_prompt(dialogues)

        # Step 2: Send the content to GPT using the chat completion API
        gpt_response = self.one_time_thread_respond(content)
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

        content = "Classify the emotions for the following dialogues:\n"

        for i, dialogue in enumerate(dialogues):
            content += f"Dialogue {i + 1}:\n"
            for j, utterance in enumerate(dialogue):
                content += f"Utterance {j + 1}: \"{utterance[0]}\"\n"
            content += "\n"

        print(content)
        return content

    def one_time_thread_respond(self, content, max_retries=20):
        """
        Sends the prompt to the GPT chat completion API with retry logic for rate limits.
        Automatically deletes the thread after success or failure to avoid clutter.

        Args:
            content (str): The prompt string for GPT.
            max_retries (int): Maximum number of retries in case of rate limit errors.

        Returns:
            str: The response from GPT.
        """
        retries = 0

        try:
            # Create a new thread
            thread = self.client.beta.threads.create()
            while retries < max_retries:
                try:
                    # Add the message to this thread
                    message = self.client.beta.threads.messages.create(
                        thread_id=thread.id,
                        role="user",
                        content=content
                    )

                    # Create a run for the assistant to respond to the thread
                    run = self.client.beta.threads.runs.create_and_poll(
                        thread_id=thread.id,
                        assistant_id=self.assistant.id)

                    # search for the latest message
                    thread_messages = self.client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id)
                    last_msg = thread_messages.data[-1]


                    if last_msg:
                        return last_msg.content[0].text.value
                    else:
                        raise Exception(f"No result from the run!\nMessage: {message}")

                except RateLimitError as e:
                    retries += 1
                    wait_time = 2 ** retries  # Exponential backoff: 2, 4, 8, 16, etc. seconds
                    print(f"Rate limit reached. Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)

                raise Exception(f"Failed after {max_retries} retries due to rate limit.")

        # no matter the result in the first try, delete the thread
        finally:
            # Clean up and delete the thread after we're done
            try:
                self.client.beta.threads.delete(thread_id=thread.id)
                print(f"Thread {thread.id} deleted successfully.")
            except Exception as e:
                print(f"Error deleting thread {thread.id}: {e}")

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
