# Initialize the GPT assistant predictor
from dotenv import load_dotenv

from Code.Model.GPT_connector import GPTAssistantPredictor

load_dotenv('../proj_key.env')
gpt_assistant = GPTAssistantPredictor()

# Example batch of dialogues (list of utterances)
dialogues = [
    [("How are you?",), ("I'm fine, thanks.",), ("What about you?",)],
    [("I'm excited about this project!",), ("Let's get started.",)]
]

# Get predictions
predictions = gpt_assistant.predict(dialogues)

# Output will be a list of lists, with each list containing the predicted emotion labels for each dialogue
print(predictions)
