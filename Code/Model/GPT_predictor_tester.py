# Initialize the GPT assistant predictor
from dotenv import load_dotenv

from Code.Model.GPT_predictor import GPTAssistantPredictor_new

load_dotenv('../proj_key.env')
gpt_assistant_save_path = ('../GPT_assistant_save/4o_mini_1.pkl')
gpt_assistant = GPTAssistantPredictor_new()
gpt_assistant.generate_instruction()
gpt_assistant.get_or_create_assistant(path=gpt_assistant_save_path)


# Example batch of dialogues (list of utterances)
dialogues = [
    [("How are you?",), ("I'm fine, thanks.",), ("What about you?",)],
    [("I'm excited about this project!",), ("Let's get started.",)]
]

# Get predictions
predictions = gpt_assistant.predict(dialogues)

# Output will be a list of lists, with each list containing the predicted emotion labels for each dialogue
print(predictions)
