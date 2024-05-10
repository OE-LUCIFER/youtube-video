from typing import List, Dict, Union
import json
from webscout.AI import OPENGPT

def main() -> None:
    # Initialize OPENGPT with conversation mode, max tokens, and timeout
    Databuilder = OPENGPT(is_conversation=False, max_tokens=8000, timeout=30, assistant_id="bca37014-6f97-4f2b-8928-81ea8d478d88")

    print("Starting the databuilder. Type '/bye' to end the conversation.")

    while True:
        # Get user input
        user_input: str = input(">>> ")

        # Exit condition
        if user_input.lower() == '/bye':
            break

        # Send the message to the model and get the response
        response: str = Databuilder.chat(user_input)
        
        # Prepare the data to append to dataset.json
        data: Dict[str, Union[str, List[Dict[str, str]]]] = {
            "input": "",
            "output": response,
            "instruction": user_input
        }

        # Load existing data from dataset.json with ensure_ascii=False
        try:
            with open('dataset.json', 'r', encoding='utf-8') as file:
                try:
                    interactions: List[Dict[str, Union[str, str]]] = json.load(file)
                except json.JSONDecodeError:
                    # If the file is empty or not valid JSON, initialize interactions as an empty list
                    interactions = []
        except FileNotFoundError:
            interactions = []

        # Ensure interactions is a list
        if not isinstance(interactions, list):
            interactions = []

        # Append the new interaction
        interactions.append(data)

        # Save the updated data back to dataset.json with ensure_ascii=False
        with open('dataset.json', 'w', encoding='utf-8') as file:
            json.dump(interactions, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
