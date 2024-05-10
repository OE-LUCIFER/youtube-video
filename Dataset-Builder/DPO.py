from typing import List, Dict, Union
import json
import time  # Import the time module for adding a delay
from webscout.AI import OPENGPT

def main() -> None:
    """
    Starts a conversation with the specified OPENGPT model.

    Returns:
        None
    """
    # Initialize OPENGPT with conversation mode, max tokens, and timeout
    opengpt_instance = OPENGPT(is_conversation=False, max_tokens=8000, timeout=30, assistant_id="bca37014-6f97-4f2b-8928-81ea8d478d88")

    print("Converting dataset to DPO dataset.")

    # Load existing data from feeling.json with ensure_ascii=False
    try:
        with open('dataset.json', 'r', encoding='utf-8') as file:
            interactions: List[Dict[str, Union[str, str]]] = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Error loading feelings.json")
        return

    processed_prompts = set()  # To store processed prompts and ensure uniqueness

    for interaction in interactions:
        # Get the instruction as the prompt
        prompt = interaction["instruction"]

        # Check if the prompt has already been processed
        if prompt in processed_prompts:
            continue

        # Get the output as the chosen response
        chosen = interaction["output"]

        # Send the prompt to the model and get the response
        response: str = opengpt_instance.chat(prompt)

        # Prepare the data to append to a new dataset
        DPO: Dict[str, Union[str, List[Dict[str, str]]]] = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": response
        }

        # Append the new interaction to a new dataset
        with open('dpo.json', 'a', encoding='utf-8') as file:
            json.dump(DPO, file, indent=4, ensure_ascii=False)
            file.write(",\n")
        
        # Add the processed prompt to the set
        processed_prompts.add(prompt)

        # Add a sleep to pause execution for 1 second (adjust as needed)
        time.sleep(1)  # 1 second delay between interactions

    print("Databuilder finished.")

if __name__ == "__main__":
    main()
