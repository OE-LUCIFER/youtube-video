import json

# Load the JSON data from the file
with open('dataset.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Open the output file in write mode
with open('dataset.jsonl', 'w', encoding='utf-8') as file:
    # Iterate over each item in the data list
    for item in data:
        # Convert the item to a JSON string and write it to the file
        file.write(json.dumps(item, ensure_ascii=False) + '\n')